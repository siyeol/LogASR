#!/usr/bin/env python3
"""Sequential OOD WER + silence-pack suite. Resumable; one GPU job at a time."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/SpeechPTQ")
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "ood"
LOGS = ROOT / "logs"
STATUS = RESULTS / "status.json"
SUMMARY = RESULTS / "SUMMARY.json"
SUITE_LOG = LOGS / "ood_suite.log"
PIDS_CUR = Path("/sys/fs/cgroup/pids.current")
PIDS_MAX = Path("/sys/fs/cgroup/pids.max")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTHONPATH", str(SRC))

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(SRC)
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"
ENV["TORCH_NUM_THREADS"] = "1"
ENV["TOKENIZERS_PARALLELISM"] = "false"
ENV["HF_DEACTIVATE_ASYNC_LOAD"] = "1"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with SUITE_LOG.open("a") as f:
        f.write(line + "\n")


def pids() -> tuple[int, int]:
    cur = int(PIDS_CUR.read_text().strip()) if PIDS_CUR.exists() else 0
    raw = PIDS_MAX.read_text().strip() if PIDS_MAX.exists() else "max"
    mx = 10**9 if raw == "max" else int(raw)
    return cur, mx


def wait_pid_headroom(free_need: int = 8, timeout: float = 180.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur, mx = pids()
        if mx - cur >= free_need:
            return
        log(f"waiting for PIDs ({cur}/{mx}, need {free_need} free)")
        time.sleep(15)
    cur, mx = pids()
    log(f"PID wait timed out ({cur}/{mx}); continuing anyway")


def write_status(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2))


def job_done(out: Path) -> bool:
    if not out.is_file():
        return False
    try:
        data = json.loads(out.read_text())
    except Exception:
        return False
    return "wer" in data


def run_cmd(cmd: list[str], log_path: Path, timeout: int | None = None) -> int:
    wait_pid_headroom(8)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}")
    with log_path.open("a") as lf:
        lf.write(f"\n===== {now()} =====\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=ENV,
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return int(proc.returncode)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def whisper_cmd(mode: str, dataset: str, out: Path, pack: str = "none", **extra) -> list[str]:
    split = "test-clean" if dataset == "librispeech" else "test"
    cmd = py(
        str(SRC / "run_eval.py"),
        "--mode",
        mode,
        "--dataset",
        dataset,
        "--split",
        split,
        "--pack",
        pack,
        "--batch-size",
        str(extra.get("bs", 8)),
        "--max-new-tokens",
        str(extra.get("max_new", 384)),
        "--out",
        str(out),
    )
    if dataset == "ami":
        cmd += ["--ami-stitch-sec", "30"]
    if dataset == "commonvoice":
        cmd += ["--lang", extra.get("lang", "all")]
    if mode == "w4a4":
        cmd += [
            "--w-mode",
            "log",
            "--a-mode",
            "log_token",
            "--scope",
            "enc_ffn",
            "--backend",
            "fake",
            "--group-size",
            "0",
        ]
    return cmd


def qwen_cmd(mode: str, dataset: str, out: Path, **extra) -> list[str]:
    split = "test-clean" if dataset == "librispeech" else "test"
    cmd = py(
        str(SRC / "run_eval_qwen3.py"),
        "--mode",
        mode,
        "--dataset",
        dataset,
        "--split",
        split,
        "--batch-size",
        str(extra.get("bs", 4)),
        "--max-new-tokens",
        str(extra.get("max_new", 384)),
        "--ami-stitch-sec",
        "30" if dataset == "ami" else "0",
        "--out",
        str(out),
    )
    if dataset == "commonvoice":
        cmd += ["--lang", extra.get("lang", "all")]
    if mode == "w4a4":
        cmd += [
            "--w-mode",
            "log",
            "--a-mode",
            "log_token",
            "--scope",
            "enc_ffn",
            "--backend",
            "fake",
        ]
    return cmd


def seed_existing() -> None:
    """Reuse already-finished AMI FP16 / packing JSONs."""
    mapping = {
        ROOT / "results" / "fp16_ami_test_stitch30.json": RESULTS / "whisper_fp16_ami.json",
        ROOT / "results" / "qwen3_fp16_ami_test_stitch30.json": RESULTS / "qwen_fp16_ami.json",
        ROOT / "results" / "fp16_ami_test_stitch30_pack_collapse.json": RESULTS
        / "whisper_pack_collapse_ami.json",
        ROOT
        / "results"
        / "qwen3_silence_collapse_ami_test_stitch30_hold1_db-35.json": RESULTS
        / "qwen_pack_collapse_ami.json",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for src, dst in mapping.items():
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            log(f"seeded {dst.name} from {src.name}")


def collect_summary() -> dict:
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in {"status.json", "SUMMARY.json"}:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rows.append(
            {
                "file": p.name,
                "dataset": d.get("dataset"),
                "mode": d.get("mode"),
                "pack": d.get("pack"),
                "wer": d.get("wer"),
                "wer_micro": d.get("wer_micro"),
                "wer_by_lang": d.get("wer_by_lang"),
                "n_utts": d.get("n_utts"),
                "audio_sec": d.get("audio_sec"),
                "audio_sec_out": d.get("audio_sec_out"),
                "enc_frames_in": d.get("enc_frames_in"),
                "enc_frames_out": d.get("enc_frames_out"),
                "enc_frames_saved_pct": d.get("enc_frames_saved_pct"),
                "mean_enc_frames_in": d.get("mean_enc_frames_in"),
                "mean_enc_frames_out": d.get("mean_enc_frames_out"),
                "audio_tokens_in": d.get("audio_tokens_in"),
                "audio_tokens_out": d.get("audio_tokens_out"),
                "audio_token_saved_pct": d.get("audio_token_saved_pct"),
            }
        )
    summary = {"updated": now(), "n_results": len(rows), "results": rows}
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("=== OOD suite start ===")
    cur, mx = pids()
    log(f"cgroup pids {cur}/{mx}")
    seed_existing()

    jobs: list[dict] = []

    # 1) AMI WER (packing already done)
    jobs.append(
        {
            "name": "whisper_w4a4_ami",
            "out": RESULTS / "whisper_w4a4_ami.json",
            "kind": "eval",
            "cmd": whisper_cmd("w4a4", "ami", RESULTS / "whisper_w4a4_ami.json", bs=8),
        }
    )
    jobs.append(
        {
            "name": "qwen_w4a4_ami",
            "out": RESULTS / "qwen_w4a4_ami.json",
            "kind": "eval",
            "cmd": qwen_cmd("w4a4", "ami", RESULTS / "qwen_w4a4_ami.json", bs=4),
        }
    )

    english = [
        ("earnings22", 4, 2),
        ("voxpopuli", 8, 4),
        ("tedlium", 8, 4),
    ]
    for ds, w_bs, q_bs in english:
        jobs.append(
            {
                "name": f"prep_{ds}",
                "out": ROOT / "data" / "ood" / ds / "en" / "all.trans.txt",
                "kind": "prep",
                "cmd": py(str(SRC / "ood_data.py"), ds),
            }
        )
        jobs.append(
            {
                "name": f"whisper_fp16_{ds}",
                "out": RESULTS / f"whisper_fp16_{ds}.json",
                "kind": "eval",
                "cmd": whisper_cmd("fp16", ds, RESULTS / f"whisper_fp16_{ds}.json", bs=w_bs),
            }
        )
        jobs.append(
            {
                "name": f"whisper_w4a4_{ds}",
                "out": RESULTS / f"whisper_w4a4_{ds}.json",
                "kind": "eval",
                "cmd": whisper_cmd("w4a4", ds, RESULTS / f"whisper_w4a4_{ds}.json", bs=w_bs),
            }
        )
        jobs.append(
            {
                "name": f"qwen_fp16_{ds}",
                "out": RESULTS / f"qwen_fp16_{ds}.json",
                "kind": "eval",
                "cmd": qwen_cmd("fp16", ds, RESULTS / f"qwen_fp16_{ds}.json", bs=q_bs),
            }
        )
        jobs.append(
            {
                "name": f"qwen_w4a4_{ds}",
                "out": RESULTS / f"qwen_w4a4_{ds}.json",
                "kind": "eval",
                "cmd": qwen_cmd("w4a4", ds, RESULTS / f"qwen_w4a4_{ds}.json", bs=q_bs),
            }
        )
        jobs.append(
            {
                "name": f"whisper_pack_{ds}",
                "out": RESULTS / f"whisper_pack_collapse_{ds}.json",
                "kind": "eval",
                "cmd": whisper_cmd(
                    "fp16",
                    ds,
                    RESULTS / f"whisper_pack_collapse_{ds}.json",
                    pack="collapse",
                    bs=w_bs,
                ),
            }
        )
        jobs.append(
            {
                "name": f"qwen_pack_{ds}",
                "out": RESULTS / f"qwen_pack_collapse_{ds}.json",
                "kind": "eval",
                "cmd": qwen_cmd(
                    "silence_collapse",
                    ds,
                    RESULTS / f"qwen_pack_collapse_{ds}.json",
                    bs=q_bs,
                ),
            }
        )

    jobs.append(
        {
            "name": "prep_commonvoice",
            "out": ROOT / "data" / "ood" / "manifests" / "commonvoice_ready.json",
            "kind": "prep_cv",
            "cmd": py(str(SRC / "ood_data.py"), "commonvoice", "--lang", "all"),
        }
    )
    cv_out_w_fp = RESULTS / "whisper_fp16_commonvoice.json"
    cv_out_w_q = RESULTS / "whisper_w4a4_commonvoice.json"
    cv_out_q_fp = RESULTS / "qwen_fp16_commonvoice.json"
    cv_out_q_q = RESULTS / "qwen_w4a4_commonvoice.json"
    cv_out_w_p = RESULTS / "whisper_pack_collapse_commonvoice.json"
    cv_out_q_p = RESULTS / "qwen_pack_collapse_commonvoice.json"
    jobs.append(
        {
            "name": "whisper_fp16_commonvoice",
            "out": cv_out_w_fp,
            "kind": "eval",
            "cmd": whisper_cmd("fp16", "commonvoice", cv_out_w_fp, bs=8, lang="all"),
        }
    )
    jobs.append(
        {
            "name": "whisper_w4a4_commonvoice",
            "out": cv_out_w_q,
            "kind": "eval",
            "cmd": whisper_cmd("w4a4", "commonvoice", cv_out_w_q, bs=8, lang="all"),
        }
    )
    jobs.append(
        {
            "name": "qwen_fp16_commonvoice",
            "out": cv_out_q_fp,
            "kind": "eval",
            "cmd": qwen_cmd("fp16", "commonvoice", cv_out_q_fp, bs=4, lang="all"),
        }
    )
    jobs.append(
        {
            "name": "qwen_w4a4_commonvoice",
            "out": cv_out_q_q,
            "kind": "eval",
            "cmd": qwen_cmd("w4a4", "commonvoice", cv_out_q_q, bs=4, lang="all"),
        }
    )
    jobs.append(
        {
            "name": "whisper_pack_commonvoice",
            "out": cv_out_w_p,
            "kind": "eval",
            "cmd": whisper_cmd(
                "fp16", "commonvoice", cv_out_w_p, pack="collapse", bs=8, lang="all"
            ),
        }
    )
    jobs.append(
        {
            "name": "qwen_pack_commonvoice",
            "out": cv_out_q_p,
            "kind": "eval",
            "cmd": qwen_cmd("silence_collapse", "commonvoice", cv_out_q_p, bs=4, lang="all"),
        }
    )

    failed: list[str] = []
    for i, job in enumerate(jobs, 1):
        name = job["name"]
        out = Path(job["out"])
        write_status(
            {
                "phase": "running",
                "job": name,
                "index": i,
                "n_jobs": len(jobs),
                "updated": now(),
                "failed": failed,
            }
        )
        if job["kind"] == "eval" and job_done(out):
            log(f"[{i}/{len(jobs)}] SKIP {name} (exists)")
            continue
        if job["kind"] == "prep" and out.is_file():
            log(f"[{i}/{len(jobs)}] SKIP {name} (cache exists)")
            continue
        if job["kind"] == "prep_cv":
            # always run; prepare_commonvoice is incremental per locale
            pass
        log(f"[{i}/{len(jobs)}] START {name}")
        rc = 1
        for attempt in range(1, 4):
            try:
                rc = run_cmd(job["cmd"], LOGS / f"ood_{name}.log")
            except subprocess.TimeoutExpired:
                log(f"{name} timed out (attempt {attempt})")
                rc = 124
            if rc == 0:
                break
            log(f"{name} rc={rc} attempt={attempt}; sleep 30s")
            time.sleep(30)
        if job["kind"] == "prep_cv":
            # mark ready if at least one CV locale cached
            cv_root = ROOT / "data" / "ood" / "commonvoice"
            n_lang = len(list(cv_root.glob("*/all.trans.txt"))) if cv_root.is_dir() else 0
            out.write_text(json.dumps({"n_langs": n_lang, "updated": now()}, indent=2))
            if n_lang == 0:
                log(f"{name} produced 0 locales")
                failed.append(name)
                continue
        if rc != 0 and job["kind"] != "prep_cv":
            log(f"FAIL {name} rc={rc}")
            failed.append(name)
        else:
            log(f"DONE {name} rc={rc}")
        collect_summary()

    summary = collect_summary()
    write_status({"phase": "done", "updated": now(), "failed": failed, "n_results": summary["n_results"]})
    log(f"=== OOD suite finished failed={failed} n_results={summary['n_results']} ===")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
