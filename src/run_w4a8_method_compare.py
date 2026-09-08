"""Plan / skip-if-done runner for the W4A8 method table.

Does not launch GPU work unless invoked as:
  python src/run_w4a8_method_compare.py --run
Default is --dry-run.

W4A8 methods we can actually execute here:
  RTN, SmoothQuant+RTN, GPTQ-W + per-tensor A8, AWQ-W + per-tensor A8, ours (log A8)

W4A16 / mixed-W papers (cite only, no code):
  FADE, GenPTQ, K-means MP
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/SpeechPTQ")
PY = "/venv/main/bin/python"
OUT = ROOT / "results" / "compare" / "w4a8"
GPTQ_WHISPER = ROOT / "models" / "whisper-large-v3-W4A16-G128"
GPTQ_QWEN_LLM = ROOT / "models" / "qwen3-asr-1.7b-W4A16-G128-llm"
GPTQ_QWEN_ENC = ROOT / "models" / "qwen3-asr-1.7b-W4A16-G128-llm-encgptq"
AWQ_WHISPER = ROOT / "models" / "whisper-large-v3-AWQ-W4A16-G128"
AWQ_QWEN_ENC = ROOT / "models" / "qwen3-asr-1.7b-W4A16-G128-llm-encawq"
EXISTING = ROOT / "results" / "ablation_w4a4"

CITE_ONLY = [
    {
        "name": "FADE",
        "bits": "W4A16",
        "runnable": False,
        "reason": "No public ASR-FADE code. Weight-only; not a W4A8 method.",
        "ref": "arXiv:2601.02455",
    },
    {
        "name": "GenPTQ",
        "bits": "mixed-W A16 (~W2.5A16)",
        "runnable": False,
        "reason": "No public code in this workspace. Weight-only mixed precision.",
        "ref": "Kang & Kim, EMNLP 2025 Findings",
    },
    {
        "name": "K-means MP",
        "bits": "mixed-W A16 (~W2.12A16)",
        "runnable": False,
        "reason": "No public code in this workspace. Weight-only; sparse FP32 outliers.",
        "ref": "Gu et al., Interspeech 2025",
    },
]


def done_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return d.get("wer") is not None


def ckpt_ok(path: Path) -> bool:
    p = Path(path)
    if p.suffix == ".pt":
        return p.is_file()
    if p.is_file():
        return True
    if not (p / "config.json").is_file():
        return False
    if (p / "awq_scales.pt").is_file():
        return True
    return bool(list(p.glob("*.safetensors")) + list(p.glob("pytorch_model*.bin")))


def calib_done(job: dict) -> bool:
    """config.json alone is not a GPTQ ckpt; require weights / AWQ scales."""
    out = Path(job["out"])
    if out.name == "config.json":
        return ckpt_ok(out.parent)
    return out.is_file()


def copy_qwen_processor(dst: Path) -> None:
    """oneshot often writes weights without the ASR preprocessor."""
    if not dst.is_dir():
        return
    for name in (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        src = GPTQ_QWEN_LLM / name
        if src.is_file() and not (dst / name).is_file():
            shutil.copy2(src, dst / name)


def eval_jobs() -> list[dict]:
    jobs = []
    for split in ("test-clean", "test-other"):
        jobs.append(
            {
                "id": f"whisper_rtn_{split}",
                "out": OUT / f"whisper_rtn_w4a8_{split}.json",
                "cmd": [
                    PY, "src/run_eval.py", "--mode", "w4a4", "--split", split,
                    "--w-mode", "uniform", "--a-mode", "uniform_a8_dyn",
                    "--scope", "nolm", "--group-size", "64", "--act-group-size", "0",
                    "--smoothquant-alpha", "0", "--backend", "fake",
                    "--batch-size", "16", "--max-new-tokens", "224", "--pack", "none",
                    "--out", str(OUT / f"whisper_rtn_w4a8_{split}.json"),
                ],
            }
        )
        sq_existing = EXISTING / f"whisper_w64_a0_w4a8_{split}.json"
        jobs.append(
            {
                "id": f"whisper_sq_{split}",
                "out": sq_existing,
                "cmd": None,
                "reuse": str(sq_existing),
                "note": "already SQ+RTN W4A8 g_w=64 g_a=0",
            }
        )
        jobs.append(
            {
                "id": f"whisper_gptq_{split}",
                "out": OUT / f"whisper_gptq_w4a8_{split}.json",
                "needs_ckpt": str(GPTQ_WHISPER),
                "cmd": [
                    PY, "src/run_eval.py", "--mode", "w4a4", "--split", split,
                    "--model-id", str(GPTQ_WHISPER),
                    "--w-mode", "none", "--a-mode", "uniform_a8_dyn",
                    "--scope", "nolm", "--group-size", "0", "--act-group-size", "0",
                    "--backend", "fake", "--batch-size", "16", "--max-new-tokens", "224",
                    "--pack", "none",
                    "--out", str(OUT / f"whisper_gptq_w4a8_{split}.json"),
                ],
                "note": "GPTQ-W + per-tensor uniform A8",
            }
        )
        jobs.append(
            {
                "id": f"whisper_awq_{split}",
                "out": OUT / f"whisper_awq_w4a8_{split}.json",
                "needs_ckpt": str(AWQ_WHISPER / "awq_scales.pt"),
                "cmd": [
                    PY, "src/run_eval.py", "--mode", "w4a4", "--split", split,
                    "--model-id", "openai/whisper-large-v3",
                    "--w-mode", "uniform", "--a-mode", "uniform_a8_dyn",
                    "--scope", "nolm", "--group-size", "64", "--act-group-size", "0",
                    "--awq-scales", str(AWQ_WHISPER / "awq_scales.pt"),
                    "--backend", "fake", "--batch-size", "16", "--max-new-tokens", "224",
                    "--pack", "none",
                    "--out", str(OUT / f"whisper_awq_w4a8_{split}.json"),
                ],
                "note": "AWQ-W (Edge-ASR scales + grouped RTN g=64) + per-tensor uniform A8",
            }
        )
        ours_w = EXISTING / f"whisper_unifW64_logA8_a0_{split}.json"
        jobs.append(
            {
                "id": f"whisper_ours_{split}",
                "out": ours_w,
                "cmd": None,
                "reuse": str(ours_w),
                "note": "log A8 grouping-free, g_w=64",
            }
        )
        jobs.append(
            {
                "id": f"qwen_rtn_{split}",
                "out": OUT / f"qwen_enc_rtn_w4a8_{split}.json",
                "cmd": [
                    PY, "src/run_eval_qwen3.py", "--mode", "w4a4", "--split", split,
                    "--model-id", str(GPTQ_QWEN_LLM),
                    "--w-mode", "uniform", "--a-mode", "encoder_a8",
                    "--scope", "encoder", "--group-size", "64", "--act-group-size", "0",
                    "--smoothquant-alpha", "0", "--backend", "fake",
                    "--batch-size", "8", "--max-new-tokens", "256",
                    "--out", str(OUT / f"qwen_enc_rtn_w4a8_{split}.json"),
                ],
            }
        )
        jobs.append(
            {
                "id": f"qwen_sq_{split}",
                "out": OUT / f"qwen_enc_sq_w4a8_{split}.json",
                "cmd": [
                    PY, "src/run_eval_qwen3.py", "--mode", "w4a4", "--split", split,
                    "--model-id", str(GPTQ_QWEN_LLM),
                    "--w-mode", "uniform", "--a-mode", "encoder_a8",
                    "--scope", "encoder", "--group-size", "64", "--act-group-size", "0",
                    "--smoothquant-alpha", "0.5", "--backend", "fake",
                    "--batch-size", "8", "--max-new-tokens", "256",
                    "--out", str(OUT / f"qwen_enc_sq_w4a8_{split}.json"),
                ],
                "note": "not the old HF+nolm qwen_w64_a0_w4a8 (that RTN-quantizes the LLM)",
            }
        )
        jobs.append(
            {
                "id": f"qwen_gptq_{split}",
                "out": OUT / f"qwen_enc_gptq_w4a8_{split}.json",
                "needs_ckpt": str(GPTQ_QWEN_ENC),
                "cmd": [
                    PY, "src/run_eval_qwen3.py", "--mode", "w4a4", "--split", split,
                    "--model-id", str(GPTQ_QWEN_ENC),
                    "--w-mode", "none", "--a-mode", "encoder_a8",
                    "--scope", "encoder", "--group-size", "0", "--act-group-size", "0",
                    "--backend", "fake", "--batch-size", "8", "--max-new-tokens", "256",
                    "--out", str(OUT / f"qwen_enc_gptq_w4a8_{split}.json"),
                ],
                "note": "encoder GPTQ-W + per-tensor A8; LLM stays GPTQ W4A16",
            }
        )
        jobs.append(
            {
                "id": f"qwen_awq_{split}",
                "out": OUT / f"qwen_enc_awq_w4a8_{split}.json",
                "needs_ckpt": str(AWQ_QWEN_ENC / "awq_scales.pt"),
                "cmd": [
                    PY, "src/run_eval_qwen3.py", "--mode", "w4a4", "--split", split,
                    "--model-id", str(GPTQ_QWEN_LLM),
                    "--w-mode", "uniform", "--a-mode", "encoder_a8",
                    "--scope", "encoder", "--group-size", "64", "--act-group-size", "0",
                    "--awq-scales", str(AWQ_QWEN_ENC / "awq_scales.pt"),
                    "--backend", "fake", "--batch-size", "8", "--max-new-tokens", "256",
                    "--out", str(OUT / f"qwen_enc_awq_w4a8_{split}.json"),
                ],
                "note": "encoder AWQ-W + per-tensor A8; LLM stays GPTQ W4A16",
            }
        )
        ours_q = EXISTING / f"qwen_gptq_enc_wuni_g64_alogA8_a0_{split}.json"
        jobs.append(
            {
                "id": f"qwen_ours_{split}",
                "out": ours_q,
                "cmd": None,
                "reuse": str(ours_q),
                "note": "GPTQ LLM + encoder uniform W g=64, log A8 g_a=0",
            }
        )
    return jobs


def calib_jobs() -> list[dict]:
    return [
        {
            "id": "calib_whisper_awq",
            "out": AWQ_WHISPER / "awq_scales.pt",
            "cmd": [PY, "src/quant_whisper_awq.py"],
        },
        {
            "id": "calib_qwen_enc_gptq",
            "out": GPTQ_QWEN_ENC / "config.json",
            "cmd": [PY, "src/quant_qwen3_enc_gptq.py"],
        },
        {
            "id": "calib_qwen_enc_awq",
            "out": AWQ_QWEN_ENC / "awq_scales.pt",
            "cmd": [PY, "src/quant_qwen3_enc_awq.py"],
        },
    ]


def status_row(job: dict) -> str:
    out = Path(job["out"])
    if job.get("reuse"):
        ok = done_ok(Path(job["reuse"]))
    elif out.name == "config.json":
        ok = ckpt_ok(out.parent)
    elif out.suffix == ".json":
        ok = done_ok(out)
    else:
        ok = out.is_file()
    flag = "DONE" if ok else "TODO"
    extra = ""
    if job.get("needs_ckpt") and not ckpt_ok(Path(job["needs_ckpt"])):
        extra = "  [needs calib ckpt]"
        if ok:
            extra = ""
        else:
            flag = "WAIT"
    return f"  {flag:4s}  {job['id']:28s}  {out}{extra}"


def print_plan() -> None:
    print("=== cite-only (will not run) ===")
    for c in CITE_ONLY:
        print(f"  {c['name']:12s} {c['bits']:22s}  {c['reason']}")
    print("=== calib checkpoints ===")
    print(f"  {'HAVE' if ckpt_ok(GPTQ_WHISPER) else 'MISS'}  whisper GPTQ  {GPTQ_WHISPER}")
    print(f"  {'HAVE' if ckpt_ok(GPTQ_QWEN_LLM) else 'MISS'}  qwen LLM GPTQ {GPTQ_QWEN_LLM}")
    for j in calib_jobs():
        print(status_row(j))
    print("=== eval jobs (LS clean+other) ===")
    for j in eval_jobs():
        print(status_row(j))


def run_cmd(cmd: list[str]) -> bool:
    print("=== RUN", " ".join(cmd), flush=True)
    ec = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    if ec != 0:
        print(f"=== FAIL exit={ec}  {' '.join(cmd)}", flush=True)
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="Actually launch calib+eval. Default is dry-run.")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cite_w4a16.json").write_text(json.dumps(CITE_ONLY, indent=2))
    print_plan()
    if not args.run:
        print("\ndry-run only. Re-run with --run to start GPU jobs.", flush=True)
        return
    copy_qwen_processor(GPTQ_QWEN_ENC)
    copy_qwen_processor(AWQ_QWEN_ENC)
    failed: list[str] = []

    def try_eval(j: dict) -> None:
        out = Path(j["out"])
        if j.get("reuse"):
            print("=== SKIP reuse", j["id"], j["reuse"], flush=True)
            return
        if done_ok(out):
            print("=== SKIP", j["id"], flush=True)
            return
        need = j.get("needs_ckpt")
        if need and not ckpt_ok(Path(need)):
            print("=== DEFER", j["id"], "missing", need, flush=True)
            return
        if not run_cmd(j["cmd"]):
            failed.append(j["id"])

    # Evals whose checkpoints already exist (RTN / SQ / Whisper GPTQ / ours).
    for j in eval_jobs():
        need = j.get("needs_ckpt")
        if need and not ckpt_ok(Path(need)):
            continue
        try_eval(j)
    for j in calib_jobs():
        if calib_done(j):
            print("=== SKIP", j["id"], flush=True)
            continue
        if not run_cmd(j["cmd"]):
            failed.append(j["id"])
    for j in eval_jobs():
        try_eval(j)
    if failed:
        print("FAILED:", ", ".join(failed), flush=True)
        raise SystemExit(1)
    print("ALL JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
