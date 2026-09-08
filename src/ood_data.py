"""Cache AMI-style flac+transcripts for ESB-style and Common Voice test sets.

English sets (earnings22 / voxpopuli / tedlium) try ESB mirrors first, then
upstream Hugging Face configs. Common Voice is downloaded per locale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ami import SR, _to_mono16k, decode_hf_audio

CACHE_ROOT = Path("/workspace/SpeechPTQ/data/ood")
MANIFEST_ROOT = CACHE_ROOT / "manifests"

TEXT_KEYS = (
    "text",
    "sentence",
    "transcription",
    "normalized_text",
    "raw_text",
    "transcript",
)

# Common Voice locales that both Whisper-large-v3 and Qwen3-ASR-1.7B cover.
# unit=char for scripts where word WER is meaningless.
CV_LANGS: list[dict] = [
    {"cv": "en", "whisper": "en", "qwen": "en", "unit": "word"},
    {"cv": "ar", "whisper": "ar", "qwen": "ar", "unit": "word"},
    {"cv": "cs", "whisper": "cs", "qwen": "cs", "unit": "word"},
    {"cv": "da", "whisper": "da", "qwen": "da", "unit": "word"},
    {"cv": "de", "whisper": "de", "qwen": "de", "unit": "word"},
    {"cv": "el", "whisper": "el", "qwen": "el", "unit": "word"},
    {"cv": "es", "whisper": "es", "qwen": "es", "unit": "word"},
    {"cv": "fa", "whisper": "fa", "qwen": "fa", "unit": "word"},
    {"cv": "fi", "whisper": "fi", "qwen": "fi", "unit": "word"},
    {"cv": "fr", "whisper": "fr", "qwen": "fr", "unit": "word"},
    {"cv": "hi", "whisper": "hi", "qwen": "hi", "unit": "word"},
    {"cv": "hu", "whisper": "hu", "qwen": "hu", "unit": "word"},
    {"cv": "id", "whisper": "id", "qwen": "id", "unit": "word"},
    {"cv": "it", "whisper": "it", "qwen": "it", "unit": "word"},
    {"cv": "ja", "whisper": "ja", "qwen": "ja", "unit": "char"},
    {"cv": "ko", "whisper": "ko", "qwen": "ko", "unit": "word"},
    {"cv": "mk", "whisper": "mk", "qwen": "mk", "unit": "word"},
    {"cv": "nl", "whisper": "nl", "qwen": "nl", "unit": "word"},
    {"cv": "pl", "whisper": "pl", "qwen": "pl", "unit": "word"},
    {"cv": "pt", "whisper": "pt", "qwen": "pt", "unit": "word"},
    {"cv": "ro", "whisper": "ro", "qwen": "ro", "unit": "word"},
    {"cv": "ru", "whisper": "ru", "qwen": "ru", "unit": "word"},
    {"cv": "sv-SE", "whisper": "sv", "qwen": "sv", "unit": "word"},
    {"cv": "th", "whisper": "th", "qwen": "th", "unit": "char"},
    {"cv": "tr", "whisper": "tr", "qwen": "tr", "unit": "word"},
    {"cv": "vi", "whisper": "vi", "qwen": "vi", "unit": "word"},
    {"cv": "zh-CN", "whisper": "zh", "qwen": "zh", "unit": "char"},
    {"cv": "ja", "whisper": "ja", "qwen": "ja", "unit": "char"},  # duplicate guard
]

# de-dup while keeping order
_seen = set()
_cv_unique = []
for _row in CV_LANGS:
    if _row["cv"] not in _seen:
        _seen.add(_row["cv"])
        _cv_unique.append(_row)
CV_LANGS = _cv_unique

ENGLISH_DATASETS = ("earnings22", "voxpopuli", "tedlium", "gigaspeech")

_SOURCES = {
    "earnings22": [
        ("hf-audio/esb-datasets-test-only-sorted", "earnings22", "test"),
        ("hf-audio/esb-datasets-test-only", "earnings22", "test"),
        ("distil-whisper/earnings22", None, "test"),
        ("revdotcom/earnings22", None, "test"),
    ],
    "voxpopuli": [
        ("hf-audio/esb-datasets-test-only-sorted", "voxpopuli", "test"),
        ("hf-audio/esb-datasets-test-only", "voxpopuli", "test"),
        ("facebook/voxpopuli", "en", "test"),
    ],
    "tedlium": [
        ("esb/datasets", "tedlium", "test"),
        ("hf-audio/esb-datasets-test-only-sorted", "tedlium", "test"),
        ("hf-audio/esb-datasets-test-only", "tedlium", "test"),
        ("LIUM/tedlium", "release3", "test"),
        ("LIUM/tedlium", "release1", "test"),
    ],
    "gigaspeech": [
        ("hf-audio/esb-datasets-test-only-sorted", "gigaspeech", "test"),
        ("hf-audio/esb-datasets-test-only", "gigaspeech", "test"),
        ("speechcolab/gigaspeech", "xs", "test"),
    ],
}

CV_REPOS = (
    "mozilla-foundation/common_voice_17_0",
    "mozilla-foundation/common_voice_16_1",
    "mozilla-foundation/common_voice_13_0",
    "fsicoli/common_voice_17_0",
)


def _row_text(row: dict) -> str:
    for k in TEXT_KEYS:
        if k in row and row[k] is not None:
            t = str(row[k]).strip()
            if t:
                return t
    return ""


def _row_id(row: dict, i: int, prefix: str) -> str:
    for k in ("id", "audio_id", "utt_id", "path", "client_id"):
        if k in row and row[k] is not None:
            raw = str(row[k]).replace("/", "_").replace(" ", "_")
            if raw:
                return f"{prefix}_{i:06d}_{raw[:80]}"
    return f"{prefix}_{i:06d}"


def cache_dir(dataset: str, lang: str) -> Path:
    return CACHE_ROOT / dataset / lang


def trans_path(dataset: str, lang: str) -> Path:
    return cache_dir(dataset, lang) / "all.trans.txt"


def load_cached(dataset: str, lang: str = "en") -> list[dict]:
    trans = trans_path(dataset, lang)
    if not trans.is_file():
        raise FileNotFoundError(trans)
    meta = CV_LANGS_BY_CV.get(lang, {"whisper": "en", "qwen": "en", "unit": "word"})
    if dataset in ENGLISH_DATASETS or lang == "en":
        whisper_lang, qwen_lang, unit = "en", "en", "word"
        if dataset == "commonvoice":
            whisper_lang, qwen_lang, unit = meta["whisper"], meta["qwen"], meta["unit"]
    else:
        whisper_lang, qwen_lang, unit = meta["whisper"], meta["qwen"], meta["unit"]
    items: list[dict] = []
    with trans.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            if not flac.is_file():
                continue
            items.append(
                {
                    "id": utt_id,
                    "path": str(flac),
                    "text": text,
                    "language": whisper_lang,
                    "qwen_language": qwen_lang,
                    "unit": unit,
                    "cv_lang": lang,
                    "dataset": dataset,
                }
            )
    return items


CV_LANGS_BY_CV = {r["cv"]: r for r in CV_LANGS}


def _write_items(dataset: str, lang: str, rows: list[dict]) -> list[dict]:
    out_dir = cache_dir(dataset, lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = trans_path(dataset, lang)
    n_ok = 0
    hours = 0.0
    max_dur = 0.0
    items: list[dict] = []
    with trans.open("w") as tf:
        for i, row in enumerate(rows):
            text = _row_text(row)
            if not text:
                continue
            aud = row.get("audio")
            if aud is None:
                continue
            arr, sr = decode_hf_audio(aud)
            wave = _to_mono16k(arr, sr)
            if wave.size < int(0.15 * SR):
                continue
            utt_id = _row_id(row, i, f"{dataset}_{lang}")
            # keep ids filesystem-safe
            utt_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in utt_id)[:120]
            flac = out_dir / f"{utt_id}.flac"
            if not flac.is_file():
                sf.write(str(flac), wave, SR, format="FLAC")
            dur = wave.shape[0] / SR
            hours += dur / 3600.0
            max_dur = max(max_dur, dur)
            # one-line transcript: collapse newlines
            text_one = " ".join(text.split())
            tf.write(f"{utt_id} {text_one}\n")
            n_ok += 1
            items.append(
                {
                    "id": utt_id,
                    "path": str(flac),
                    "text": text_one,
                    "dataset": dataset,
                    "cv_lang": lang,
                }
            )
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    man = {
        "dataset": dataset,
        "lang": lang,
        "n_utts": n_ok,
        "hours": round(hours, 4),
        "max_dur_sec": round(max_dur, 3),
        "cache": str(out_dir),
    }
    (MANIFEST_ROOT / f"{dataset}_{lang}.json").write_text(json.dumps(man, indent=2))
    print(f"[ood] cached {dataset}/{lang}: n={n_ok} hours={hours:.2f} max={max_dur:.1f}s", flush=True)
    return items


def _try_load_dataset(repo: str, config: str | None, split: str):
    from datasets import Audio, load_dataset

    kwargs = dict(split=split)
    try:
        kwargs["verification_mode"] = "no_checks"
    except Exception:
        pass
    try:
        if config:
            ds = load_dataset(repo, config, **kwargs)
        else:
            ds = load_dataset(repo, **kwargs)
    except TypeError:
        kwargs.pop("verification_mode", None)
        kwargs["ignore_verifications"] = True
        if config:
            ds = load_dataset(repo, config, **kwargs)
        else:
            ds = load_dataset(repo, **kwargs)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    return ds


def prepare_english(dataset: str) -> list[dict]:
    if dataset not in _SOURCES:
        raise ValueError(dataset)
    if trans_path(dataset, "en").is_file():
        items = load_cached(dataset, "en")
        print(f"[ood] reuse cache {dataset}/en n={len(items)}", flush=True)
        return items
    last_err = None
    for repo, config, split in _SOURCES[dataset]:
        print(f"[ood] try {dataset}: {repo} config={config} split={split}", flush=True)
        try:
            ds = _try_load_dataset(repo, config, split)
            rows = [ds[i] for i in range(len(ds))]
            items = _write_items(dataset, "en", rows)
            if items:
                return load_cached(dataset, "en")
        except Exception as e:
            last_err = e
            print(f"[ood] failed {repo}: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError(f"Could not prepare {dataset}: {last_err}")


def prepare_commonvoice_lang(cv_lang: str) -> list[dict]:
    if trans_path("commonvoice", cv_lang).is_file():
        items = load_cached("commonvoice", cv_lang)
        print(f"[ood] reuse cache commonvoice/{cv_lang} n={len(items)}", flush=True)
        return items
    last_err = None
    if cv_lang == "en":
        for repo, config, split in (
            ("hf-audio/esb-datasets-test-only-sorted", "common_voice", "test"),
            ("hf-audio/esb-datasets-test-only", "common_voice", "test"),
        ):
            print(f"[ood] try commonvoice/en ESB: {repo}", flush=True)
            try:
                ds = _try_load_dataset(repo, config, split)
                rows = [ds[i] for i in range(len(ds))]
                items = _write_items("commonvoice", "en", rows)
                if items:
                    return load_cached("commonvoice", "en")
            except Exception as e:
                last_err = e
                print(f"[ood] failed {repo}: {type(e).__name__}: {e}", flush=True)
    for repo in CV_REPOS:
        print(f"[ood] try commonvoice/{cv_lang}: {repo}", flush=True)
        try:
            ds = _try_load_dataset(repo, cv_lang, "test")
            rows = [ds[i] for i in range(len(ds))]
            items = _write_items("commonvoice", cv_lang, rows)
            if items:
                return load_cached("commonvoice", cv_lang)
        except Exception as e:
            last_err = e
            print(f"[ood] failed {repo}/{cv_lang}: {type(e).__name__}: {e}", flush=True)
    print(f"[ood] SKIP commonvoice/{cv_lang}: {last_err}", flush=True)
    return []


def prepare_commonvoice(langs: list[str] | None = None) -> dict[str, list[dict]]:
    wanted = langs or [r["cv"] for r in CV_LANGS]
    out: dict[str, list[dict]] = {}
    for lang in wanted:
        items = prepare_commonvoice_lang(lang)
        if items:
            out[lang] = items
    return out


def load_ood_items(dataset: str, lang: str = "en", limit: int = 0) -> list[dict]:
    """Load cached items; prepare cache if missing."""
    if dataset in ENGLISH_DATASETS:
        items = prepare_english(dataset)
    elif dataset == "commonvoice":
        if lang in ("all", "*", ""):
            by_lang = prepare_commonvoice()
            items = []
            for cv_lang in by_lang:
                items.extend(load_cached("commonvoice", cv_lang))
        else:
            raw = prepare_commonvoice_lang(lang)
            items = load_cached("commonvoice", lang) if raw or trans_path("commonvoice", lang).is_file() else []
    else:
        raise ValueError(f"unknown ood dataset {dataset}")
    if limit and limit > 0:
        items = items[:limit]
    return items


def grouped_error_rates(items: list[dict], refs: list[str], hyps: list[str]) -> tuple[dict, float]:
    """Per-language WER (or CER for char langs) plus micro-average."""
    from collections import defaultdict

    from jiwer import cer as corpus_cer
    from jiwer import wer as corpus_wer

    buckets: dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": [], "unit": "word"})
    for it, r, h in zip(items, refs, hyps):
        key = str(it.get("cv_lang") or it.get("language") or "en")
        buckets[key]["refs"].append(r)
        buckets[key]["hyps"].append(h)
        buckets[key]["unit"] = it.get("unit", "word")
    per: dict = {}
    tot_err = 0.0
    tot_ref = 0
    for key, b in buckets.items():
        if b["unit"] == "char":
            rs = [x.replace(" ", "") for x in b["refs"]]
            hs = [x.replace(" ", "") for x in b["hyps"]]
            rate = float(corpus_cer(rs, hs) * 100.0) if rs else 0.0
            nref = sum(len(x) for x in rs)
        else:
            rate = float(corpus_wer(b["refs"], b["hyps"]) * 100.0) if b["refs"] else 0.0
            nref = sum(len(x.split()) for x in b["refs"])
        per[key] = {
            "error_rate": round(rate, 4),
            "unit": b["unit"],
            "n_utts": len(b["refs"]),
            "n_ref_units": nref,
        }
        tot_err += (rate / 100.0) * nref
        tot_ref += nref
    overall = 100.0 * tot_err / max(tot_ref, 1)
    return per, round(overall, 4)


def chunk_wave(wave: np.ndarray, max_samples: int = 30 * SR, overlap: int = SR) -> list[np.ndarray]:
    """Split longer-than-30s audio for Whisper's 1500-frame encoder."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.shape[0] <= max_samples:
        return [wave]
    hop = max(max_samples - max(overlap, 0), max_samples // 2)
    chunks = []
    start = 0
    n = wave.shape[0]
    while start < n:
        end = min(start + max_samples, n)
        chunks.append(wave[start:end])
        if end >= n:
            break
        start += hop
    return chunks


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"])
    p.add_argument("--lang", default="all")
    args = p.parse_args()
    if args.dataset == "commonvoice":
        if args.lang in ("all", "*", ""):
            prepare_commonvoice()
        else:
            prepare_commonvoice_lang(args.lang)
    else:
        prepare_english(args.dataset)
