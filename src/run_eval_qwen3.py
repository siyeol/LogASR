"""Qwen3-ASR eval on LibriSpeech (FP16/bf16, silence-mel pad, log W4A4)."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from jiwer import wer as corpus_wer
from transformers import AutoModelForMultimodalLM, AutoProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from ami import load_ami_ihm_test, load_ami_stitch30, stitch_pause_windows
from data import SR, load_audio, load_split
from ood_data import grouped_error_rates, load_ood_items
from quant import collect_act_ch_absmax, replace_linears
from silence_collapse import collapse_silence


def compressed_tensors_load_kwargs(model_id: str) -> dict:
    from transformers import AutoConfig
    from transformers.utils.quantization_config import CompressedTensorsConfig

    cfg = AutoConfig.from_pretrained(model_id)
    qc = getattr(cfg, "quantization_config", None)
    if qc is None:
        return {}
    if isinstance(qc, CompressedTensorsConfig):
        qc.run_compressed = False
        return {"quantization_config": qc}
    if isinstance(qc, dict) and qc.get("quant_method") == "compressed-tensors":
        return {
            "quantization_config": CompressedTensorsConfig.from_dict(
                {**qc, "run_compressed": False}
            )
        }
    return {}


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = torch.device("cuda")
RESULTS = Path("/workspace/SpeechPTQ/results")
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp16", "silence_pad", "w4a4", "w4a8", "silence_collapse"], required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument(
        "--dataset",
        default="librispeech",
        choices=["librispeech", "ami", "earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"],
    )
    p.add_argument(
        "--lang",
        default="en",
        help="commonvoice locale, or 'all'. English OOD sets ignore this.",
    )
    p.add_argument("--out", default="", help="Optional JSON output path.")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--w-mode", default="log")
    p.add_argument("--a-mode", default="log_token")
    p.add_argument("--scope", default="nolm")
    p.add_argument("--backend", default="fake", choices=["fake", "int4"])
    p.add_argument("--smoothquant-alpha", type=float, default=0.0)
    p.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="0 = grouping-free. mixed_log_* keeps encoder FFN log grouping-free.",
    )
    p.add_argument(
        "--act-group-size",
        type=int,
        default=-1,
        help="Activation group size. -1 = same as --group-size. 0 = per-tensor activations.",
    )
    p.add_argument(
        "--awq-scales",
        default="",
        help="Path to Edge-ASR AWQ scale dict (awq_scales.pt). Applied before grouped W quant.",
    )
    p.add_argument("--ami-stitch-sec", type=float, default=0.0, help="If >0, stitch AMI chunks with real pause gaps")
    p.add_argument("--collapse-hold-frames", type=int, default=1)
    p.add_argument("--collapse-rel-db", type=float, default=-35.0)
    return p.parse_args()


def build_silence_mel(processor, n_samples: int = 16000) -> torch.Tensor:
    """Log-mel of 1s zeros. Pad frames in batches are 0, which is not real silence mel."""
    sil = np.zeros(n_samples, dtype=np.float32)
    feats = processor.feature_extractor(
        sil, sampling_rate=SR, return_tensors="pt", padding=False
    )
    return feats["input_features"][0].contiguous()  # [mel, T]


def fill_pad_with_silence_mel(
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
    silence_mel: torch.Tensor,
) -> torch.Tensor:
    """Replace masked (pad) log-mel frames with tiled real silence spectrogram."""
    sil = silence_mel.to(device=input_features.device, dtype=input_features.dtype)
    b, c, t = input_features.shape
    t_sil = sil.shape[-1]
    reps = (t + t_sil - 1) // t_sil
    sil_t = sil.repeat(1, reps)[:, :t].unsqueeze(0).expand(b, -1, -1)
    valid = input_features_mask.unsqueeze(1).bool()
    return torch.where(valid, input_features, sil_t)


@torch.inference_mode()
def transcribe_batch(model, processor, waves, max_new_tokens, silence_mel=None, language="English"):
    inputs = processor.apply_transcription_request(
        audio=waves,
        language=language,
    )
    inputs = inputs.to(DEVICE, model.dtype)
    if silence_mel is not None and "input_features" in inputs and "input_features_mask" in inputs:
        inputs["input_features"] = fill_pad_with_silence_mel(
            inputs["input_features"], inputs["input_features_mask"], silence_mel
        )
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    gen = out[:, prompt_len:]
    texts = processor.decode(gen, return_format="transcription_only")
    n_tok = audio_token_lengths(processor, inputs)
    return texts, n_tok


def audio_token_lengths(processor, inputs) -> np.ndarray:
    n_window = getattr(processor.feature_extractor, "n_window", 50)
    mask = inputs["input_features_mask"]
    if torch.is_tensor(mask):
        lens = mask.sum(-1).detach().cpu()
    else:
        lens = mask.sum(-1)
    return processor._get_audio_token_length(lens, n_window)


def count_audio_tokens(processor, waves) -> np.ndarray:
    packed = processor.apply_transcription_request(audio=waves, language="English")
    return audio_token_lengths(processor, packed)


def item_wave(item: dict) -> np.ndarray:
    if "audio" in item:
        return item["audio"]
    return load_audio(item["path"])


def _normalize(text: str, language: str, en_norm, basic_norm) -> str:
    lang = (language or "en").split("-")[0].lower()
    if lang in ("en", "english"):
        return en_norm(text)
    return basic_norm(text)


def load_items(args) -> list[dict]:
    if args.dataset == "ami":
        if args.ami_stitch_sec and args.ami_stitch_sec > 0:
            try:
                items = load_ami_stitch30()
                print(f"[ami] loaded {len(items)} cached stitch windows")
            except FileNotFoundError:
                items = load_ami_ihm_test(limit=0)
                items = stitch_pause_windows(items, target_sec=args.ami_stitch_sec)
        else:
            items = load_ami_ihm_test(limit=0)
        if args.limit and args.limit > 0:
            items = items[: args.limit]
        return items
    if args.dataset in ("earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"):
        items = load_ood_items(args.dataset, lang=getattr(args, "lang", "en"), limit=0)
        print(f"[{args.dataset}] loaded {len(items)} cached utts lang={getattr(args, 'lang', 'en')}")
        if args.limit and args.limit > 0:
            items = items[: args.limit]
        return items
    items = load_split(args.split)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    return items


def main():
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = load_items(args)
    if not items:
        raise RuntimeError(f"no items for dataset={args.dataset} lang={getattr(args, 'lang', '')}")
    print(
        f"dataset={args.dataset} split={args.split} n={len(items)} mode={args.mode} "
        f"bs={args.batch_size} model={args.model_id} stitch={args.ami_stitch_sec}"
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    load_kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    load_kw.update(compressed_tensors_load_kwargs(args.model_id))
    model = AutoModelForMultimodalLM.from_pretrained(args.model_id, **load_kw)
    model.to(DEVICE)
    model.eval()

    silence_mel = None
    if args.mode == "silence_pad":
        silence_mel = build_silence_mel(processor)
        print(f"[silence_pad] template mel {tuple(silence_mel.shape)} {silence_mel.dtype}")
    do_collapse = args.mode == "silence_collapse"

    if args.mode == "w4a4":
        ch_stats = {}
        if args.smoothquant_alpha > 0:
            try:
                calib_src = load_split("test-other")
            except FileNotFoundError:
                calib_src = items
            n_cal = max(args.batch_size * 4, args.batch_size)
            calib = calib_src[:n_cal]
            packed = []
            for start in range(0, len(calib), args.batch_size):
                waves = [item_wave(x) for x in calib[start : start + args.batch_size]]
                inputs = processor.apply_transcription_request(audio=waves, language="English")
                packed.append(inputs.to(DEVICE, model.dtype))

            def gen_fn(inputs):
                model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )

            print(f"[w4a4] SmoothQuant calib {len(calib)} utts alpha={args.smoothquant_alpha}")
            ch_stats = collect_act_ch_absmax(model, packed, gen_fn, max_batches=len(packed))
        pre_scales = None
        awq_path = getattr(args, "awq_scales", "") or ""
        if awq_path:
            raw = torch.load(awq_path, map_location="cpu", weights_only=False)
            pre_scales = {k: v.float() for k, v in raw.items()}
            print(f"[w4a4] AWQ scales {awq_path} n={len(pre_scales)}")
        n = replace_linears(
            model,
            {},
            w_mode=args.w_mode,
            a_mode=args.a_mode,
            scope=args.scope,
            backend=args.backend,
            smoothquant_alpha=args.smoothquant_alpha,
            act_ch_absmax=ch_stats,
            group_size=getattr(args, "group_size", 0),
            act_group_size=getattr(args, "act_group_size", -1),
            smooth_scales=pre_scales,
        )
        ag = getattr(args, "act_group_size", -1)
        print(
            f"[w4a4] replaced {n} Linear layers backend={args.backend} "
            f"sq={args.smoothquant_alpha} g_w={getattr(args, 'group_size', 0)} g_a={ag}"
        )
        model.to(DEVICE)
        model.eval()
    if args.mode == "w4a8":
        # LLM A8 collapses Qwen3 (~37% WER smoke). Keep GPTQ/fp LLM as W4A16;
        # apply grouping-free log W4 + uniform A8 on the speech encoder.
        n_enc = replace_linears(
            model, {}, w_mode="log", a_mode="uniform_a8_dyn", scope="encoder", backend="fake"
        )
        print(f"[w4a8] encoder log-W4 A8 n={n_enc} (LLM activations left at A16)")
        model.to(DEVICE)
        model.eval()

    en_norm = EnglishTextNormalizer()
    basic_norm = BasicTextNormalizer()
    items = sorted(items, key=lambda x: str(x.get("qwen_language") or x.get("language") or "en"))
    hyps: list[str] = []
    refs: list[str] = []
    total_audio = 0.0
    total_audio_out = 0.0
    n_tokens_in = 0
    n_tokens_out = 0
    collapse_runs = 0
    torch.cuda.reset_peak_memory_stats()
    mem_after_load = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        waves = [item_wave(x) for x in chunk]
        total_audio += sum(len(w) for w in waves) / SR
        waves_in = waves
        if do_collapse:
            collapsed = []
            for w in waves:
                w2, st = collapse_silence(
                    w,
                    hold_frames=args.collapse_hold_frames,
                    rel_db=args.collapse_rel_db,
                )
                collapsed.append(w2)
                total_audio_out += st["audio_sec_out"]
                collapse_runs += st["silence_runs"]
            n_tokens_in += int(np.sum(count_audio_tokens(processor, waves_in)))
            waves = collapsed
        else:
            total_audio_out += sum(len(w) for w in waves) / SR
        langs = [str(x.get("qwen_language") or x.get("language") or "en") for x in chunk]
        max_tok = args.max_new_tokens
        max_dur = max((len(w) / SR) for w in waves)
        if max_dur > 35:
            max_tok = max(max_tok, int(max_dur * 12))
        # keep language homogeneous in a call
        if len(set(langs)) == 1:
            texts, tok_out = transcribe_batch(
                model, processor, waves, max_tok, silence_mel, language=langs[0]
            )
            if isinstance(texts, str):
                texts = [texts]
        else:
            texts = []
            tok_parts = []
            for w, lang in zip(waves, langs):
                t, tok = transcribe_batch(
                    model, processor, [w], max_tok, silence_mel, language=lang
                )
                texts.extend(t if isinstance(t, list) else [t])
                tok_parts.append(tok)
            tok_out = np.concatenate([np.atleast_1d(p) for p in tok_parts])
        if not do_collapse:
            n_tokens_in += int(np.sum(tok_out))
        n_tokens_out += int(np.sum(tok_out))
        for t, x, lang in zip(texts, chunk, langs):
            hyps.append(_normalize(t, lang, en_norm, basic_norm))
            refs.append(_normalize(x["text"], lang, en_norm, basic_norm))
        done = start + len(chunk)
        if done % (args.batch_size * 5) == 0 or done == len(items):
            elapsed = time.perf_counter() - t0
            rtf = elapsed / max(total_audio, 1e-8)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(
                f"  {done}/{len(items)}  audio={total_audio:.1f}s  elapsed={elapsed:.1f}s  "
                f"RTF={rtf:.4f}  peak_mem={peak:.0f}MiB"
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    wer = corpus_wer(refs, hyps) * 100.0
    rtf = elapsed / max(total_audio, 1e-8)
    tag = f"qwen3_{args.mode}_{args.dataset}_{args.split}"
    if args.dataset in ("earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"):
        tag += f"_{getattr(args, 'lang', 'en')}"
    if args.dataset == "ami" and args.ami_stitch_sec:
        tag += f"_stitch{int(args.ami_stitch_sec)}"
    if args.model_id != MODEL_ID:
        tag += "_" + Path(args.model_id).name.replace(" ", "_")
    if args.mode == "w4a4":
        tag += f"_{args.w_mode}_{args.a_mode}_{args.scope}_{args.backend}"
        if args.smoothquant_alpha > 0:
            tag += f"_sq{args.smoothquant_alpha}"
    if args.mode == "w4a8":
        tag += "_encLogW4A8_llmA16"
    if args.mode == "silence_collapse":
        tag += f"_hold{args.collapse_hold_frames}_db{args.collapse_rel_db:g}"
    if args.limit and args.limit > 0:
        tag += f"_n{args.limit}_bs{args.batch_size}"
    saved = 0.0 if n_tokens_in == 0 else 100.0 * (1.0 - n_tokens_out / n_tokens_in)
    out = {
        "mode": args.mode,
        "dataset": args.dataset,
        "model_id": args.model_id,
        "split": args.split,
        "n_utts": len(items),
        "wer": round(wer, 4),
        "rtf": round(rtf, 6),
        "elapsed_sec": round(elapsed, 3),
        "audio_sec": round(total_audio, 3),
        "audio_sec_out": round(total_audio_out, 3),
        "audio_tokens_in": int(n_tokens_in),
        "audio_tokens_out": int(n_tokens_out),
        "audio_token_saved_pct": round(saved, 2),
        "ami_stitch_sec": args.ami_stitch_sec,
        "collapse_hold_frames": args.collapse_hold_frames if do_collapse else None,
        "collapse_rel_db": args.collapse_rel_db if do_collapse else None,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": str(model.dtype),
        "gpu_mem_after_load_mib": round(mem_after_load / (1024**2), 1),
        "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "w_mode": args.w_mode if args.mode == "w4a4" else ("log_encoder_keep_llm" if args.mode == "w4a8" else None),
        "a_mode": args.a_mode if args.mode == "w4a4" else ("uniform_a8_dyn" if args.mode == "w4a8" else None),
        "scope": args.scope if args.mode == "w4a4" else ("encoder" if args.mode == "w4a8" else None),
        "backend": args.backend if args.mode in ("w4a4", "w4a8") else None,
        "smoothquant_alpha": args.smoothquant_alpha if args.mode == "w4a4" else 0.0,
        "group_size": getattr(args, "group_size", 0) if args.mode == "w4a4" else 0,
        "act_group_size": getattr(args, "act_group_size", -1) if args.mode == "w4a4" else -1,
        "lang": getattr(args, "lang", "en"),
    }
    per_lang, micro = grouped_error_rates(items, refs, hyps)
    if len(per_lang) > 1:
        out["wer_by_lang"] = per_lang
        out["wer_micro"] = micro
        out["wer"] = micro
    path = Path(args.out) if getattr(args, "out", "") else RESULTS / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
