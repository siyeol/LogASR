"""Fast Whisper-large-v3 eval on LibriSpeech (baseline / W4A4 / KV template)."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
from jiwer import wer as corpus_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from ami import load_ami_stitch30
from data import SR, load_audio, load_split
from ood_data import chunk_wave, grouped_error_rates, load_ood_items
from kv_template import (
    MAX_ENC_FRAMES,
    apply_silence_template,
    build_silence_template,
    cross_kv_bytes,
    encoder_frames_from_nsamples,
    pad_to_30s,
    register_template_hook,
)
from quant import collect_act_absmax, collect_act_ch_absmax, replace_linears
from silence_collapse import collapse_silence
from whisper_pack import patch_encoder_variable_length

DEVICE = torch.device("cuda")
RESULTS = Path("/workspace/SpeechPTQ/results")
MODEL_ID = "openai/whisper-large-v3"

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp16", "w4a4", "kv"], required=True)
    p.add_argument(
        "--pack",
        default="none",
        choices=["none", "nopad", "collapse"],
        help="none=pad to 30s; nopad=batch-longest (W1); collapse=RMS silence collapse then nopad (W2).",
    )
    p.add_argument("--split", default="test-clean")
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
    p.add_argument(
        "--ami-stitch-sec",
        type=float,
        default=30.0,
        help="AMI only: use cached pause-stitched windows (same files as Qwen3).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=224)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--calib-batches", type=int, default=4)
    p.add_argument(
        "--w-mode",
        default="log",
        choices=["none", "log", "uniform", "mixed_log_encffn", "mixed_log_encoder"],
    )
    p.add_argument(
        "--a-mode",
        default="log_dyn",
        choices=[
            "none",
            "log_static",
            "log_dyn",
            "log_token",
            "log_token_a8",
            "log_dyn_a8",
            "log_ch",
            "uniform_static",
            "uniform_dyn",
            "hybrid",
            "enc_log_dec_token",
            "log_skip_dec_ffn",
            "a4_enc_fc2",
            "a45_fc2",
            "a45_ffn",
            "a45_fc2_dyn",
            "a45_ffn_dyn",
            "enc_ffn_a4",
            "uniform_a8_dyn",
            "a8",
            "mixed_log_encffn",
            "mixed_log_encoder",
            "encoder_a8",
            "mixed_log_encffn_a8",
            "mixed_log_encffn_enc_a8",
            "mixed_log_encattn_a8",
            "mixed_log_encattn_enc_a8",
            "mixed_log_encoder_a8",
            "mixed_log_decoder_a8",
            "mixed_uni_encattn_a4_a8",
            "mixed_uni_encffn_a4_a8",
            "mixed_uni_encoder_a4_a8",
        ],
    )
    p.add_argument(
        "--scope",
        default="nolm",
        choices=["all", "nolm", "blocks", "encoder", "decoder", "enc_fc2", "enc_ffn", "enc_attn", "enc_fc2_dec"],
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--backend", default="fake", choices=["fake", "int4"])
    p.add_argument(
        "--smoothquant-alpha",
        type=float,
        default=0.0,
        help="If >0, apply SmoothQuant (alpha) on scoped Linears before W quant.",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Group size along in_features for uniform/log grids. 0 = grouping-free. "
        "Encoder FFN log stays grouping-free in mixed_log_* modes.",
    )
    p.add_argument(
        "--awq-scales",
        default="",
        help="Path to Edge-ASR AWQ scale dict (awq_scales.pt). Applied before grouped W quant.",
    )
    p.add_argument(
        "--act-group-size",
        type=int,
        default=-1,
        help="Activation group size. -1 = same as --group-size. "
        "0 = per-tensor activations (weight grouping only; on-device proxy).",
    )
    return p.parse_args()


def _compressed_tensors_load_kwargs(model_id: str) -> dict:
    """Packed W4A16 needs decompress-on-load; CompressedLinear is gone."""
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


def load_model(model_id: str = MODEL_ID):
    processor = WhisperProcessor.from_pretrained(model_id)
    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
    kwargs.update(_compressed_tensors_load_kwargs(model_id))
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs
        )
    except Exception:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.to(DEVICE)
    model.eval()
    # greedy, English transcribe, no timestamps
    gen_cfg = model.generation_config
    gen_cfg.num_beams = 1
    gen_cfg.do_sample = False
    gen_cfg.return_timestamps = False
    gen_cfg.max_length = None
    return processor, model


@torch.inference_mode()
def generate_from_features(model, processor, input_features, max_new_tokens: int, language: str = "en"):
    return model.generate(
        input_features,
        language=language,
        task="transcribe",
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_timestamps=False,
    )


def featurize_batch(processor, waves, pack: str):
    if pack == "none":
        waves = [pad_to_30s(w) for w in waves]
        feats = processor(
            waves, sampling_rate=SR, return_tensors="pt"
        ).input_features.to(DEVICE, dtype=torch.float16)
        return feats
    packed = processor(
        waves,
        sampling_rate=SR,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    return packed.input_features.to(DEVICE, dtype=torch.float16)


@torch.inference_mode()
def generate_from_encoder(model, encoder_outputs, max_new_tokens: int, language: str = "en"):
    return model.generate(
        encoder_outputs=encoder_outputs,
        language=language,
        task="transcribe",
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_timestamps=False,
    )


def apply_w4a4(
    model,
    processor,
    items,
    batch_size,
    max_new_tokens,
    calib_batches,
    w_mode,
    a_mode,
    scope,
    backend="fake",
    smoothquant_alpha=0.0,
    group_size=0,
    act_group_size=-1,
    awq_scales=None,
):
    need_calib = a_mode in ("log_static", "uniform_static") or smoothquant_alpha > 0
    stats = {}
    ch_stats = {}
    if need_calib:
        try:
            calib_src = load_split("test-other")
        except FileNotFoundError:
            calib_src = items
        calib = calib_src[: max(batch_size * calib_batches, batch_size)]
        feats_list = []
        for start in range(0, len(calib), batch_size):
            chunk = calib[start : start + batch_size]
            waves = [load_audio(x["path"]) for x in chunk]
            feats = processor(
                [pad_to_30s(w) for w in waves],
                sampling_rate=SR,
                return_tensors="pt",
            ).input_features.to(DEVICE, dtype=torch.float16)
            feats_list.append(feats)

        def gen_fn(feats):
            generate_from_features(model, processor, feats, max_new_tokens)

        print(f"[w4a4] calibrating on {len(calib)} utts / {len(feats_list)} batches")
        if a_mode in ("log_static", "uniform_static"):
            stats = collect_act_absmax(model, feats_list, gen_fn, max_batches=len(feats_list))
        if smoothquant_alpha > 0:
            ch_stats = collect_act_ch_absmax(
                model, feats_list, gen_fn, max_batches=len(feats_list)
            )
            print(f"[w4a4] SmoothQuant alpha={smoothquant_alpha} channels={len(ch_stats)}")
    pre_scales = None
    if awq_scales:
        raw = torch.load(awq_scales, map_location="cpu", weights_only=False)
        pre_scales = {k: v.float() for k, v in raw.items()}
        print(f"[w4a4] AWQ scales {awq_scales} n={len(pre_scales)}")
    n = replace_linears(
        model,
        stats,
        w_mode=w_mode,
        a_mode=a_mode,
        scope=scope,
        backend=backend,
        smoothquant_alpha=smoothquant_alpha,
        act_ch_absmax=ch_stats,
        group_size=group_size,
        act_group_size=act_group_size,
        smooth_scales=pre_scales,
    )
    print(
        f"[w4a4] w={w_mode} a={a_mode} scope={scope} backend={backend} "
        f"sq={smoothquant_alpha} g_w={group_size} g_a={act_group_size} replaced {n} Linear layers"
    )
    model.to(DEVICE)
    model.eval()
    return stats


def _normalize(text: str, language: str, en_norm, basic_norm) -> str:
    lang = (language or "en").split("-")[0].lower()
    if lang == "en":
        return en_norm(text)
    return basic_norm(text)


def load_items(args) -> list[dict]:
    if args.dataset == "ami":
        items = load_ami_stitch30()
        print(f"[ami] loaded {len(items)} cached stitch windows")
        if args.limit and args.limit > 0:
            return items[: args.limit]
        return items
    if args.dataset in ("earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"):
        items = load_ood_items(args.dataset, lang=getattr(args, "lang", "en"), limit=0)
        print(f"[{args.dataset}] loaded {len(items)} cached utts lang={getattr(args, 'lang', 'en')}")
        if args.limit and args.limit > 0:
            return items[: args.limit]
        return items
    items = load_split(args.split)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    return items


def run(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = load_items(args)
    if not items:
        raise RuntimeError(f"no items for dataset={args.dataset} lang={getattr(args, 'lang', '')}")
    print(
        f"dataset={args.dataset} split={args.split} n={len(items)} "
        f"mode={args.mode} pack={getattr(args, 'pack', 'none')} bs={args.batch_size}"
    )

    processor, model = load_model(args.model_id)
    print(f"model={args.model_id}")
    pack = getattr(args, "pack", "none")
    if pack != "none":
        patch_encoder_variable_length(model)
        print(f"[pack] {pack}: variable-length encoder (no 30s canvas)")
    if args.mode == "w4a4":
        apply_w4a4(
            model,
            processor,
            items,
            args.batch_size,
            args.max_new_tokens,
            args.calib_batches,
            args.w_mode,
            args.a_mode,
            args.scope,
            backend=args.backend,
            smoothquant_alpha=args.smoothquant_alpha,
            group_size=getattr(args, "group_size", 0),
            act_group_size=getattr(args, "act_group_size", -1),
            awq_scales=getattr(args, "awq_scales", "") or None,
        )

    template = None
    n_samples_ref = {"n_samples": None}
    kv_hook = None
    if args.mode == "kv":
        print("[kv] building 30s silence encoder template")
        template = build_silence_template(model, processor, DEVICE)
        print(f"[kv] template {tuple(template.shape)} {template.dtype}")
        kv_hook = register_template_hook(model, n_samples_ref, template)

    en_norm = EnglishTextNormalizer()
    basic_norm = BasicTextNormalizer()
    items = sorted(items, key=lambda x: str(x.get("language", "en")))
    hyps: list[str] = []
    refs: list[str] = []
    total_audio = 0.0
    kv_full_bytes = 0
    kv_tmpl_bytes = 0
    sum_enc_frames = 0
    sum_enc_frames_in = 0
    sum_batch_enc_frames = 0
    n_batches = 0
    audio_sec_out = 0.0
    torch.cuda.reset_peak_memory_stats()
    mem_after_load = torch.cuda.memory_allocated()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def decode_short_batch(waves, language: str):
        feats = featurize_batch(processor, waves, pack)
        n_samples = [len(w) for w in waves]
        n_samples_ref["n_samples"] = n_samples
        if pack == "none":
            ids = generate_from_features(model, processor, feats, args.max_new_tokens, language)
            batch_t = MAX_ENC_FRAMES
        else:
            with torch.inference_mode():
                enc_out = model.model.encoder(feats)
            batch_t = int(enc_out.last_hidden_state.shape[1])
            ids = generate_from_encoder(model, enc_out, args.max_new_tokens, language)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        return texts, batch_t

    def decode_long_wave(wave, language: str) -> tuple[str, int]:
        parts = []
        frames = 0
        for cw in chunk_wave(wave):
            texts, batch_t = decode_short_batch([cw], language)
            parts.append(texts[0] if texts else "")
            frames += batch_t
        return " ".join(parts), frames

    try:
        for start in range(0, len(items), args.batch_size):
            chunk = items[start : start + args.batch_size]
            waves = [load_audio(x["path"]) for x in chunk]
            n_in = [len(w) for w in waves]
            total_audio += sum(n_in) / SR
            for ns in n_in:
                sum_enc_frames_in += encoder_frames_from_nsamples(ns)
            if pack == "collapse":
                collapsed = []
                for w in waves:
                    w2, st = collapse_silence(w, hold_frames=1, rel_db=-35.0)
                    collapsed.append(w2)
                    audio_sec_out += st["audio_sec_out"]
                waves = collapsed
            else:
                audio_sec_out += sum(n_in) / SR
            n_samples = [len(w) for w in waves]
            for ns in n_samples:
                n = encoder_frames_from_nsamples(ns)
                kv_full_bytes += cross_kv_bytes(MAX_ENC_FRAMES)
                kv_tmpl_bytes += cross_kv_bytes(n)
                sum_enc_frames += n

            langs = [str(x.get("language", "en")) for x in chunk]
            texts_out: list[str] = [""] * len(chunk)
            # same-language short batches stay batched; long or mixed-lang go one-by-one
            can_batch = len(set(langs)) == 1 and all(ns <= 30 * SR for ns in n_samples)
            if can_batch:
                texts, batch_t = decode_short_batch(waves, langs[0])
                texts_out = list(texts)
                sum_batch_enc_frames += batch_t * len(chunk)
                n_batches += 1
            else:
                for i, (w, lang) in enumerate(zip(waves, langs)):
                    if len(w) > 30 * SR:
                        t, fr = decode_long_wave(w, lang)
                        texts_out[i] = t
                        sum_batch_enc_frames += fr
                    else:
                        texts, batch_t = decode_short_batch([w], lang)
                        texts_out[i] = texts[0] if texts else ""
                        sum_batch_enc_frames += batch_t
                    n_batches += 1

            for t, x in zip(texts_out, chunk):
                lang = str(x.get("language", "en"))
                hyps.append(_normalize(t, lang, en_norm, basic_norm))
                refs.append(_normalize(x["text"], lang, en_norm, basic_norm))

            done = start + len(chunk)
            if done % (args.batch_size * 10) == 0 or done == len(items):
                elapsed = time.perf_counter() - t0
                rtf_so_far = elapsed / max(total_audio, 1e-8)
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                print(
                    f"  {done}/{len(items)}  audio={total_audio:.1f}s  elapsed={elapsed:.1f}s  "
                    f"RTF={rtf_so_far:.4f}  peak_mem={peak:.0f}MiB"
                )
    finally:
        if kv_hook is not None:
            kv_hook.remove()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    wer = corpus_wer(refs, hyps) * 100.0 if refs else 0.0
    rtf = elapsed / max(total_audio, 1e-8)
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    load_mib = mem_after_load / (1024**2)
    template_mib = cross_kv_bytes(MAX_ENC_FRAMES) / (1024**2)
    full_mib = kv_full_bytes / (1024**2)
    speech_mib = kv_tmpl_bytes / (1024**2)

    out = {
        "mode": args.mode,
        "split": args.split,
        "n_utts": len(items),
        "wer": round(wer, 4),
        "rtf": round(rtf, 6),
        "elapsed_sec": round(elapsed, 3),
        "audio_sec": round(total_audio, 3),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": "float16",
        "num_beams": 1,
        "w_mode": getattr(args, "w_mode", None),
        "a_mode": getattr(args, "a_mode", None),
        "scope": getattr(args, "scope", None),
        "backend": getattr(args, "backend", None),
        "smoothquant_alpha": getattr(args, "smoothquant_alpha", 0.0),
        "group_size": getattr(args, "group_size", 0),
        "act_group_size": getattr(args, "act_group_size", -1),
        "gpu_mem_after_load_mib": round(load_mib, 1),
        "gpu_peak_mib": round(peak_mib, 1),
        "cross_kv_full_mib": round(full_mib, 1),
        "cross_kv_speech_mib": round(speech_mib, 1),
        "cross_kv_template_shared_mib": round(template_mib, 1),
        "cross_kv_saved_pct": round(100.0 * (1.0 - speech_mib / max(full_mib, 1e-8)), 2),
        "dataset": args.dataset,
        "ami_stitch_sec": args.ami_stitch_sec if args.dataset == "ami" else None,
        "pack": pack,
        "audio_sec_out": round(audio_sec_out, 3),
        "mean_enc_frames_utt": round(sum_enc_frames / max(len(items), 1), 1),
        "mean_enc_frames_in": round(sum_enc_frames_in / max(len(items), 1), 1),
        "mean_enc_frames_out": round(sum_enc_frames / max(len(items), 1), 1),
        "enc_frames_in": int(sum_enc_frames_in),
        "enc_frames_out": int(sum_enc_frames),
        "enc_frames_saved_pct": round(
            100.0 * (1.0 - sum_enc_frames / max(sum_enc_frames_in, 1)), 2
        ),
        "mean_enc_frames_batch": round(sum_batch_enc_frames / max(len(items), 1), 1),
        "enc_frames_pad30": MAX_ENC_FRAMES,
        "t_saved_vs_pad30_pct": round(
            100.0 * (1.0 - (sum_batch_enc_frames / max(len(items), 1)) / MAX_ENC_FRAMES), 2
        ),
        "lang": getattr(args, "lang", "en"),
    }
    per_lang, micro = grouped_error_rates(items, refs, hyps)
    if len(per_lang) > 1:
        out["wer_by_lang"] = per_lang
        out["wer_micro"] = micro
        out["wer"] = micro
    if args.dataset == "ami":
        tag = f"{args.mode}_ami_{args.split}_stitch{int(args.ami_stitch_sec)}"
    elif args.dataset in ("earnings22", "voxpopuli", "tedlium", "gigaspeech", "commonvoice"):
        tag = f"{args.mode}_{args.dataset}_{args.split}_{getattr(args, 'lang', 'en')}"
    else:
        tag = f"{args.mode}_{args.split}"
    if args.model_id != MODEL_ID:
        tag += "_" + Path(args.model_id).name.replace(" ", "_")
    if pack != "none":
        tag += f"_pack_{pack}"
    if args.mode == "w4a4":
        tag += f"_{args.w_mode}_{args.a_mode}_{args.scope}"
        if getattr(args, "backend", "fake") != "fake":
            tag += f"_{args.backend}"
        sq = getattr(args, "smoothquant_alpha", 0.0) or 0.0
        if sq > 0:
            tag += f"_sq{sq}"
        gsz = getattr(args, "group_size", 0) or 0
        if gsz > 0:
            tag += f"_g{gsz}"
    if args.limit and args.limit > 0:
        tag += f"_n{args.limit}_bs{args.batch_size}"
    path = Path(args.out) if getattr(args, "out", "") else RESULTS / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    run(parse_args())
