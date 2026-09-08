"""Training-free encoder-side pause merge on Whisper-large-v3.

Encoder runs at native content length (no 30s pad). Consecutive low-RMS
encoder frames are replaced by sqrt(k) * mean. Speech frames are kept.
LibriSpeech test-clean only. Decoder attends to the merged sequence.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from jiwer import wer as corpus_wer
from transformers.modeling_outputs import BaseModelOutput
from whisper.normalizers import EnglishTextNormalizer

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data import SR, load_audio, load_split
from kv_template import SAMPLES_PER_ENC, encoder_frames_from_nsamples
from run_eval import DEVICE, generate_from_encoder, load_model
from silence_collapse import frame_rms, speech_mask_from_rms
from whisper_pack import patch_encoder_variable_length

OUT = Path("/workspace/SpeechPTQ/results/diagnostics/whisper_enc_merge_test-clean.json")
REL_DB = -35.0
MIN_SIL = 2
MAX_NEW = 224
BATCH = 1  # variable merged length; no cross-attn pad mask risk


def merge_hidden(h: torch.Tensor, wave: np.ndarray) -> torch.Tensor:
    """h: [T, d] encoder frames. Align RMS to 20 ms encoder hop."""
    t = int(h.shape[0])
    rms = frame_rms(np.asarray(wave, dtype=np.float32), SAMPLES_PER_ENC)
    if rms.shape[0] < t:
        rms = np.pad(rms, (0, t - int(rms.shape[0])))
    rms = rms[:t]
    speech = speech_mask_from_rms(rms, rel_db=REL_DB)
    pieces: list[torch.Tensor] = []
    i = 0
    while i < t:
        if speech[i]:
            pieces.append(h[i])
            i += 1
            continue
        j = i
        while j < t and not speech[j]:
            j += 1
        k = j - i
        if k < MIN_SIL:
            pieces.extend(list(h[i:j]))
        else:
            mu = h[i:j].float().mean(dim=0)
            pieces.append((mu * (k**0.5)).to(dtype=h.dtype))
        i = j
    if not pieces:
        return h[:1]
    return torch.stack(pieces, dim=0)


def main():
    items = load_split("test-clean")
    processor, model = load_model()
    patch_encoder_variable_length(model)
    model.eval()
    en_norm = EnglishTextNormalizer()
    hyps, refs = [], []
    frames_in = frames_out = 0
    audio_sec = 0.0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    print(f"[enc-merge] n={len(items)} rel_db={REL_DB} min_sil={MIN_SIL} sqrtk=1", flush=True)
    with torch.inference_mode():
        for i, it in enumerate(items):
            wave = load_audio(it["path"])
            ns = int(wave.shape[0])
            audio_sec += ns / SR
            n_in = encoder_frames_from_nsamples(ns)
            frames_in += n_in
            feats = processor(
                [wave],
                sampling_rate=SR,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_features.to(DEVICE, dtype=torch.float16)
            enc = model.model.encoder(feats)
            h = enc.last_hidden_state[0, :n_in]
            hm = merge_hidden(h, wave)
            frames_out += int(hm.shape[0])
            ids = generate_from_encoder(
                model,
                BaseModelOutput(last_hidden_state=hm.unsqueeze(0)),
                MAX_NEW,
            )
            text = processor.batch_decode(ids, skip_special_tokens=True)[0]
            hyps.append(en_norm(text))
            refs.append(en_norm(it["text"]))
            if (i + 1) % 80 == 0 or i == 0 or i + 1 == len(items):
                elapsed = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                print(
                    f"  {i+1}/{len(items)}  frames {frames_in}->{frames_out}  "
                    f"elapsed={elapsed:.1f}s  peak={peak:.0f}MiB",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    wer = corpus_wer(refs, hyps) * 100.0
    out = {
        "model": "openai/whisper-large-v3",
        "split": "test-clean",
        "n_utts": len(items),
        "wer": round(wer, 4),
        "rtf": round(elapsed / max(audio_sec, 1e-8), 6),
        "elapsed_sec": round(elapsed, 3),
        "audio_sec": round(audio_sec, 3),
        "enc_frames_in": int(frames_in),
        "enc_frames_out": int(frames_out),
        "mean_enc_frames_in": round(frames_in / len(items), 1),
        "mean_enc_frames_out": round(frames_out / len(items), 1),
        "enc_frames_saved_pct": round(100.0 * (1.0 - frames_out / max(frames_in, 1)), 2),
        "rel_db": REL_DB,
        "min_sil_frames": MIN_SIL,
        "sqrtk": True,
        "layer": "encoder_hidden",
        "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "note": "Native-rate encoder (no 30s pad). Pause runs merged after encoder.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
