"""AWQ W4A16 of Whisper-large-v3 via Edge-ASR scale search.

Calibration: LibriSpeech test-other (test-clean held out).
Eval later applies grouped uniform W4 g=64 + grouping-free uniform A8
using the saved per-channel scales (w_mode=uniform, --awq-scales).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from awq_local import collect_linear_inputs, save_awq_ckpt, search_awq_scales
from data import SR, load_audio, load_split
from kv_template import pad_to_30s
from run_eval import DEVICE, generate_from_features

MODEL_ID = "openai/whisper-large-v3"
SAVE_DIR = "/workspace/SpeechPTQ/models/whisper-large-v3-AWQ-W4A16-G128"
NUM_CALIBRATION_SAMPLES = 32
GROUP_SIZE = 64
SCOPE = "nolm"


def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    model.to(DEVICE)
    model.eval()
    model.config.forced_decoder_ids = None

    items = load_split("test-other")[:NUM_CALIBRATION_SAMPLES]
    print(f"[calib] building {len(items)} samples from test-other")
    batches = []
    for it in items:
        audio = pad_to_30s(load_audio(it["path"]))
        feats = processor(
            audio,
            sampling_rate=SR,
            return_tensors="pt",
        ).input_features.to(DEVICE, dtype=torch.float16)
        batches.append(feats)

    def gen_fn(feats):
        generate_from_features(model, processor, feats, max_new_tokens=32)

    caches = collect_linear_inputs(model, gen_fn, batches, SCOPE)
    print(f"[awq] cached {len(caches)} Linear inputs")
    scales = search_awq_scales(model, caches, SCOPE, group_size=GROUP_SIZE)
    save_awq_ckpt(
        SAVE_DIR,
        scales,
        {
            "base_model": MODEL_ID,
            "scope": SCOPE,
            "group_size": GROUP_SIZE,
            "n_calib": NUM_CALIBRATION_SAMPLES,
            "n_grid": 20,
            "formula": "Edge-ASR AWQ s=xmax^a/wmax^(1-a)",
        },
    )


if __name__ == "__main__":
    main()
