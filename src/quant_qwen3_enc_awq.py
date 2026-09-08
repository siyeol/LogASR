"""AWQ of Qwen3-ASR audio encoder (Edge-ASR scale search).

Starts from the LLM-GPTQ checkpoint. language_model stays GPTQ W4A16.
Calibration: LibriSpeech test-other (test-clean held out).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from awq_local import collect_linear_inputs, save_awq_ckpt, search_awq_scales
from data import load_audio, load_split
from run_eval_qwen3 import DEVICE, compressed_tensors_load_kwargs, item_wave

MODEL_ID = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm"
SAVE_DIR = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm-encawq"
NUM_CALIBRATION_SAMPLES = 32
GROUP_SIZE = 64
SCOPE = "encoder"


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    kw = dict(dtype=torch.bfloat16)
    kw.update(compressed_tensors_load_kwargs(MODEL_ID))
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **kw)
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False

    items = load_split("test-other")[:NUM_CALIBRATION_SAMPLES]
    print(f"[calib] building {len(items)} samples from test-other")
    batches = []
    for it in items:
        packed = processor.apply_transcription_request(
            audio=item_wave(it), language="English"
        )
        batches.append(packed.to(DEVICE, model.dtype))
        print(f"  calib {len(batches)}/{len(items)} id={it['id']}")

    def gen_fn(inputs):
        model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    caches = collect_linear_inputs(model, gen_fn, batches, SCOPE)
    print(f"[awq] cached {len(caches)} encoder Linear inputs")
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
