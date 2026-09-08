"""GPTQ W4A16 of Qwen3-ASR language_model only (audio encoder left in BF16).

Calibration: LibriSpeech test-other (test-clean held out).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForMultimodalLM, AutoProcessor

from data import load_audio, load_split

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
SAVE_DIR = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm"
NUM_CALIBRATION_SAMPLES = 128


class CalibTorchDS(TorchDataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _as_cpu_tensor(value):
    t = value
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    return torch.as_tensor(t)


def build_calib_loader(processor, n: int) -> DataLoader:
    items = load_split("test-other")[:n]
    rows = []
    keep = ("input_ids", "attention_mask", "input_features", "input_features_mask")
    for it in items:
        audio = load_audio(it["path"])
        packed = processor.apply_transcription_request(audio=audio, language="English")
        row = {}
        for key in keep:
            if key not in packed or packed[key] is None:
                continue
            t = _as_cpu_tensor(packed[key])
            if t.ndim > 0 and t.shape[0] == 1:
                t = t.squeeze(0)
            row[key] = t
        rows.append(row)
        n_tok = int(row["input_ids"].shape[-1])
        print(f"  calib {len(rows)}/{n} id={it['id']} seq={n_tok} feats={tuple(row['input_features'].shape)}")
    return DataLoader(CalibTorchDS(rows), batch_size=1, collate_fn=data_collator)


def data_collator(batch):
    assert len(batch) == 1
    item = batch[0]
    out = {}
    for key, value in item.items():
        t = torch.as_tensor(value)
        if t.ndim == 1 or (key in ("input_features",) and t.ndim == 2):
            t = t.unsqueeze(0)
        if key == "input_features":
            t = t.to(dtype=torch.bfloat16)
        out[key] = t
    return out


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False

    print(f"[calib] building {NUM_CALIBRATION_SAMPLES} samples from test-other")
    ds = build_calib_loader(processor, NUM_CALIBRATION_SAMPLES)

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:.*audio_tower.*",
            "re:.*multi_modal_projector.*",
        ],
        dampening_frac=0.01,
    )

    print("[gptq] oneshot W4A16 on language_model only")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        data_collator=data_collator,
        pipeline="basic",
        output_dir=SAVE_DIR,
        save_compressed=True,
        max_seq_length=2048,
    )
    processor.save_pretrained(SAVE_DIR)
    print(f"saved {SAVE_DIR}")


if __name__ == "__main__":
    main()
