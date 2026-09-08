"""GPTQ W4A16 on Whisper encoder FFN only (fc1/fc2). Decoder stays FP16."""

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
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data import SR, load_audio, load_split

MODEL_ID = "openai/whisper-large-v3"
SAVE_DIR = "/workspace/SpeechPTQ/models/whisper-large-v3-encffn-W4A16-G128"
NUM_CALIBRATION_SAMPLES = 128


class CalibTorchDS(TorchDataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def build_calib_loader(processor, model_dtype, n: int) -> DataLoader:
    items = load_split("test-other")[:n]
    rows = []
    for it in items:
        audio = load_audio(it["path"])
        text = " " + it["text"].capitalize()
        inputs = processor(
            audio=audio,
            sampling_rate=SR,
            text=text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        rows.append(
            {
                "input_features": inputs["input_features"].to(dtype=model_dtype).squeeze(0),
                "decoder_input_ids": inputs["labels"].squeeze(0),
            }
        )
    return DataLoader(CalibTorchDS(rows), batch_size=1, collate_fn=data_collator)


def data_collator(batch):
    assert len(batch) == 1
    item = batch[0]
    out = {}
    for key, value in item.items():
        t = torch.as_tensor(value)
        if key == "input_features" and t.ndim == 2:
            t = t.unsqueeze(0)
        if key == "decoder_input_ids" and t.ndim == 1:
            t = t.unsqueeze(0)
        out[key] = t
    return out


def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.forced_decoder_ids = None

    print(f"[calib] building {NUM_CALIBRATION_SAMPLES} samples from test-other")
    ds = build_calib_loader(processor, model.dtype, NUM_CALIBRATION_SAMPLES)

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:.*proj_out",
            "re:.*decoder.*",
            "re:.*self_attn.*",
        ],
        dampening_frac=0.01,
    )

    print("[gptq] oneshot W4A16 on encoder FFN only")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        data_collator=data_collator,
        pipeline="basic",
        output_dir=SAVE_DIR,
        save_compressed=True,
        max_seq_length=448,
    )
    processor.save_pretrained(SAVE_DIR)
    print(f"saved {SAVE_DIR}")


if __name__ == "__main__":
    main()
