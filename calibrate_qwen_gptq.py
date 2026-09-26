#!/usr/bin/env python3
"""Calibrate the Qwen3-ASR text decoder with GPTQ W4A16 (G=128)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from qwen_asr import Qwen3ASRModel

from log_asr.audio import load_audio, load_librispeech
from log_asr.qwen import build_qwen_inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="dev-clean")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output", default="models/qwen3-asr-1.7b-decoder-gptq-w4a16-g128")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--pipeline", choices=["basic", "sequential"], default="sequential")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def _collate(batch):
    result = {}
    for key, value in batch[0].items():
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1 or (key == "input_features" and tensor.ndim == 2):
            tensor = tensor.unsqueeze(0)
        if key == "input_features":
            tensor = tensor.to(torch.bfloat16)
        result[key] = tensor
    return result


def main():
    args = parse_args()
    items = load_librispeech(args.data_root, args.split)
    selected = random.Random(args.seed).sample(items, min(args.samples, len(items)))
    asr = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=32,
    )
    rows = []
    for index, item in enumerate(selected, 1):
        request = build_qwen_inputs(
            asr, [load_audio(item["path"])], language="English"
        )
        row = {}
        for key, value in request.items():
            if key not in {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}:
                continue
            tensor = value[0].detach().cpu()
            if key in {"input_ids", "attention_mask"}:
                tensor = tensor[: args.max_seq_length]
            row[key] = tensor.numpy()
        rows.append(row)
        if index % 16 == 0:
            print(f"prepared {index}/{len(selected)}", flush=True)

    # Calibrate the thinker directly: unlike the top-level generation wrapper,
    # it has a real forward method that the sequential tracer can partition.
    calibration_model = asr.model.thinker
    # Avoid a transformers save-time validation failure in the upstream config.
    asr.model.generation_config.temperature = None
    if args.group_size != 128:
        raise ValueError("llmcompressor's W4A16 preset uses the paper setting G=128")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["re:.*lm_head.*", "re:.*audio_tower.*"],
        sequential_targets="Qwen3ASRThinkerTextDecoderLayer",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=calibration_model,
        processor=asr.processor,
        dataset=HFDataset.from_list(rows),
        recipe=recipe,
        num_calibration_samples=len(rows),
        data_collator=_collate,
        pipeline=args.pipeline,
        output_dir=None,
        save_compressed=False,
        max_seq_length=args.max_seq_length,
    )
    from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained

    modify_save_pretrained(asr.model)
    asr.model.save_pretrained(output, save_compressed=True, max_shard_size="1GB")
    asr.processor.save_pretrained(output)
    metadata = {
        "base_model": args.model_id,
        "component": "text_decoder",
        "method": "GPTQ",
        "precision": "W4A16",
        "weight_group_size": args.group_size,
        "calibration_samples": len(rows),
        "calibration_split": args.split,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "pipeline": args.pipeline,
    }
    (output / "log_asr_gptq.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    if not args.skip_reload_check:
        del asr
        torch.cuda.empty_cache()
        from log_asr.qwen import load_qwen, transcribe_qwen_batch

        checked, _, _ = load_qwen(
            mode="w4a4", decoder_checkpoint=str(output), batch_size=1, max_new_tokens=64
        )
        text = transcribe_qwen_batch(checked, [load_audio(selected[0]["path"])])[0]
        if not text.strip() or "ModelError" in text:
            raise RuntimeError(f"GPTQ reload smoke test failed: {text!r}")
        print(f"reload smoke test: {text}")


if __name__ == "__main__":
    main()
