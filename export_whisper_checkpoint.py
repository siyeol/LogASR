#!/usr/bin/env python3
"""Export Whisper Log-ASR weights as a compressed W4 (G=64) checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-large-v3")
    parser.add_argument("--output", default="checkpoints/whisper-large-v3-log-asr-w4a4-g64")
    parser.add_argument("--group-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.generation_config.return_timestamps = False

    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
            "num_bits": 4,
            "type": "int",
            "symmetric": True,
            "strategy": "group",
            "group_size": args.group_size,
        },
            "input_activations": None,
            "output_activations": None,
        }
    }
    recipe = QuantizationModifier(config_groups=config_groups)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=model,
        processor=processor,
        recipe=recipe,
        output_dir=str(output),
        save_compressed=True,
    )
    processor.save_pretrained(output)

    metadata = {
        "base_model": args.model_id,
        "weight_quantization": "symmetric uniform W4",
        "weight_group_size": args.group_size,
        "activation_quantization": "Log-ASR A4 per-token, mu=127 (runtime recipe)",
        "silence_packaging": "optional runtime recipe; no additional learned parameters",
    }
    (output / "log_asr_checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "recipe_w4a4.json").write_text(json.dumps({
        "mode": "w4a4",
        "silence_packaging": False,
        "weight_group_size": args.group_size,
        "activation_mu": 127,
    }, indent=2) + "\n")
    (output / "recipe_w4a4_sfp.json").write_text(json.dumps({
        "mode": "w4a4",
        "silence_packaging": True,
        "frame_ms": 80.0,
        "relative_db": -35.0,
        "absolute_floor": 1e-4,
        "min_silence_frames": 2,
        "hold_frames": 1,
    }, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
