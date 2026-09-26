"""Qwen3-ASR loading, Log-ASR encoder quantization, and transcription."""

from __future__ import annotations

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from transformers import AutoConfig

from .quantization import apply_log_asr_w4a4

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
SAMPLE_RATE = 16_000


def _quantization_config(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    return getattr(config, "quantization_config", None)


def is_gptq_w4a16_checkpoint(model_id: str) -> bool:
    config = _quantization_config(model_id)
    if config is None:
        return False
    text = str(config).lower()
    if isinstance(config, dict):
        text += " " + str(config.get("config_groups", "")).lower()
    return "gptq" in text or "compressed-tensors" in text or "compressed_tensors" in text


def _decompressed_load_kwargs(model_id: str) -> dict:
    """Load packed compressed-tensors checkpoints in emulation mode."""
    config = _quantization_config(model_id)
    if not config:
        return {}
    try:
        from transformers.utils.quantization_config import CompressedTensorsConfig
    except ImportError:
        return {}
    if isinstance(config, CompressedTensorsConfig):
        config.run_compressed = False
        return {"quantization_config": config}
    if isinstance(config, dict) and config.get("quant_method") == "compressed-tensors":
        value = dict(config)
        value["run_compressed"] = False
        return {"quantization_config": CompressedTensorsConfig.from_dict(value)}
    return {}


def load_qwen(
    model_id: str = DEFAULT_QWEN_MODEL,
    mode: str = "fp16",
    decoder_checkpoint: str | None = None,
    weight_group_size: int = 64,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
):
    """Load Qwen3-ASR; W4A4 requires a calibrated GPTQ W4A16 checkpoint."""
    if mode not in {"fp16", "w4a4"}:
        raise ValueError(f"unsupported mode: {mode}")
    source = decoder_checkpoint if mode == "w4a4" else model_id
    if mode == "w4a4":
        if not source:
            raise ValueError("--decoder-checkpoint is required for Qwen Log-ASR W4A4")
        if not is_gptq_w4a16_checkpoint(source):
            raise ValueError(
                f"{source!r} has no GPTQ/compressed-tensors quantization_config; "
                "run calibrate_qwen_gptq.py first"
            )

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device,
        "max_inference_batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    kwargs.update(_decompressed_load_kwargs(source))
    asr = Qwen3ASRModel.from_pretrained(source, **kwargs)
    replaced = 0
    if mode == "w4a4":
        replaced = apply_log_asr_w4a4(
            asr.model.thinker.audio_tower,
            weight_group_size=weight_group_size,
        )
        asr.model.to(device).eval()
    return asr, replaced, source


def transcribe_qwen_batch(asr, waveforms: list[np.ndarray], language: str = "English") -> list[str]:
    outputs = asr.transcribe(
        audio=[(np.asarray(wave, dtype=np.float32), SAMPLE_RATE) for wave in waveforms],
        language=language,
    )
    return [item.text for item in outputs]


def build_qwen_inputs(asr, waveforms: list[np.ndarray], language: str = "English"):
    prompts = [asr._build_text_prompt("", language) for _ in waveforms]
    return asr.processor(
        text=prompts,
        audio=[np.asarray(wave, dtype=np.float32) for wave in waveforms],
        return_tensors="pt",
        padding=True,
    )


def audio_token_lengths(asr, waveforms: list[np.ndarray], language: str = "English") -> np.ndarray:
    inputs = build_qwen_inputs(asr, waveforms, language=language)
    audio_token_id = asr.model.thinker.config.audio_token_id
    counts = (inputs["input_ids"] == audio_token_id).sum(dim=-1)
    return counts.detach().cpu().numpy()
