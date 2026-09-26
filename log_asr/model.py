"""Whisper loading and batched transcription helpers."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor

from .quantization import apply_ours_w4a4
from .variable_length import enable_variable_length_encoder

DEFAULT_MODEL = "openai/whisper-large-v3"
SAMPLE_RATE = 16_000


def _decompressed_load_kwargs(model_id: str) -> dict:
    config = AutoConfig.from_pretrained(model_id)
    quant_config = getattr(config, "quantization_config", None)
    if not quant_config:
        return {}
    try:
        from transformers.utils.quantization_config import CompressedTensorsConfig
    except ImportError:
        return {}
    if isinstance(quant_config, CompressedTensorsConfig):
        quant_config.run_compressed = False
        return {"quantization_config": quant_config}
    if isinstance(quant_config, dict) and quant_config.get("quant_method") == "compressed-tensors":
        value = dict(quant_config)
        value["run_compressed"] = False
        return {"quantization_config": CompressedTensorsConfig.from_dict(value)}
    return {}


def load_whisper(
    model_id: str = DEFAULT_MODEL,
    mode: str = "w4a4",
    variable_length: bool = True,
    group_size: int = 64,
    device: str = "cuda",
):
    if mode not in {"fp16", "w4a4"}:
        raise ValueError(f"unsupported mode: {mode}")
    processor = WhisperProcessor.from_pretrained(model_id)
    kwargs = {"dtype": torch.float16, "low_cpu_mem_usage": True}
    kwargs.update(_decompressed_load_kwargs(model_id))
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **kwargs
        )
    except (TypeError, ValueError):
        model = WhisperForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.to(device).eval()
    if variable_length:
        enable_variable_length_encoder(model)
    replaced = apply_ours_w4a4(model, group_size=group_size) if mode == "w4a4" else 0
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.generation_config.return_timestamps = False
    return processor, model, replaced


@torch.inference_mode()
def transcribe_batch(
    processor,
    model,
    waveforms: list[np.ndarray],
    max_new_tokens: int = 224,
    language: str = "en",
) -> list[str]:
    features = processor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    ).input_features.to(model.device, dtype=torch.float16)
    token_ids = model.generate(
        features,
        language=language,
        task="transcribe",
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_timestamps=False,
    )
    return processor.batch_decode(token_ids, skip_special_tokens=True)

