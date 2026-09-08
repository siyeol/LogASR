"""Silence-template replacement for Whisper encoder memory (cross-attn KV)."""

from __future__ import annotations

import numpy as np
import torch

SR = 16000
HOP = 160
CONV_STRIDE = 2
SAMPLES_PER_ENC = HOP * CONV_STRIDE  # 320
MAX_ENC_FRAMES = 1500
MAX_SAMPLES = 30 * SR


def pad_to_30s(wave: np.ndarray) -> np.ndarray:
    if wave.shape[0] >= MAX_SAMPLES:
        return wave[:MAX_SAMPLES]
    out = np.zeros(MAX_SAMPLES, dtype=np.float32)
    out[: wave.shape[0]] = wave
    return out


@torch.inference_mode()
def build_silence_template(model, processor, device: torch.device) -> torch.Tensor:
    silence = np.zeros(MAX_SAMPLES, dtype=np.float32)
    feats = processor(
        silence, sampling_rate=SR, return_tensors="pt"
    ).input_features.to(device, dtype=model.dtype)
    enc = model.model.encoder(feats)
    return enc.last_hidden_state[0].contiguous()  # [1500, d]


def encoder_frames_from_nsamples(n_samples: int) -> int:
    n = int(n_samples) // SAMPLES_PER_ENC
    return max(1, min(MAX_ENC_FRAMES, n))


def apply_silence_template(
    encoder_hidden: torch.Tensor,
    n_samples: list[int],
    template: torch.Tensor,
    inplace: bool = True,
) -> torch.Tensor:
    """Replace pad frames with the shared positional silence template.

    encoder_hidden: [B, 1500, d]
    template: [1500, d]
    """
    hs = encoder_hidden if inplace else encoder_hidden.clone()
    for i, ns in enumerate(n_samples):
        n = encoder_frames_from_nsamples(ns)
        if n < MAX_ENC_FRAMES:
            hs[i, n:] = template[n:]
    return hs


def register_template_hook(model, n_samples_ref: dict, template: torch.Tensor):
    """Patch encoder outputs inside generate() so we do not run the encoder twice."""

    def hook(module, args, output):
        hs = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        apply_silence_template(hs, n_samples_ref["n_samples"], template, inplace=True)
        return output

    return model.model.encoder.register_forward_hook(hook)


def cross_kv_bytes(n_frames: int, n_layers: int = 32, d_model: int = 1280, dtype_bytes: int = 2) -> int:
    """Decoder cross-attention K+V storage for one utterance."""
    return n_layers * 2 * int(n_frames) * d_model * dtype_bytes
