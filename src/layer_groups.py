"""Classify Linear names into encoder/LLM role groups."""

from __future__ import annotations

from quant import is_language_decoder, is_speech_encoder


def layer_group(name: str) -> str:
    n = name.lower()
    if n.endswith("lm_head") or n.endswith("proj_out"):
        return "lm_head"
    if "multi_modal_projector" in n or n.endswith("proj1") or n.endswith("proj2"):
        return "projector"
    enc = is_speech_encoder(n)
    dec = is_language_decoder(n)
    is_ffn = n.endswith(("fc1", "fc2", "gate_proj", "up_proj", "down_proj"))
    is_attn = any(
        k in n
        for k in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "qkv",
            "encoder_attn",
            "self_attn",
        )
    ) and not is_ffn
    if enc and is_ffn:
        return "enc_ffn"
    if enc and is_attn:
        return "enc_attn"
    if enc:
        return "enc_other"
    if dec and is_ffn:
        return "llm_ffn"
    if dec and is_attn:
        return "llm_attn"
    if dec:
        return "llm_other"
    return "other"
