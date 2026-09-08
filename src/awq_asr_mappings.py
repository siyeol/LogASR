"""AWQ layer mappings for Whisper and Qwen3-ASR audio encoder.

llmcompressor has no Whisper / Qwen3ASR registry entry. Encoder blocks are
Whisper-style (self_attn_layer_norm, q/k/v/out, final_layer_norm, fc1/fc2).
"""

from __future__ import annotations

from llmcompressor.modifiers.transform.awq import AWQMapping

# Encoder + decoder. Decoder adds encoder_attn (cross-attn) before FFN.
WHISPER_AWQ_MAPPINGS = [
    AWQMapping(
        "re:.*self_attn_layer_norm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*out_proj$"]),
    AWQMapping(
        "re:.*encoder_attn_layer_norm$",
        [
            "re:.*encoder_attn.q_proj$",
            "re:.*encoder_attn.k_proj$",
            "re:.*encoder_attn.v_proj$",
        ],
    ),
    AWQMapping("re:.*final_layer_norm$", ["re:.*fc1$"]),
    AWQMapping("re:.*fc1$", ["re:.*fc2$"]),
]

# Audio tower only (same block as Whisper encoder; no cross-attn).
QWEN_ENC_AWQ_MAPPINGS = [
    AWQMapping(
        "re:.*self_attn_layer_norm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*out_proj$"]),
    AWQMapping("re:.*final_layer_norm$", ["re:.*fc1$"]),
    AWQMapping("re:.*fc1$", ["re:.*fc2$"]),
]
