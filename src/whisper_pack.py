"""Variable-length Whisper encoder: skip the hard 3000-mel / 1500-frame pad."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput


def patch_encoder_variable_length(model) -> None:
    """Allow T_mel != 3000. Uses the first T encoder position embeddings."""
    enc = model.model.encoder
    if getattr(enc, "_speechptq_varlen", False):
        return

    @torch.inference_mode()
    def packed_forward(input_features, attention_mask=None, **kwargs):
        if input_features.shape[-1] % 2 == 1:
            input_features = F.pad(input_features, (0, 1))
        inputs_embeds = F.gelu(enc.conv1(input_features))
        inputs_embeds = F.gelu(enc.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        t = inputs_embeds.shape[1]
        max_pos = enc.embed_positions.num_embeddings
        if t > max_pos:
            inputs_embeds = inputs_embeds[:, :max_pos]
            t = max_pos
        hidden_states = inputs_embeds + enc.embed_positions.weight[:t]
        hidden_states = F.dropout(hidden_states, p=enc.dropout, training=enc.training)
        for encoder_layer in enc.layers:
            hidden_states = encoder_layer(hidden_states, None, **kwargs)
        hidden_states = enc.layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    enc.forward = packed_forward
    enc._speechptq_varlen = True
