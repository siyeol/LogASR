"""Variable-length Whisper encoder used after silence packaging."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

SAMPLES_PER_ENCODER_FRAME = 320
MAX_ENCODER_FRAMES = 1500


def encoder_frames(num_samples: int) -> int:
    return max(1, min(MAX_ENCODER_FRAMES, int(num_samples) // SAMPLES_PER_ENCODER_FRAME))


def enable_variable_length_encoder(model) -> None:
    """Patch Whisper to accept fewer than 3000 mel frames."""
    encoder = model.model.encoder
    if getattr(encoder, "_ours_variable_length", False):
        return

    @torch.inference_mode()
    def packed_forward(input_features, attention_mask=None, **kwargs):
        del attention_mask
        if input_features.shape[-1] % 2:
            input_features = F.pad(input_features, (0, 1))
        hidden = F.gelu(encoder.conv1(input_features))
        hidden = F.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        length = min(hidden.shape[1], encoder.embed_positions.num_embeddings)
        hidden = hidden[:, :length] + encoder.embed_positions.weight[:length]
        hidden = F.dropout(hidden, p=encoder.dropout, training=encoder.training)
        for layer in encoder.layers:
            hidden = layer(
                hidden,
                None,
                None,
                output_attentions=bool(kwargs.get("output_attentions", False)),
            )[0]
        return BaseModelOutput(last_hidden_state=encoder.layer_norm(hidden))

    encoder.forward = packed_forward
    encoder._ours_variable_length = True

