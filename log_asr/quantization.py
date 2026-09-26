"""Reference fake quantization for Log-ASR."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 7
LOG_DYNAMIC_RANGE = 2**QMAX - 1  # mu = 127 for signed 4-bit indices


def _group_last_dim(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    original_size = int(x.shape[-1])
    padding = (-original_size) % group_size
    if padding:
        x = F.pad(x, (0, padding))
    return x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size), original_size


def _restore_last_dim(x: torch.Tensor, original_size: int) -> torch.Tensor:
    flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return flat[..., :original_size]


def uniform_w4_fake_quant(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Symmetric group-wise uniform INT4 quantize-dequantize for weights."""
    dtype = x.dtype
    groups, original_size = _group_last_dim(x.float(), group_size)
    absmax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = absmax / QMAX
    codes = torch.clamp(torch.round(groups / scale), -QMAX, QMAX)
    return _restore_last_dim(codes * scale, original_size).to(dtype)


def log_a4_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Grouping-free per-token Log-ASR activation quantize-dequantize.

    The final dimension is the channel dimension. One alpha is shared by every
    channel in each token; no activation-side channel groups are used.
    """
    dtype = x.dtype
    values = x.float()
    token_absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    alpha = token_absmax / LOG_DYNAMIC_RANGE
    mapped = torch.sign(values) * torch.log2(1.0 + values.abs() / alpha)
    codes = torch.clamp(torch.round(mapped), -QMAX, QMAX)
    restored = torch.sign(codes) * alpha * (torch.pow(2.0, codes.abs()) - 1.0)
    return restored.to(dtype)


class LogASRW4A4Linear(nn.Module):
    """Uniform group-W4 Linear with grouping-free per-token Log-A4 inputs."""

    def __init__(self, linear: nn.Linear, weight_group_size: int = 64):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_group_size = int(weight_group_size)
        weight = uniform_w4_fake_quant(linear.weight.detach(), self.weight_group_size)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(log_a4_fake_quant(x), self.weight, self.bias)


def apply_log_asr_w4a4(
    module: nn.Module,
    weight_group_size: int = 64,
    predicate: Callable[[str, nn.Linear], bool] | None = None,
    prefix: str = "",
) -> int:
    """Replace selected Linear layers with the Log-ASR reference layer."""
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if predicate is None or predicate(full_name, child):
                setattr(
                    module,
                    name,
                    LogASRW4A4Linear(child, weight_group_size=weight_group_size),
                )
                replaced += 1
        else:
            replaced += apply_log_asr_w4a4(
                child,
                weight_group_size=weight_group_size,
                predicate=predicate,
                prefix=full_name,
            )
    return replaced


# Backward-compatible name used by the Whisper CLI.
def apply_ours_w4a4(module: nn.Module, group_size: int = 64) -> int:
    return apply_log_asr_w4a4(module, weight_group_size=group_size)
