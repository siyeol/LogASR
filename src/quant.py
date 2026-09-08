"""Log-scale and uniform fake-quant for Whisper / Qwen Linear layers.

Log codebook (grouping-free, 4-bit):
    alpha = absmax / 127
    f(x)  = sign(x) * log2(1 + |x| / alpha)
    y     = clip(round(f(x)), -7, 7)
    xhat  = sign(y) * alpha * (2^|y| - 1)

8-bit log (`log_token_a8` / `log_dyn_a8`) keeps the same curvature
(alpha = absmax / 127, reconstruction saturates at 2^7 - 1) but stores
the log-domain value as int8 (y in [-127, 127]):
    f8    = f * (127 / 7)
    y     = clip(round(f8), -127, 127)
    xhat  = sign(y) * alpha * (2^{|y| * 7 / 127} - 1)

Dynamic alpha uses the tensor (or token) absmax at runtime so eval
    outliers are not clipped by a too-small calibration max.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX = 7
QMAX_A8 = 127


def _absmax(x: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return x.abs().amax().clamp_min(1e-8)
    return x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8)


def log_quant_dequant(x: torch.Tensor, alpha, qmax: int = QMAX) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.float()
    if not torch.is_tensor(alpha):
        a = torch.tensor(max(float(alpha), 1e-8), device=x.device, dtype=torch.float32)
    else:
        a = alpha.float().clamp_min(1e-8)
    f = torch.sign(x32) * torch.log2(1.0 + x32.abs() / a)
    if qmax == QMAX:
        y = torch.clamp(torch.round(f), -qmax, qmax)
        xhat = torch.sign(y) * a * (torch.pow(2.0, y.abs()) - 1.0)
    else:
        # Same 4-bit log curve; finer integer grid (typically qmax=127).
        scale = float(qmax) / float(QMAX)
        y = torch.clamp(torch.round(f * scale), -qmax, qmax)
        xhat = torch.sign(y) * a * (torch.pow(2.0, y.abs() / scale) - 1.0)
    return xhat.to(orig_dtype)


def uniform_quant_dequant(x: torch.Tensor, absmax, qmax: int = QMAX) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.float()
    if not torch.is_tensor(absmax):
        am = torch.tensor(max(float(absmax), 1e-8), device=x.device, dtype=torch.float32)
    else:
        am = absmax.float().clamp_min(1e-8)
    scale = am / qmax
    y = torch.clamp(torch.round(x32 / scale), -qmax, qmax)
    return (y * scale).to(orig_dtype)


def alpha_from_absmax(absmax: float, qmax: int = QMAX) -> float:
    return max(float(absmax), 1e-8) / (2**qmax - 1)


def _grouped_view(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int]:
    """Reshape last dim into groups; pad if needed. Returns (grouped, orig_k, pad)."""
    k = int(x.shape[-1])
    pad = (-k) % group_size
    if pad:
        x = F.pad(x, (0, pad))
    grouped_k = x.shape[-1]
    return x.reshape(*x.shape[:-1], grouped_k // group_size, group_size), k, pad


def _ungroup(g: torch.Tensor, orig_k: int) -> torch.Tensor:
    x = g.reshape(*g.shape[:-2], g.shape[-2] * g.shape[-1])
    return x[..., :orig_k]


def log_quant_dequant_grouped(x: torch.Tensor, group_size: int, qmax: int = QMAX) -> torch.Tensor:
    orig_dtype = x.dtype
    g, orig_k, _ = _grouped_view(x.float(), group_size)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    # Curvature is always the 4-bit log codebook (absmax / 127).
    alpha = amax / (2**QMAX - 1)
    f = torch.sign(g) * torch.log2(1.0 + g.abs() / alpha)
    if qmax == QMAX:
        y = torch.clamp(torch.round(f), -qmax, qmax)
        xhat = torch.sign(y) * alpha * (torch.pow(2.0, y.abs()) - 1.0)
    else:
        scale = float(qmax) / float(QMAX)
        y = torch.clamp(torch.round(f * scale), -qmax, qmax)
        xhat = torch.sign(y) * alpha * (torch.pow(2.0, y.abs() / scale) - 1.0)
    return _ungroup(xhat, orig_k).to(orig_dtype)


def uniform_quant_dequant_grouped(x: torch.Tensor, group_size: int, qmax: int = QMAX) -> torch.Tensor:
    orig_dtype = x.dtype
    g, orig_k, _ = _grouped_view(x.float(), group_size)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    y = torch.clamp(torch.round(g / scale), -qmax, qmax)
    return _ungroup(y * scale, orig_k).to(orig_dtype)


def is_speech_encoder(name: str) -> bool:
    n = name.lower()
    return ".encoder." in f".{n}." or "audio_tower" in n


def is_language_decoder(name: str) -> bool:
    n = name.lower()
    return ".decoder." in f".{n}." or "language_model" in n


def match_scope(name: str, scope: str) -> bool:
    n = name.lower()
    is_lm = n.endswith("proj_out") or n.endswith("lm_head") or ".proj_out" in n
    if scope == "all":
        return True
    if is_lm:
        return False
    if scope in ("nolm", "blocks"):
        return True
    if scope == "encoder":
        return is_speech_encoder(n)
    if scope == "decoder":
        return is_language_decoder(n)
    if scope == "enc_fc2":
        return is_speech_encoder(n) and n.endswith("fc2")
    if scope == "enc_ffn":
        return is_speech_encoder(n) and (n.endswith("fc1") or n.endswith("fc2"))
    if scope == "enc_attn":
        return is_speech_encoder(n) and (
            n.endswith("q_proj")
            or n.endswith("k_proj")
            or n.endswith("v_proj")
            or n.endswith("out_proj")
        )
    if scope == "enc_fc2_dec":
        return n.endswith("fc2") or is_language_decoder(n)
    raise ValueError(f"unknown scope {scope}")


def act_mode_for_layer(name: str, a_mode: str) -> str:
    n = name.lower()
    enc = is_speech_encoder(n)
    dec = is_language_decoder(n)
    is_ffn = n.endswith("fc1") or n.endswith("fc2")
    if a_mode == "hybrid":
        if enc and n.endswith("fc2"):
            return "log_dyn"
        return "uniform_dyn"
    if a_mode == "enc_log_dec_token":
        return "log_dyn" if enc else "log_token"
    if a_mode == "log_skip_dec_ffn":
        if dec and is_ffn:
            return "none"
        return "log_dyn"
    if a_mode == "a4_enc_fc2":
        if enc and n.endswith("fc2"):
            return "log_dyn"
        return "none"
    if a_mode == "a45_fc2":
        if enc and n.endswith("fc2"):
            return "uniform_a8_dyn"
        return "log_token"
    if a_mode == "a45_ffn":
        if enc and is_ffn:
            return "uniform_a8_dyn"
        return "log_token"
    if a_mode == "a45_fc2_dyn":
        if enc and n.endswith("fc2"):
            return "uniform_a8_dyn"
        return "log_dyn"
    if a_mode == "a45_ffn_dyn":
        if enc and is_ffn:
            return "uniform_a8_dyn"
        return "log_dyn"
    if a_mode == "enc_ffn_a4":
        # Mixed PTQ: log A4 only on speech-encoder FFN; remaining Linears are W-only (A16).
        if enc and is_ffn:
            return "log_token"
        return "none"
    if a_mode in ("a8", "uniform_a8", "uniform_a8_dyn"):
        return "uniform_a8_dyn"
    if a_mode == "mixed_log_encffn":
        if enc and is_ffn:
            return "log_token"
        return "uniform_dyn"
    if a_mode == "mixed_log_encoder":
        return "log_token" if enc else "uniform_dyn"
    if a_mode == "encoder_a8":
        return "uniform_a8_dyn" if enc else "none"
    if a_mode == "mixed_log_encffn_a8":
        if enc and is_ffn:
            return "log_token"
        return "uniform_a8_dyn"
    if a_mode == "mixed_log_encffn_enc_a8":
        if enc and is_ffn:
            return "log_token"
        if enc:
            return "uniform_a8_dyn"
        return "none"
    is_ffn_full = n.endswith(("fc1", "fc2", "gate_proj", "up_proj", "down_proj"))
    is_attn = (
        any(
            k in n
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "qkv", "self_attn")
        )
        and not is_ffn_full
    )
    if a_mode == "mixed_log_encattn_a8":
        if enc and is_attn:
            return "log_token"
        return "uniform_a8_dyn"
    if a_mode == "mixed_log_encattn_enc_a8":
        if enc and is_attn:
            return "log_token"
        if enc:
            return "uniform_a8_dyn"
        return "none"
    if a_mode == "mixed_log_encoder_a8":
        # Encoder proposed W4A4 (log A); remaining Linears uniform A8.
        return "log_token" if enc else "uniform_a8_dyn"
    if a_mode == "mixed_log_decoder_a8":
        # Decoder/LLM proposed W4A4 (log A); encoder (and other) uniform A8.
        return "log_token" if dec else "uniform_a8_dyn"
    if a_mode == "mixed_uni_encattn_a4_a8":
        if enc and is_attn:
            return "uniform_dyn"
        return "uniform_a8_dyn"
    if a_mode == "mixed_uni_encffn_a4_a8":
        if enc and is_ffn_full:
            return "uniform_dyn"
        return "uniform_a8_dyn"
    if a_mode == "mixed_uni_encoder_a4_a8":
        return "uniform_dyn" if enc else "uniform_a8_dyn"
    return a_mode


def weight_mode_for_layer(name: str, w_mode: str) -> str:
    n = name.lower()
    enc = is_speech_encoder(n)
    is_ffn = n.endswith("fc1") or n.endswith("fc2")
    if w_mode == "mixed_log_encffn":
        return "log" if enc and is_ffn else "uniform"
    if w_mode == "mixed_log_encoder":
        return "log" if enc else "uniform"
    return w_mode


def group_size_for_layer(name: str, w_mode: str, group_size: int) -> int:
    """Apply group_size to all scoped Linears.

    mixed_log_* keeps encoder-FFN log grouping-free; plain --w-mode log uses
    grouped log on every layer including encoder FFN.
    """
    if group_size <= 0:
        return 0
    if w_mode not in ("mixed_log_encffn", "mixed_log_encoder"):
        return group_size
    n = name.lower()
    resolved = weight_mode_for_layer(name, w_mode)
    if (
        resolved == "log"
        and is_speech_encoder(n)
        and (n.endswith("fc1") or n.endswith("fc2"))
    ):
        return 0
    return group_size


def smoothquant_scales(weight: torch.Tensor, act_ch_absmax: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-input-channel SmoothQuant scales. W' = W * s, X' = X / s."""
    w_max = weight.detach().abs().amax(dim=0).float().clamp_min(1e-8)
    a_max = act_ch_absmax.detach().float().reshape(-1).to(w_max.device).clamp_min(1e-8)
    if a_max.numel() != w_max.numel():
        raise ValueError(f"SmoothQuant dim mismatch: act {tuple(a_max.shape)} vs in {tuple(w_max.shape)}")
    return a_max.pow(alpha) / w_max.pow(1.0 - alpha)


class QuantLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        name: str,
        w_mode: str,
        a_mode: str,
        act_absmax: float,
        smooth_scale: torch.Tensor | None = None,
        group_size: int = 0,
        act_group_size: int | None = None,
    ):
        super().__init__()
        self.name = name
        self.w_mode = weight_mode_for_layer(name, w_mode)
        self.a_mode = act_mode_for_layer(name, a_mode)
        self.group_size = group_size_for_layer(name, w_mode, group_size)
        # None / negative → activations use the same groups as weights.
        if act_group_size is None or act_group_size < 0:
            self.act_group_size = self.group_size
        else:
            self.act_group_size = int(act_group_size)
        w = linear.weight.data
        if smooth_scale is not None:
            s = smooth_scale.detach().to(device=w.device, dtype=w.dtype).reshape(-1)
            w = w * s.unsqueeze(0)
            inv = (1.0 / s.float().clamp_min(1e-8)).to(dtype=w.dtype)
            self.register_buffer("inv_smooth", inv)
        else:
            self.inv_smooth = None
        gsz = self.group_size
        if self.w_mode == "none":
            w_hat = w
        elif self.w_mode == "log":
            if gsz > 0:
                w_hat = log_quant_dequant_grouped(w, gsz)
            else:
                w_hat = log_quant_dequant(w, alpha_from_absmax(w.abs().amax().item()))
        elif self.w_mode == "uniform":
            if gsz > 0:
                w_hat = uniform_quant_dequant_grouped(w, gsz)
            else:
                w_hat = uniform_quant_dequant(w, w.abs().amax().item())
        else:
            raise ValueError(self.w_mode)
        self.weight = nn.Parameter(w_hat, requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(
            linear.bias.data.clone(), requires_grad=False
        )
        self.a_alpha = alpha_from_absmax(act_absmax)
        self.a_absmax = max(float(act_absmax), 1e-8)

    def _quant_act(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.a_mode
        gsz = self.act_group_size
        if mode in ("none", "fp16"):
            return x
        if gsz > 0 and mode in ("log_dyn", "log_token", "log_static", "log_ch"):
            return log_quant_dequant_grouped(x, gsz)
        if gsz > 0 and mode in ("log_dyn_a8", "log_token_a8"):
            return log_quant_dequant_grouped(x, gsz, qmax=QMAX_A8)
        if gsz > 0 and mode in ("uniform_dyn", "uniform_static"):
            return uniform_quant_dequant_grouped(x, gsz)
        if mode == "log_static":
            return log_quant_dequant(x, self.a_alpha)
        if mode == "log_dyn":
            amax = _absmax(x)
            return log_quant_dequant(x, amax / (2**QMAX - 1))
        if mode == "log_token":
            amax = _absmax(x, dim=-1)
            return log_quant_dequant(x, amax / (2**QMAX - 1))
        if mode == "log_dyn_a8":
            amax = _absmax(x)
            return log_quant_dequant(x, amax / (2**QMAX - 1), qmax=QMAX_A8)
        if mode == "log_token_a8":
            amax = _absmax(x, dim=-1)
            return log_quant_dequant(x, amax / (2**QMAX - 1), qmax=QMAX_A8)
        if mode == "log_ch":
            dims = tuple(range(x.ndim - 1))
            amax = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
            return log_quant_dequant(x, amax / (2**QMAX - 1))
        if mode == "uniform_static":
            return uniform_quant_dequant(x, self.a_absmax)
        if mode == "uniform_dyn":
            return uniform_quant_dequant(x, _absmax(x))
        if mode == "uniform_a8_dyn":
            if gsz > 0:
                return uniform_quant_dequant_grouped(x, gsz, qmax=127)
            return uniform_quant_dequant(x, _absmax(x), qmax=127)
        raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_smooth is not None:
            x = x * self.inv_smooth.to(dtype=x.dtype)
        return F.linear(self._quant_act(x), self.weight, self.bias)


def collect_act_absmax(model: nn.Module, calib_features, generate_fn, max_batches: int = 4) -> dict[str, float]:
    stats: dict[str, float] = defaultdict(float)
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, out):
            x = inp[0]
            if not torch.is_tensor(x):
                return
            a = x.detach().abs().amax().item()
            if a > stats[name]:
                stats[name] = a

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for i, feats in enumerate(calib_features):
            generate_fn(feats)
            if i + 1 >= max_batches:
                break
    for h in hooks:
        h.remove()
    return dict(stats)


def collect_act_ch_absmax(model: nn.Module, calib_features, generate_fn, max_batches: int = 4) -> dict[str, torch.Tensor]:
    """Per-input-channel absmax of Linear activations, for SmoothQuant."""
    stats: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, out):
            x = inp[0]
            if not torch.is_tensor(x):
                return
            ch = x.detach().abs().reshape(-1, x.shape[-1]).amax(0).float().cpu()
            if name not in stats:
                stats[name] = ch
            else:
                stats[name] = torch.maximum(stats[name], ch)

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for i, feats in enumerate(calib_features):
            generate_fn(feats)
            if i + 1 >= max_batches:
                break
    for h in hooks:
        h.remove()
    return stats


INT4_AMODES = {"log_static", "log_dyn", "log_token"}


def replace_linears(
    root: nn.Module,
    act_absmax: dict[str, float],
    w_mode: str,
    a_mode: str,
    scope: str,
    prefix: str = "",
    backend: str = "fake",
    smoothquant_alpha: float = 0.0,
    act_ch_absmax: dict[str, torch.Tensor] | None = None,
    group_size: int = 0,
    act_group_size: int | None = None,
    smooth_scales: dict[str, torch.Tensor] | None = None,
) -> int:
    n = 0
    ch_stats = act_ch_absmax or {}
    pre_scales = smooth_scales or {}
    for child_name, child in list(root.named_children()):
        full = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            if match_scope(full, scope):
                amax = act_absmax.get(full, child.weight.detach().abs().amax().item())
                resolved_a = act_mode_for_layer(full, a_mode)
                resolved_w = weight_mode_for_layer(full, w_mode)
                gsz = group_size_for_layer(full, w_mode, group_size)
                use_int4 = (
                    backend == "int4"
                    and resolved_w == "log"
                    and resolved_a in INT4_AMODES
                    and gsz == 0
                )
                smooth = None
                if full in pre_scales and resolved_w == "uniform":
                    smooth = pre_scales[full]
                # SmoothQuant is for affine/uniform layers. Do not migrate scale on log FFN.
                elif smoothquant_alpha > 0 and full in ch_stats and resolved_w == "uniform":
                    smooth = smoothquant_scales(
                        child.weight.data, ch_stats[full], smoothquant_alpha
                    )
                if use_int4:
                    from log_int4 import LogInt4Linear

                    setattr(root, child_name, LogInt4Linear(child, full, resolved_a, amax))
                else:
                    setattr(
                        root,
                        child_name,
                        QuantLinear(
                            child,
                            full,
                            w_mode,
                            a_mode,
                            amax,
                            smooth_scale=smooth,
                            group_size=group_size,
                            act_group_size=act_group_size,
                        ),
                    )
                n += 1
        else:
            n += replace_linears(
                child,
                act_absmax,
                w_mode,
                a_mode,
                scope,
                full,
                backend=backend,
                smoothquant_alpha=smoothquant_alpha,
                act_ch_absmax=act_ch_absmax,
                group_size=group_size,
                act_group_size=act_group_size,
                smooth_scales=smooth_scales,
            )
    return n
