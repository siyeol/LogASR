"""Log-codebook W4A4 using INT8 tensor-core GEMM.

The codebook is not affine, so INT4 MMA on the indices {-7..7} would be
wrong. Reconstruction z = sign(y)*(2^|y|-1) lives in int8
({-127,-63,...,127}), so the kernel is:

  1. pack/store 4-bit indices (weights)
  2. LUT to int8 z
  3. torch._int_mm (cuBLAS INT8 tensor cores on Ampere)
  4. fused scale by alpha_a * alpha_w, write fp16

Activation log-quant is a Triton elementwise kernel (~0.16 ms at encoder
fc1 size), not Python log2/pow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

QMAX = 7


def log_lut_int8(device=None) -> torch.Tensor:
    vals = []
    for y in range(-QMAX, QMAX + 1):
        if y == 0:
            vals.append(0)
        else:
            vals.append((1 if y > 0 else -1) * ((1 << abs(y)) - 1))
    t = torch.tensor(vals, dtype=torch.int8)
    return t.to(device) if device is not None else t


def log_indices(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    a = alpha.float().clamp_min(1e-8)
    f = torch.sign(x32) * torch.log2(1.0 + x32.abs() / a)
    return torch.clamp(torch.round(f), -QMAX, QMAX).to(torch.int8)


def pack_nibbles(idx0_14: torch.Tensor) -> torch.Tensor:
    u = idx0_14.to(torch.uint8)
    return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor, k: int) -> torch.Tensor:
    lo = packed & 0xF
    hi = packed >> 4
    u = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return u[..., :k]


@triton.jit
def _log_quant_i8_kernel(
    x_ptr,
    alpha_ptr,
    lut_ptr,
    z_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_am,
    stride_zm,
    stride_zk,
    SCALAR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if SCALAR:
        a = tl.load(alpha_ptr)
        a = tl.maximum(a, 1e-8)
    else:
        a = tl.load(alpha_ptr + offs_m * stride_am, mask=offs_m < M, other=1.0)
        a = tl.maximum(a, 1e-8)[:, None]
    ax = tl.abs(x)
    f = tl.math.log2(1.0 + ax / a)
    sgn = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    yq = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(sgn * f), -7.0), 7.0)
    idx = (yq + 7.0).to(tl.int32)
    z = tl.load(lut_ptr + idx, mask=mask, other=0)
    tl.store(
        z_ptr + offs_m[:, None] * stride_zm + offs_k[None, :] * stride_zk,
        z,
        mask=mask,
    )


@triton.jit
def _scale_bias_kernel(
    acc_ptr,
    alpha_a_ptr,
    alpha_w_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    stride_accm,
    stride_accn,
    stride_am,
    stride_om,
    stride_on,
    SCALAR_A: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.load(
        acc_ptr + offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn,
        mask=mask,
        other=0,
    ).to(tl.float32)
    alpha_w = tl.load(alpha_w_ptr)
    if SCALAR_A:
        alpha_a = tl.load(alpha_a_ptr)
        out = acc * (alpha_a * alpha_w)
    else:
        alpha_a = tl.load(alpha_a_ptr + offs_m * stride_am, mask=offs_m < M, other=0.0)
        out = acc * (alpha_a[:, None] * alpha_w)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = out + bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.float16),
        mask=mask,
    )


def quant_act_int8(x: torch.Tensor, alpha: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2.shape
    z = torch.empty(m, k, device=x.device, dtype=torch.int8)
    scalar = alpha.ndim == 0
    if scalar:
        alpha = alpha.to(device=x.device, dtype=torch.float32).reshape(())
        stride_am = 0
    else:
        alpha = alpha.reshape(-1).contiguous().to(dtype=torch.float32)
        stride_am = alpha.stride(0)
    lut = lut.to(device=x.device, dtype=torch.int8)
    bm, bk = 32, 128
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _log_quant_i8_kernel[grid](
        x2,
        alpha,
        lut,
        z,
        m,
        k,
        x2.stride(0),
        x2.stride(1),
        stride_am,
        z.stride(0),
        z.stride(1),
        SCALAR=scalar,
        BLOCK_M=bm,
        BLOCK_K=bk,
    )
    return z


def alpha_from_x(x: torch.Tensor, a_mode: str, static_alpha: float) -> torch.Tensor:
    if a_mode == "log_static":
        return torch.tensor(max(static_alpha, 1e-8), device=x.device, dtype=torch.float32)
    if a_mode == "log_dyn":
        return (x.abs().amax().float().clamp_min(1e-8) / (2**QMAX - 1)).reshape(())
    if a_mode == "log_token":
        return (x.abs().amax(dim=-1).float().clamp_min(1e-8) / (2**QMAX - 1)).contiguous()
    if a_mode == "log_ch":
        dims = tuple(range(x.ndim - 1))
        return x.abs().amax(dim=dims).float().clamp_min(1e-8) / (2**QMAX - 1)
    raise ValueError(a_mode)


def _int_mm_fp16(
    z_a: torch.Tensor,
    w_kn: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_w: torch.Tensor,
    bias: torch.Tensor | None,
    m_out: int,
) -> torch.Tensor:
    """z_a [M,K] int8, w_kn [K,N] int8 -> [m_out, N] fp16. Pads M for _int_mm."""
    m, k = z_a.shape
    n = w_kn.shape[1]
    pad = 0
    if m <= 16:
        pad = 32 - m
        z_a = F.pad(z_a, (0, 0, 0, pad))
        m_pad = z_a.shape[0]
    else:
        m_pad = m
    acc = torch._int_mm(z_a, w_kn)
    out = torch.empty(m_pad, n, device=z_a.device, dtype=torch.float16)
    scalar = alpha_a.ndim == 0
    if scalar:
        alpha_a = alpha_a.to(dtype=torch.float32, device=z_a.device).reshape(())
        stride_am = 0
    else:
        if pad:
            alpha_a = F.pad(alpha_a.reshape(-1), (0, pad))
        alpha_a = alpha_a.reshape(-1).contiguous().to(dtype=torch.float32)
        stride_am = alpha_a.stride(0)
    alpha_w = alpha_w.to(dtype=torch.float32, device=z_a.device).reshape(())
    has_bias = bias is not None
    if not has_bias:
        bias = acc  # dummy
    bm, bn = 64, 128
    grid = (triton.cdiv(m_pad, bm), triton.cdiv(n, bn))
    _scale_bias_kernel[grid](
        acc,
        alpha_a,
        alpha_w,
        bias,
        out,
        m_pad,
        n,
        acc.stride(0),
        acc.stride(1),
        stride_am,
        out.stride(0),
        out.stride(1),
        SCALAR_A=scalar,
        HAS_BIAS=has_bias,
        BLOCK_M=bm,
        BLOCK_N=bn,
    )
    return out[:m_out]


class LogPackedFp16Linear(nn.Module):
    """Log-codebook W4 as INT8 reconstruction; compute is FP16 GEMM.

    Persistent encoder-FFN storage is 8-bit (LUT of the log codebook), not a
    second packed nibble copy. Decoder/attn stay true 4-bit via Linear4bit.
    Dequant is a cast+scale; GEMM is the same cuBLAS FP16 path as baseline.
    """

    def __init__(self, linear: nn.Linear, name: str):
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach()
        absmax = w.abs().amax().clamp_min(1e-8)
        alpha = absmax / (2**QMAX - 1)
        y = log_indices(w, alpha)
        idx = (y.to(torch.int16) + QMAX).to(torch.uint8)
        lut = log_lut_int8(device=w.device)
        z_w = lut[idx.long()]
        self.register_buffer("weight_int8", z_w.contiguous())
        self.register_buffer("weight_alpha", alpha.detach().to(dtype=w.dtype).reshape(()))
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().contiguous())

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, stored=int8"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(dtype=x.dtype) * self.weight_alpha.to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype, device=x.device)
        return F.linear(x, w, b)


class LogInt4Linear(nn.Module):
    """Weight-packed log-int4 + INT8 GEMM Linear."""

    def __init__(self, linear: nn.Linear, name: str, a_mode: str, act_absmax: float):
        super().__init__()
        self.name = name
        self.a_mode = a_mode
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach()
        absmax = w.abs().amax().clamp_min(1e-8)
        alpha = absmax / (2**QMAX - 1)
        y = log_indices(w, alpha)
        idx = (y.to(torch.int16) + QMAX).to(torch.uint8)
        if w.shape[1] % 2 != 0:
            raise ValueError(f"{name}: in_features must be even to pack int4")
        lut = log_lut_int8(device=w.device)
        z_w = lut[idx.long()]  # [N, K] int8 reconstruction
        self.register_buffer("weight_packed", pack_nibbles(idx))
        self.register_buffer("weight_int8_kn", z_w.T.contiguous())  # [K, N]
        self.register_buffer("weight_alpha", alpha.detach().float().reshape(()))
        self.register_buffer("lut", lut)
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16).contiguous())
        self.static_alpha = max(float(act_absmax), 1e-8) / (2**QMAX - 1)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, a_mode={self.a_mode}, "
            f"packed={tuple(self.weight_packed.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        lut = self.lut.to(device=x.device, dtype=torch.int8)
        w_kn = self.weight_int8_kn
        alpha_w = self.weight_alpha.to(device=x.device, dtype=torch.float32)
        if self.a_mode in ("none", "fp16"):
            raise RuntimeError("LogInt4Linear needs log activations")
        alpha_a = alpha_from_x(x2, self.a_mode, self.static_alpha)
        z_a = quant_act_int8(x2, alpha_a, lut)
        y = _int_mm_fp16(z_a, w_kn, alpha_a, alpha_w, self.bias, x2.shape[0])
        return y.view(*orig_shape[:-1], self.out_features)
