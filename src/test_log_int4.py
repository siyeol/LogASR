"""Correctness + speed of log W4A4 INT8 GEMM vs fake-quant."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from log_int4 import LogInt4Linear
from quant import QuantLinear


def bench(fn, n_warm=15, n_iter=60):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def main():
    torch.manual_seed(0)
    device = "cuda"
    for M, K, N in [(32, 64, 96), (17, 1280, 1280), (24000, 1280, 5120)]:
        linear = nn.Linear(K, N, bias=True).to(device=device, dtype=torch.float16)
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        fake = QuantLinear(linear, "t", "log", "log_token", linear.weight.abs().amax().item()).to(device)
        kern = LogInt4Linear(linear, "t", "log_token", linear.weight.abs().amax().item()).to(device)
        yf = fake(x)
        yi = kern(x)
        diff = (yf.float() - yi.float()).abs()
        print(
            f"M={M} K={K} N={N} maxdiff={diff.max().item():.4e} mean={diff.mean().item():.4e}"
        )

    M, K, N = 24000, 1280, 5120
    linear = nn.Linear(K, N, bias=True).to(device=device, dtype=torch.float16)
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    fake = QuantLinear(linear, "t", "log", "log_token", linear.weight.abs().amax().item()).to(device)
    kern = LogInt4Linear(linear, "t", "log_token", linear.weight.abs().amax().item()).to(device)
    t_fp = bench(lambda: F.linear(x, linear.weight, linear.bias))
    t_fq = bench(lambda: fake(x))
    t_k = bench(lambda: kern(x))
    print(
        f"fc1 ms  fp16={t_fp*1e3:.3f}  fake={t_fq*1e3:.3f}  int4={t_k*1e3:.3f}  "
        f"vs_fp16={t_k/t_fp:.2f}x vs_fake={t_k/t_fq:.2f}x"
    )


if __name__ == "__main__":
    main()
