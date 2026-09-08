"""Edge-ASR AWQ scale search without llmcompressor.

Edge-ASR (arXiv:2507.07877) uses
  s_j = max(|X_j|)^α / max(|W_j|)^{1-α}
and grids α ∈ [0, 1]. Weights are then grouped uniform INT4 (Whisper g=64).

llmcompressor's sequential AWQ pipeline hits NaN / incomplete mappings on
Whisper / Qwen3-ASR. This module searches per Linear using cached inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from quant import match_scope, uniform_quant_dequant_grouped

N_GRID = 20
MAX_TOKENS = 2048
TOKENS_PER_FWD = 64


def collect_linear_inputs(
    model: nn.Module,
    generate_fn,
    batches,
    scope: str,
) -> dict[str, dict]:
    """Per-scoped-Linear: channel absmax + a token reservoir of inputs."""
    caches: dict[str, dict] = {}
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, out):
            x = inp[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            flat = x.detach().reshape(-1, x.shape[-1])
            step = max(1, flat.shape[0] // TOKENS_PER_FWD)
            sample = flat[::step][:TOKENS_PER_FWD].float().cpu()
            ch = flat.abs().amax(0).float().cpu()
            if name not in caches:
                caches[name] = {"x": sample, "xmax": ch}
                return
            slot = caches[name]
            slot["xmax"] = torch.maximum(slot["xmax"], ch)
            if slot["x"].shape[0] < MAX_TOKENS:
                slot["x"] = torch.cat([slot["x"], sample], dim=0)[:MAX_TOKENS]

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and match_scope(name, scope):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for feats in batches:
            generate_fn(feats)
    for h in hooks:
        h.remove()
    return caches


def _normalize_scales(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp_min(1e-4)
    return s / (s.max() * s.min()).sqrt().clamp_min(1e-8)


def search_awq_scales(
    model: nn.Module,
    caches: dict[str, dict],
    scope: str,
    group_size: int = 64,
    n_grid: int = N_GRID,
    qmax: int = 7,
) -> dict[str, torch.Tensor]:
    """Return CPU float scales keyed by Linear module name."""
    scales: dict[str, torch.Tensor] = {}
    device = next(model.parameters()).device
    n_ok = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not match_scope(name, scope):
            continue
        slot = caches.get(name)
        if slot is None or slot["x"].numel() == 0:
            continue
        w = mod.weight.detach()
        x = slot["x"].to(device=device, dtype=torch.float32)
        xmax = slot["xmax"].to(device=device, dtype=torch.float32)
        wmax = w.abs().amax(dim=0).float().clamp_min(1e-8)
        if xmax.numel() != wmax.numel():
            print(f"[awq] skip {name}: xmax {tuple(xmax.shape)} vs in {tuple(wmax.shape)}")
            continue
        w_f = w.float()
        orig = x @ w_f.t()
        best_err = None
        best_s = torch.ones_like(wmax)
        for i in range(n_grid):
            alpha = i / max(n_grid - 1, 1)
            s = _normalize_scales(xmax.pow(alpha) / wmax.pow(1.0 - alpha))
            wq = uniform_quant_dequant_grouped(w_f * s.unsqueeze(0), group_size, qmax=qmax)
            pred = (x / s) @ wq.float().t()
            err = (pred - orig).square().mean()
            val = float(err.item())
            if not torch.isfinite(err):
                continue
            if best_err is None or val < best_err:
                best_err = val
                best_s = s.detach().cpu()
        scales[name] = best_s.float()
        n_ok += 1
        print(f"[awq] {name} tokens={x.shape[0]} err={best_err} in={w.shape[1]} out={w.shape[0]}")
        del orig, x, w_f
    print(f"[awq] searched {n_ok} / {len(caches)} layers g_w={group_size} n_grid={n_grid}")
    return scales


def save_awq_ckpt(save_dir: str | Path, scales: dict[str, torch.Tensor], meta: dict) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in scales.items()}, save_dir / "awq_scales.pt")
    (save_dir / "awq_meta.json").write_text(json.dumps(meta, indent=2))
    # Marker so the compare runner's ckpt_ok() (config.json) passes.
    cfg = {
        "model_type": "speechptq_awq_scales",
        "n_layers": len(scales),
        **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, list))},
    }
    (save_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved {save_dir} n_scales={len(scales)}")
