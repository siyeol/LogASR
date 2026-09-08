"""Count parameters and theoretical weight bits by module role."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMultimodalLM, WhisperForConditionalGeneration

from layer_groups import layer_group
from run_eval_qwen3 import compressed_tensors_load_kwargs

OUT = Path("/workspace/SpeechPTQ/results/diagnostics")


def count_linear(model: nn.Module) -> dict:
    groups = {}
    total = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        n = int(mod.weight.numel())
        if mod.bias is not None:
            n += int(mod.bias.numel())
        g = layer_group(name)
        groups[g] = groups.get(g, 0) + n
        total += n
    non_linear = sum(p.numel() for p in model.parameters()) - total
    groups["non_linear"] = int(non_linear)
    groups["linear_total"] = total
    groups["param_total"] = int(sum(p.numel() for p in model.parameters()))
    return groups


def mib(n_params: int, bits: float) -> float:
    return n_params * bits / 8 / (1024**2)


def recipes(counts: dict, kind: str) -> dict:
    """Theoretical packed weight size. Activations are not stored."""
    def bits_for(g: str, recipe: str) -> float:
        if g in ("lm_head", "non_linear"):
            return 16.0
        if recipe == "fp16":
            return 16.0
        if recipe == "gptq_all":
            return 4.0 if g != "lm_head" else 16.0
        if recipe == "paper":
            # Whisper: log W4 all Linear except lm_head
            # Qwen3: GPTQ W4 on LLM, log W4 on encoder, projector/lm_head 16
            if kind == "whisper":
                return 4.0
            if g.startswith("llm"):
                return 4.0
            if g.startswith("enc"):
                return 4.0
            return 16.0
        return 16.0

    out = {}
    for recipe in ("fp16", "paper"):
        total_mib = 0.0
        detail = {}
        for g, n in counts.items():
            if g in ("linear_total", "param_total"):
                continue
            b = 16.0 if g == "non_linear" else bits_for(g, recipe)
            mb = mib(n, b)
            detail[g] = {"params": n, "bits": b, "mib": round(mb, 1)}
            total_mib += mb
        out[recipe] = {"total_mib": round(total_mib, 1), "groups": detail}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading whisper...")
    w = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    w_counts = count_linear(w)
    w_rec = recipes(w_counts, "whisper")
    del w
    print("loading qwen3...")
    mid = "Qwen/Qwen3-ASR-1.7B-hf"
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kw.update(compressed_tensors_load_kwargs(mid))
    q = AutoModelForMultimodalLM.from_pretrained(mid, **kw)
    q_counts = count_linear(q)
    q_rec = recipes(q_counts, "qwen3")
    del q
    payload = {
        "whisper": {"counts": w_counts, "recipes": w_rec},
        "qwen3": {"counts": q_counts, "recipes": q_rec},
    }
    path = OUT / "param_bits.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
