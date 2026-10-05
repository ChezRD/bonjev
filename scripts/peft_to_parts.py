#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter of the Qwen3.5 text backbone to per-tensor parts
for gguf_lora_writer.py. B is pre-scaled by alpha/rank so the GGUF adapter can use
alpha = rank (llama.cpp scale = 1).

usage: peft_to_parts.py <adapter_model.safetensors> <out_parts_dir> [alpha_over_rank]
"""
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAP = {
    "linear_attn.in_proj_qkv": "attn_qkv.weight",
    "linear_attn.in_proj_z": "attn_gate.weight",
    "linear_attn.out_proj": "ssm_out.weight",
    "linear_attn.in_proj_a": "ssm_alpha.weight",
    "linear_attn.in_proj_b": "ssm_beta.weight",
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.gate_proj": "ffn_gate.weight",
    "mlp.up_proj": "ffn_up.weight",
    "mlp.down_proj": "ffn_down.weight",
}


def gguf_name(key):
    k = key.removeprefix("base_model.model.")
    for pref in ("model.language_model.", "model.", "language_model."):
        if k.startswith(pref):
            k = k[len(pref):]
            break
    if not k.startswith("layers."):
        return None
    rest = k[len("layers."):]
    layer, _, mod = rest.partition(".")
    mod = mod.rsplit(".lora_", 1)[0]
    base = MAP.get(mod)
    if base is None:
        return None
    return f"blk.{layer}.{base}"


def main():
    adapter, out_dir = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    os.makedirs(out_dir, exist_ok=True)
    pairs = {}
    with safe_open(adapter, "pt") as h:
        for key in h.keys():
            name = gguf_name(key)
            if name is None:
                continue
            side = "a" if ".lora_A." in key else "b"
            pairs.setdefault(name, {})[side] = h.get_tensor(key)
    written = 0
    for name, sides in sorted(pairs.items()):
        a = sides["a"].to(torch.float32)          # (r, in)
        b = sides["b"].to(torch.float32) * scale  # (out, r)
        save_file({"lora_a": a.half(), "lora_b": b.half()},
                  os.path.join(out_dir, f"{name}.safetensors"))
        written += 1
    print(f"{written} tensor pairs -> {out_dir} (scale {scale})")


if __name__ == "__main__":
    main()
