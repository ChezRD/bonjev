#!/usr/bin/env python3
"""Per-layer / per-module magnitude profile of LoRA adapters.

Inputs are either extracted parts dirs (lora_a/lora_b safetensors) or PEFT
adapter_model.safetensors files. For PEFT adapters with an available base model,
the relative delta norm ||BA||/||W|| is computed too.

usage: lora_profile.py <label> <parts_dir_or_peft_file> [base_safetensors_index.json]
"""
import collections
import glob
import json
import math
import os
import sys

import torch
from safetensors import safe_open

CATS = {
    "attn_qkv.weight": "gdn_qkv",
    "attn_gate.weight": "gdn_gate",
    "ssm_out.weight": "gdn_out",
    "ssm_alpha.weight": "gdn_a",
    "ssm_beta.weight": "gdn_b",
    "attn_q.weight": "attn_q",
    "attn_k.weight": "attn_k",
    "attn_v.weight": "attn_v",
    "attn_output.weight": "attn_o",
    "ffn_gate.weight": "ffn_gate",
    "ffn_up.weight": "ffn_up",
    "ffn_down.weight": "ffn_down",
}
PEFT_MAP = {
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


def iter_parts(parts_dir):
    for p in sorted(glob.glob(f"{parts_dir}/*.safetensors")):
        name = os.path.basename(p)[:-len(".safetensors")]
        with safe_open(p, "pt") as h:
            a = h.get_tensor("lora_a").to(torch.float32)
            b = h.get_tensor("lora_b").to(torch.float32)
        yield name, a, b


def iter_peft(path):
    with safe_open(path, "pt") as h:
        pairs = {}
        for key in h.keys():
            k = key.removeprefix("base_model.model.")
            for pref in ("model.language_model.", "model.", "language_model."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    break
            if not k.startswith("layers."):
                continue
            rest = k[len("layers."):]
            layer, _, mod = rest.partition(".")
            mod = mod.rsplit(".lora_", 1)[0]
            base = PEFT_MAP.get(mod)
            if base is None:
                continue
            side = "a" if ".lora_A." in key else "b"
            pairs.setdefault(f"blk.{layer}.{base}", {})[side] = h.get_tensor(key)
    for name, sides in sorted(pairs.items()):
        yield name, sides["a"].to(torch.float32), sides["b"].to(torch.float32)


def base_weight_norm(index_path, gguf_name):
    """||W||_F for the base tensor behind a gguf name, from the HF index."""
    layer, _, rest = gguf_name[len("blk."):].partition(".")
    weight = rest  # e.g. attn_qkv.weight
    # reverse map to HF key
    rev = {
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }.get(weight)
    if rev is None:
        return None
    key = f"model.language_model.layers.{layer}.{rev}"
    idx = json.load(open(index_path))
    shard = idx["weight_map"].get(key)
    if shard is None:
        return None
    root = os.path.dirname(index_path)
    with safe_open(os.path.join(root, shard), "pt") as h:
        w = h.get_tensor(key).to(torch.float32)
    return w.norm().item()


def main():
    label, src = sys.argv[1], sys.argv[2]
    index_path = sys.argv[3] if len(sys.argv) > 3 else None
    it = iter_parts(src) if os.path.isdir(src) else iter_peft(src)

    per_layer = collections.defaultdict(lambda: collections.defaultdict(float))
    per_cat = collections.defaultdict(float)
    rel_vals = []
    rank = 0
    for name, a, b in it:
        rank = max(rank, a.shape[0])
        layer = int(name.split(".")[1])
        cat = CATS.get(name.split(".", 2)[2], "other")
        delta = b @ a
        nrm = delta.norm().item()
        per_layer[layer][cat] += nrm * nrm
        per_cat[cat] += nrm * nrm
        if index_path:
            wn = base_weight_norm(index_path, name)
            if wn:
                rel_vals.append(nrm / wn)
    layers = sorted(per_layer)
    tot = {c: math.sqrt(v) for c, v in per_cat.items()}
    grand = math.sqrt(sum(v * v for v in tot.values()))
    print(f"== {label}  rank<= {rank}  layers {layers[0]}..{layers[-1]} ({len(layers)})")
    print("   category ||dW||_F:", {k: round(v, 3) for k, v in sorted(tot.items())})
    print(f"   total ||dW||_F ~ {grand:.2f}" + (f"   rel median {sorted(rel_vals)[len(rel_vals)//2]:.4f} max {max(rel_vals):.4f}" if rel_vals else ""))
    lyr = {l: math.sqrt(sum(per_layer[l].values())) for l in layers}
    top = sorted(lyr.items(), key=lambda kv: -kv[1])[:6]
    bot = sorted(lyr.items(), key=lambda kv: kv[1])[:6]
    print("   top layers:", ", ".join(f"{l}:{v:.2f}" for l, v in top))
    print("   low layers:", ", ".join(f"{l}:{v:.2f}" for l, v in bot))
    # per-layer share of the top category
    cats = sorted({c for l in layers for c in per_layer[l]})
    print("   layer profile (||dW|| per layer, " + "/".join(cats) + "):")
    for l in layers:
        row = " ".join(f"{c}:{math.sqrt(per_layer[l].get(c, 0.0)):.2f}" for c in cats)
        print(f"     {l:>2}: {row}")


if __name__ == "__main__":
    main()
