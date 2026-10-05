#!/usr/bin/env python3
"""Fast per-layer/per-module LoRA mass profile.

Frobenius norm of dW = B@A without forming dW:
    ||B A||_F^2 = tr((B^T B)(A A^T))  -> r x r matrices, cheap even at r=512.

usage:
  lora_block_profile.py parts:<name>=<dir> [parts:...] [peft:<name>=<file>:<alpha>:<r>]
Prints total mass, layer windows (0-15 / 16-39 / 40-63), top layers, group shares
(FFN / attn / GDN).
"""
import glob
import os
import re
import sys

import torch
from safetensors import safe_open


def group_of(mod):
    if mod.startswith("mlp.") or mod.startswith("ffn_"):
        return "FFN"
    if "self_attn" in mod or mod.startswith("attn_"):
        return "attn"
    return "GDN"


def mass(a, b, scale=1.0):
    # a: r x in, b: out x r
    gb = b.t() @ b
    ga = a @ a.t()
    return float((gb * ga).sum().clamp(min=0).sqrt()) * scale


def profile_parts(path):
    per_layer, per_group = {}, {}
    for f in glob.glob(os.path.join(path, "blk.*.safetensors")):
        m = re.match(r"blk\.(\d+)\.(.+)\.weight\.safetensors$", os.path.basename(f))
        if not m:
            continue
        layer, mod = int(m.group(1)), m.group(2)
        with safe_open(f, framework="pt") as h:
            keys = set(h.keys())
            if not {"lora_a", "lora_b"} <= keys:
                continue
            val = mass(h.get_tensor("lora_a").float(), h.get_tensor("lora_b").float())
        per_layer[layer] = per_layer.get(layer, 0.0) + val
        g = group_of(mod)
        per_group[g] = per_group.get(g, 0.0) + val
    return per_layer, per_group


def profile_peft(path, alpha, r):
    per_layer, per_group = {}, {}
    pairs = {}
    with safe_open(path, framework="pt") as h:
        for key in h.keys():
            m = re.search(r"layers\.(\d+)\.(.+)\.lora_([AB])\.weight", key)
            if m:
                pairs.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = h.get_tensor(key).float()
    scale = alpha / r
    for (layer, mod), ab in pairs.items():
        if "A" in ab and "B" in ab:
            val = mass(ab["A"], ab["B"], scale)
            per_layer[layer] = per_layer.get(layer, 0.0) + val
            g = group_of(mod)
            per_group[g] = per_group.get(g, 0.0) + val
    return per_layer, per_group


def report(name, per_layer, per_group):
    if not per_layer:
        print(f"{name}: no tensors")
        return
    total = sum(per_layer.values())

    def win(lo, hi):
        return sum(per_layer.get(i, 0.0) for i in range(lo, hi + 1)) / total

    top = [l for l, _ in sorted(per_layer.items(), key=lambda kv: -kv[1])[:6]]
    groups = " ".join(f"{g}={per_group.get(g, 0.0) / total:.2f}" for g in ("FFN", "attn", "GDN"))
    print(
        f"{name:14s} total={total:8.1f} | win 0-15={win(0,15):.2f} 16-39={win(16,39):.2f} "
        f"40-63={win(40,63):.2f} | top_layers={top} | {groups}"
    )


def main():
    for arg in sys.argv[1:]:
        kind, _, rest = arg.partition(":")
        if kind == "parts":
            name, _, path = rest.partition("=")
            report(name, *profile_parts(path))
        elif kind == "peft":
            name, _, tail = rest.partition("=")
            path, alpha, r = tail.rsplit(":", 2)
            report(name, *profile_peft(path, float(alpha), float(r)))
        else:
            print("bad arg", arg, file=sys.stderr)


if __name__ == "__main__":
    main()
