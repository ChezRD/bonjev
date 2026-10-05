#!/usr/bin/env python3
"""Concatenate LoRA part blocks instead of summing them.

Keeps every input block intact (no SVD re-mixing); the output rank is the sum
of the input ranks. Missing tensors in one input are simply skipped.

usage: spectral_concat.py <out_dir> <parts_dir>[:scale] ...
"""
import glob
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_parts(spec):
    if not os.path.isdir(spec):
        raise SystemExit(f"not a parts dir: {spec}")
    out = {}
    for p in glob.glob(f"{spec}/*.safetensors"):
        name = os.path.basename(p)[:-len(".safetensors")]
        with safe_open(p, "pt") as h:
            out[name] = (h.get_tensor("lora_a").float(), h.get_tensor("lora_b").float())
    return out


def parse_spec(raw):
    if ":" in raw:
        path, _, tail = raw.rpartition(":")
        try:
            return path, float(tail)
        except ValueError:
            pass
    return raw, 1.0


def main():
    out_dir = sys.argv[1]
    specs = [parse_spec(raw) for raw in sys.argv[2:]]
    os.makedirs(out_dir, exist_ok=True)
    loaded = [(load_parts(p), s) for p, s in specs]
    names = sorted({n for d, _ in loaded for n in d})
    max_rank = 0
    for name in names:
        blocks_a, blocks_b = [], []
        for d, scale in loaded:
            if name in d:
                a, b = d[name]
                blocks_a.append(a)
                blocks_b.append(b * scale)
        if not blocks_a:
            continue
        a = torch.vstack(blocks_a).half().contiguous()
        b = torch.hstack(blocks_b).half().contiguous()
        max_rank = max(max_rank, a.shape[0])
        save_file({"lora_a": a, "lora_b": b}, os.path.join(out_dir, f"{name}.safetensors"))
    print(f"concat {len(names)} tensors -> {out_dir} (max rank {max_rank})")


if __name__ == "__main__":
    main()
