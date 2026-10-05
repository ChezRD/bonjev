#!/usr/bin/env python3
"""Extract a LoRA delta between a HuggingFace base checkpoint and a fine-tuned one.

For every tensor present in both checkpoints:
  - diff in fp32; bit-identical tensors are skipped (untouched by the fine-tune),
  - 2D weights are factored with randomized SVD into lora_a (R, in) / lora_b (out, R),
  - other tensors (norm vectors, A_log, biases, conv1d) are metrics only.

usage: extract_delta.py <base_dir> <finetuned.safetensors|dir> <out_parts_dir> [--rank R] [--threads N]
"""
import argparse
import gc
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_index(path):
    if os.path.isdir(path):
        idx = sorted(glob.glob(os.path.join(path, "model.safetensors.index.json")))
        if idx:
            mapping = json.load(open(idx[0]))["weight_map"]
            return mapping, os.path.dirname(idx[0])
        single = sorted(glob.glob(os.path.join(path, "model.safetensors")))
        if single:
            return None, path
        raise SystemExit(f"no model.safetensors index or file in {path}")
    return None, os.path.dirname(path)


def rsvd(a, r, over=32, iters=2):
    m, n = a.shape
    k = min(r + over, min(m, n))
    g = torch.randn(n, k, dtype=a.dtype)
    y = a @ g
    for _ in range(iters):
        q, _ = torch.linalg.qr(y)
        z, _ = torch.linalg.qr(a.T @ q)
        y = a @ z
    q, _ = torch.linalg.qr(y)
    b = q.T @ a
    u, s, vh = torch.linalg.svd(b, full_matrices=False)
    return (q @ u)[:, :r], s[:r], vh[:r]


# HF module name -> GGUF tensor base (same mapping as the PEFT->parts pipeline).
GGUF_MAP = {
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
PREFIXES = ("model.language_model.", "model.", "language_model.", "backbone.")


def canon(key):
    k = key.removeprefix("base_model.model.")
    for pref in PREFIXES:
        if k.startswith(pref):
            k = k[len(pref):]
            break
    return k if k.startswith("layers.") else None


def gguf_name(key):
    k = key.removeprefix("base_model.model.")
    for pref in PREFIXES:
        if k.startswith(pref):
            k = k[len(pref):]
            break
    if not k.startswith("layers."):
        return None
    layer, _, mod = k[len("layers."):].partition(".")
    mod = mod.removesuffix(".weight")
    base = GGUF_MAP.get(mod)
    return f"blk.{layer}.{base}" if base else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("finetuned")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    base_map, base_dir = find_index(args.base)
    if base_map is None:
        raise SystemExit("base must be a checkpoint directory with an index")

    ft_map = None
    ft_dir = None
    ft_file = None
    if os.path.isfile(args.finetuned):
        ft_file = args.finetuned
    else:
        ft_map, ft_dir = find_index(args.finetuned)
        if ft_map is None:
            ft_file = os.path.join(ft_dir, "model.safetensors")

    metrics = open(os.path.join(args.out, os.pardir, "metrics.jsonl"), "w")
    base_handles = {}
    ft_handles = {}
    n_done = n_skip = n_1d = 0

    def base_tensor(key):
        shard = base_map[key]
        h = base_handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(base_dir, shard), "pt")
            base_handles[shard] = h
        return h.get_tensor(key)

    def ft_tensor(key):
        if ft_map is None:
            h = ft_handles.get(None)
            if h is None:
                h = safe_open(ft_file, "pt")
                ft_handles[None] = h
            return h.get_tensor(key)
        shard = ft_map[key]
        h = ft_handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(ft_dir, shard), "pt")
            ft_handles[shard] = h
        return h.get_tensor(key)

    ft_keys = list(ft_map) if ft_map is not None else None
    if ft_keys is None:
        with safe_open(ft_file, "pt") as ft:
            ft_keys = list(ft.keys())

    base_by_canon = {}
    for k in base_map:
        c = canon(k)
        if c is not None:
            base_by_canon.setdefault(c, k)
    ft_by_canon = {}
    for k in ft_keys:
        c = canon(k)
        if c is not None:
            ft_by_canon.setdefault(c, k)

    for c in sorted(set(base_by_canon) & set(ft_by_canon)):
        base_key = base_by_canon[c]
        ft_key = ft_by_canon[c]
        tuned = ft_tensor(ft_key)
        if tuned.dim() != 2:
            n_1d += 1
            del tuned
            continue
        name = gguf_name(ft_key)
        if name is None:
            n_skip += 1
            del tuned
            continue
        base = base_tensor(base_key).float()
        tuned = tuned.float()
        diff = tuned - base
        del base, tuned
        if bool(torch.all(diff.abs() < 1e-6)):
            n_skip += 1
            del diff
            continue
        r = min(args.rank, min(diff.shape))
        u, s, vh = rsvd(diff, r)
        save_file(
            {"lora_a": vh.half().contiguous(), "lora_b": (u * s[None, :]).half().contiguous()},
            os.path.join(args.out, f"{name}.safetensors"),
        )
        metrics.write(json.dumps(
            {"name": name, "hf_key": ft_key, "shape": list(diff.shape), "rank": r,
             "sigma_max": float(s[0]), "fro": float(diff.norm())}) + "\n")
        metrics.flush()
        n_done += 1
        del diff, u, s, vh
        gc.collect()
    metrics.close()
    print(f"extracted {n_done} tensors ({n_skip} identical/non-text, {n_1d} non-2D) "
          f"-> {args.out} (rank {args.rank})")


if __name__ == "__main__":
    main()
