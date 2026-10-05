#!/usr/bin/env python3
"""Inventory of extracted LoRA parts and GGUF artifacts under the jev root.

Prints three markdown-ish tables: parts dirs (safetensors), PEFT adapters,
and GGUF files (adapters vs base models). Paths are relative to the jev root.
"""
import glob
import json
import os

from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(REPO)
WORK = ["work"]
SKIP_PARTS = ("vendor", ".git", "attic")


def parts_info(d):
    files = sorted(glob.glob(os.path.join(d, "blk.*.safetensors")))
    if not files:
        return None
    layers, mods, size = set(), set(), 0
    for p in files:
        nm = os.path.basename(p)[:-len(".safetensors")]
        seg = nm.split(".")
        if len(seg) >= 3:
            layers.add(int(seg[1]))
            mods.add(".".join(seg[2:]))
        size += os.path.getsize(p)
    with safe_open(files[0], "pt") as h:
        rank = h.get_tensor("lora_a").shape[0]
    return len(files), min(layers), max(layers), len(mods), rank, size / 1e6


def main():
    parts, peft, gguf = [], [], []
    for w in WORK:
        base = os.path.join(REPO, w)
        if not os.path.isdir(base):
            continue
        for d, _, files in os.walk(base):
            if any(s in d for s in SKIP_PARTS):
                continue
            info = parts_info(d)
            if info:
                parts.append((os.path.relpath(d, ROOT), *info))
            if "adapter_model.safetensors" in files or "adapter.safetensors" in files:
                cfg = None
                for c in ("adapter_config.json", "manifest.json"):
                    p = os.path.join(d, c)
                    if os.path.exists(p):
                        try:
                            j = json.load(open(p))
                            cfg = j.get("base_model_name_or_path") or (j.get("base_model") or {}).get("model_id")
                        except Exception:
                            cfg = None
                        break
                peft.append((os.path.relpath(d, ROOT), cfg or "?"))
            for f in files:
                if f.endswith(".gguf"):
                    p = os.path.join(d, f)
                    kind = "base" if f.startswith("Qwen") or "BF16" in f or "PTQ" in f or "Q2_K" in f else "adapter"
                    gguf.append((os.path.relpath(p, ROOT), os.path.getsize(p) / 1e6, kind))

    print("## Parts (safetensors)\n")
    print("| parts dir | tensors | layers | modules | rank | MB |")
    print("|---|---:|---|---:|---:|---:|")
    for p, n, l0, l1, m, r, mb in sorted(parts):
        print(f"| `{p}` | {n} | {l0}–{l1} | {m} | {r} | {mb:.0f} |")

    print("\n## PEFT adapters\n")
    print("| directory | base |")
    print("|---|---|")
    for p, b in sorted(peft):
        print(f"| `{p}` | {b} |")

    print("\n## GGUF\n")
    print("| file | MB | type |")
    print("|---|---:|---|")
    for p, mb, k in sorted(gguf):
        print(f"| `{p}` | {mb:.0f} | {k} |")


if __name__ == "__main__":
    main()
