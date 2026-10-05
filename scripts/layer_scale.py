#!/usr/bin/env python3
"""Scale a LoRA parts dir per layer (LiNeS ramp / per-layer profile).

usage: layer_scale.py <src_parts> <dst_parts> <spec>
spec:  linear:<lo>:<hi>:<nlayers>   factor = lo + (hi-lo)*layer/(nlayers-1)
       window:<a>:<b>:<scale>       factor = scale inside [a,b], else 1.0
       <float>                      constant factor
"""
import glob
import os
import re
import sys

from safetensors.torch import load_file, save_file


def factor(spec, layer):
    p = spec.split(":")
    if p[0] == "linear":
        lo, hi, n = float(p[1]), float(p[2]), int(p[3])
        return lo + (hi - lo) * (layer / max(1, n - 1))
    if p[0] == "window":
        a, b, s = int(p[1]), int(p[2]), float(p[3])
        return s if a <= layer <= b else 1.0
    return float(spec)


def main():
    src, dst, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "b"
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "blk.*.safetensors"))):
        m = re.match(r"blk\.(\d+)\.", os.path.basename(f))
        if not m:
            continue
        fac = factor(spec, int(m.group(1)))
        tensors = load_file(f)
        if {"lora_a", "lora_b"} <= set(tensors):
            # ΔW = lora_b @ lora_a; scale exactly one factor (or both by sqrt).
            if mode == "a":
                out = {k: (v * fac if k == "lora_a" else v) for k, v in tensors.items()}
            elif mode == "both":
                s = fac ** 0.5
                out = {k: v * s for k, v in tensors.items()}
            else:
                out = {k: (v * fac if k == "lora_b" else v) for k, v in tensors.items()}
        else:
            out = {k: v * fac for k, v in tensors.items()}
        save_file(out, os.path.join(dst, os.path.basename(f)))
        n += 1
    print(f"scaled {n} tensors {src} -> {dst} ({spec}, mode={mode})")


if __name__ == "__main__":
    main()
