#!/usr/bin/env python3
"""Interference-aware merge of LoRA parts (repo tool, no external helpers).

Pipeline flags (composable):
  --pico G      Pico-style calibration: downscale over-shared output-space
                directions of each B_i before merging (G ~ 0.5).
  --align       KnOTS-style shared left basis: SVD of concatenated B matrices,
                each adapter projected into that basis, merging in low-rank
                coordinates C_i = (U^T B_i) A_i.
  --budget A    Net-Utility direction budget: per tensor, score every singular
                direction of every adapter as utility - A * interference and
                keep the best <rank> directions (requires --align).
  --do R        DO-Merging-style: decouple magnitude (normalize each delta to
                the mean norm), then project each delta onto the orthogonal
                complement of the others' top-R subspaces (no --align).
  --rescale     rescale the merged tensor to the plain-sum Frobenius norm.

usage: spectral_merge3.py <out_dir> <rank> [flags] <parts_dir>[:scale] ...
       (parts_dir holds <name>.safetensors with lora_a/lora_b)

No flags == plain sum of scaled deltas + randomized-SVD truncation to <rank>.
Sources live under a local work dir (e.g. <work>/preserve/parts,
<work>/pool/parts, <work>/merges/parts); see docs/RECIPES.md.
"""
import glob
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


def load_parts(spec):
    if not os.path.isdir(spec):
        raise SystemExit(f"not a parts dir: {spec} (convert the adapter to parts first)")
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


def rsvd(a, r, over=32, iters=2):
    m, n = a.shape
    k = min(r + over, min(m, n))
    g = torch.randn(n, k, dtype=a.dtype)
    y = a @ g
    for _ in range(iters):
        q, _ = torch.linalg.qr(y)
        z, _ = torch.linalg.qr(a.t() @ q)
        y = a @ z
    q, _ = torch.linalg.qr(y)
    b = q.t() @ a
    u, s, vh = torch.linalg.svd(b, full_matrices=False)
    return (q @ u)[:, :r], s[:r], vh[:r]


def pico_calibrate(pairs, gamma):
    """pairs: list of (A, B, scale). Returns calibrated B matrices."""
    bs = [b for _, b, _ in pairs]
    norms2 = [float((b * b).sum()) + 1e-12 for b in bs]
    svds = []
    for i, b in enumerate(bs):
        u, s, vh = torch.linalg.svd(b, full_matrices=False)
        o = torch.zeros(u.shape[1])
        for j, bj in enumerate(bs):
            if i == j:
                continue
            proj = bj.t() @ u
            o += (proj * proj).sum(dim=0) / norms2[j]
        svds.append((u, s, vh, o))
    omax = max(float(x[3].max()) for x in svds) + 1e-12
    out = []
    for u, s, vh, o in svds:
        f = 1.0 / (1.0 + gamma * (o / omax))
        out.append(u @ torch.diag(s * f) @ vh)
    return out


def do_decompose(deltas, r):
    """Magnitude-decoupled, orthogonally-projected deltas."""
    mean_norm = sum(float(d.norm()) for d in deltas) / len(deltas)
    normed = [d / (float(d.norm()) + 1e-12) * mean_norm for d in deltas]
    out = []
    for i, d in enumerate(normed):
        for j, o in enumerate(normed):
            if i == j:
                continue
            _, _, vh = rsvd(o, min(r, min(o.shape)))
            d = d - (d @ vh.t()) @ vh
        out.append(d)
    return sum(out)


def budget_merge(cs, rank, alpha):
    """Net-Utility selection over singular directions of aligned matrices C_i."""
    p_all, q_all, sig, lab, smax = [], [], [], [], []
    for i, c in enumerate(cs):
        u, s, vh = torch.linalg.svd(c, full_matrices=False)
        p_all.append(u)
        q_all.append(vh.t())
        sig.append(s)
        lab += [i] * s.numel()
        smax += [float(s[0]) + 1e-12] * s.numel()
    p_all = torch.cat(p_all, dim=1)          # (k, total)
    q_all = torch.cat(q_all, dim=1)          # (in, total)
    sig = torch.cat(sig)
    lab = torch.tensor(lab)
    smax = torch.tensor(smax)
    gp = p_all.t() @ p_all
    gq = q_all.t() @ q_all
    cross = (lab[:, None] != lab[None, :]).float()
    w = sig / smax
    interf = ((gp * gp + gq * gq) * cross) @ w
    score = sig / smax - alpha * interf
    k = min(rank, score.numel())
    idx = torch.topk(score, k).indices
    return (p_all[:, idx] * sig[idx]) @ q_all[:, idx].t()


def main():
    args = sys.argv[1:]
    out_dir, rank = args.pop(0), int(args.pop(0))
    opt, raw_specs = {}, []
    value_flags = {"--pico", "--do", "--budget"}
    while args:
        tok = args.pop(0)
        if tok.startswith("--"):
            opt[tok] = float(args.pop(0)) if tok in value_flags else None
        else:
            raw_specs.append(tok)
    specs = [parse_spec(raw) for raw in raw_specs]
    pico_g = opt.get("--pico")
    do_r = opt.get("--do")
    alpha = opt.get("--budget")
    aligned = "--align" in opt
    rescale = "--rescale" in opt
    if alpha is not None and not aligned:
        raise SystemExit("--budget requires --align")
    if do_r is not None and aligned:
        raise SystemExit("--do is incompatible with --align (choose one path)")

    os.makedirs(out_dir, exist_ok=True)
    loaded = [(load_parts(p), s) for p, s in specs]
    names = sorted({n for d, _ in loaded for n in d})
    total = 0.0
    for name in names:
        pairs = []
        for d, scale in loaded:
            if name in d:
                a, b = d[name]
                pairs.append((a, b, scale))
        if pico_g is not None and len(pairs) > 1:
            cal = pico_calibrate(pairs, pico_g)
            pairs = [(a, cal[i], s) for i, (a, _, s) in enumerate(pairs)]

        plain = None
        if do_r is not None:
            deltas = [(b @ a) * s for a, b, s in pairs]
            merged = do_decompose(deltas, int(do_r))
            plain = sum(deltas)
        elif aligned:
            bcat = torch.cat([b * s for _, b, s in pairs], dim=1)  # (out, sum r)
            u, _, _ = torch.linalg.svd(bcat, full_matrices=False)
            cs = [(u.t() @ b) @ a * s for a, b, s in pairs]
            merged = budget_merge(cs, rank, alpha) if alpha is not None else sum(cs)
            merged = u @ merged  # lift from the shared basis back to output space
            if rescale:
                plain = sum((b @ a) * s for a, b, s in pairs)
        else:
            merged = plain = sum((b @ a) * s for a, b, s in pairs)

        if rescale and plain is not None and plain.shape == merged.shape:
            pn, mn = float(plain.norm()) + 1e-12, float(merged.norm()) + 1e-12
            merged = merged * (pn / mn)

        r = min(rank, min(merged.shape))
        u, s, vh = rsvd(merged, r)
        save_file({"lora_a": vh.half().contiguous(), "lora_b": (u * s[None, :]).half().contiguous()},
                  f"{out_dir}/{name}.safetensors")
        total += float((s ** 2).sum())
    tag = ",".join(f"{k}={v}" for k, v in opt.items()) or "sum"
    print(f"merged[{tag}] {len(names)} tensors -> {out_dir} (rank {rank}, ||dW||_F {total ** 0.5:.2f})")


if __name__ == "__main__":
    main()
