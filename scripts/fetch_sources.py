#!/usr/bin/env python3
"""Download the source models (trained + base) into the HF cache and show how to
slice them into delta parts. Does NOT run the (slow) slicing — use --check to
verify that the expected files and target parts paths line up.

Sources and method are documented in docs/RECIPES.md (provenance table).

Usage:
  scripts/fetch_sources.py [--check] [name ...]
    --check   only verify paths (no download)
    name...   restrict to some sources (default: all)
"""
import argparse, glob, os, sys

HUB = os.path.expanduser("~/.cache/huggingface/hub")

# name -> repo, revision, kind, base, out (repo-relative parts dir)
# kind: peft (LoRA adapter) | delta (full fine-tune) | clef | suffix
SOURCES = {
    "kev4b":     ("jaredpalmer/kev-4b", "qwen3", "peft", "Qwen/Qwen3-4B-Base",
                  "work/experiments/pool/parts/kev4b_q3_parts"),
    "candigate": ("CullenYap/CandiGate-Qwen3-4B", "main", "peft", "Qwen/Qwen3-4B",
                  "work/experiments/pool/parts/candigate_parts"),
    "senna":     ("sennaLLMLearner/qwen3-4b-system-one-lora", "main", "peft", "Qwen/Qwen3-4B-Instruct-2507",
                  "work/experiments/pool/parts/senna_parts"),
    "kev8b":     ("jaredpalmer/kev-8b", "main", "peft", "Qwen/Qwen3-8B-Base",
                  "work/experiments/merges8b/parts/kev8b"),
    "lct8b":     ("CaoHaoWei/Jev-LCT-Qwen3-8B", "main", "delta", "Qwen/Qwen3-8B",
                  "work/experiments/merges8b/parts/lct"),
    "tiny17b":   ("lostargon/Tiny-Jev-1.7B", "main", "delta", "Qwen/Qwen3-1.7B",
                  "work/experiments/merges17b/parts/m17_x_tiny"),
    "clef27b":   ("Cloudflare/clef", "main", "clef", "Qwen/Qwen3.8-27B",
                  "work/deltas/clef-27b/parts_text"),
    "autotrust": ("autotrust/JEV-27B", "962701f", "peft", "Qwen/Qwen3.8-27B",
                  "work/experiments/merges/parts/_at_parts"),
    "vega27b":   ("vllm-sr/Decision-2.0-Vega-27B", "main", "peft", "Qwen/Qwen3.8-27B",
                  "work/experiments/merges/parts/vega27_parts"),
    # plumb: the org/repo of the plumb/2 suffix adapter is not pinned in our notes.
    "plumb27b":  (None, "main", "suffix", "Qwen/Qwen3.5-27B",
                  "work/experiments/pool/parts/plumb_parts"),
}


def cache_dir(repo):
    return os.path.join(HUB, "models--" + repo.replace("/", "--"))


def cache_present(repo):
    return repo and os.path.isdir(cache_dir(repo)) and bool(glob.glob(cache_dir(repo) + "/snapshots/*"))


def extract_cmd(name, repo, kind, out):
    if kind == "peft":
        return f"python scripts/peft_to_parts.py <snapshot>/adapter_model.safetensors {out}"
    if kind == "delta":
        return f"python scripts/extract_delta.py <base_snapshot> <snapshot> {out} --rank 128"
    if kind == "clef":
        return (f"CLEF_QWEN_DIR=<base_snapshot> CLEF_CLEF_DIR=<snapshot> CLEF_OUT={out} "
                f"python scripts/extract_clef.py")
    return f"# {name}: suffix adapter conversion not scripted -> {out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    names = a.names or list(SOURCES)
    rc = 0
    for name in names:
        if name not in SOURCES:
            print(f"unknown source: {name}"); rc = 1; continue
        repo, rev, kind, base, out = SOURCES[name]
        print(f"\n== {name}: {repo or '<UNPINNED>'}@{rev} ({kind}); base {base}")
        if repo is None:
            print("   !! repo not pinned in notes (find & pin before fetch)"); rc = 1
        else:
            present = cache_present(repo)
            print(f"   adapter cache: {'present' if present else 'MISSING'}")
            if not present and not a.check:
                from huggingface_hub import snapshot_download
                print(f"   downloading -> {snapshot_download(repo, revision=rev)}")
            elif not present:
                rc = 1
        bp = cache_present(base)
        print(f"   base cache: {'present' if bp else 'MISSING'} ({base})")
        print(f"   parts target: {out} ({'exists' if os.path.isdir(out) else 'not built yet'})")
        print(f"   slice: {extract_cmd(name, repo, kind, out)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
