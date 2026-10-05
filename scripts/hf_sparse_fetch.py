"""Fetch only selected tensors of a remote GGUF into a sparse local file at their original offsets.

Ported from professorpalmer/bonsai-ada-surgery surgery/hf_sparse_fetch.py, with the gguf import
adapted to this repo (vendor/prism-llama.cpp/gguf-py, falling back to an installed `gguf`).

usage: python3 scripts/hf_sparse_fetch.py <url> <out.gguf> <name-prefix> [<name-prefix> ...]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

_here = os.path.dirname(os.path.abspath(__file__))
_gguf_py = os.path.join(_here, "..", "vendor", "prism-llama.cpp", "gguf-py")
if os.path.isdir(_gguf_py):
    sys.path.insert(0, _gguf_py)
from gguf import GGUFReader  # noqa: E402

HEADER_PROBE = 96 << 20  # header + tensor infos + tokenizer vocab fit comfortably in this


def fetch_range(url: str, start: int, end: int, fh, retries: int = 6) -> None:
    """Stream bytes [start, end] into fh at offset start."""
    pos = start
    while pos <= end:
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"expected 206, got {r.status}")
                fh.seek(pos)
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pos += len(chunk)
                    print(f"\r  {pos - start:>13,} / {end - start + 1:,}", end="", flush=True)
            print()
            return
        except Exception as e:  # noqa: BLE001
            retries -= 1
            if retries <= 0:
                raise
            print(f"\n  retry after {e}", flush=True)
            time.sleep(3)


def total_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def make_sparse(path: str, size: int) -> None:
    with open(path, "wb") as fh:
        fh.truncate(size)
    if os.name == "nt":
        subprocess.run(["fsutil", "sparse", "setflag", path], check=True, capture_output=True)
        subprocess.run(["fsutil", "sparse", "setrange", path, "0", str(size)], check=True, capture_output=True)


def main() -> None:
    args = sys.argv[1:]
    resume = "--resume" in args
    skip_until = None
    if resume:
        args.remove("--resume")
    if "--skip-until" in args:
        i = args.index("--skip-until")
        skip_until = args[i + 1]
        del args[i:i + 2]
    url, out, prefixes = args[0], args[1], args[2:]
    size = total_size(url)
    print(f"remote size {size:,}")
    if resume and os.path.exists(out) and os.path.getsize(out) == size:
        print("resuming into existing sparse file")
    else:
        make_sparse(out, size)
        with open(out, "r+b") as fh:
            print("header:")
            fetch_range(url, 0, min(HEADER_PROBE, size) - 1, fh)
    rd = GGUFReader(out, "r")
    wanted = []
    skipping = skip_until is not None
    for t in rd.tensors:
        if any(t.name.startswith(p) for p in prefixes):
            if skipping:
                if t.name == skip_until:
                    skipping = False
                else:
                    print(f"skip (already fetched) {t.name}")
                    continue
            start = int(t.data_offset)
            nbytes = int(t.n_bytes)
            wanted.append((t.name, start, start + nbytes - 1))
    del rd
    if not wanted:
        sys.exit("no tensors matched")
    tot = sum(e - s + 1 for _, s, e in wanted)
    print(f"{len(wanted)} tensors, {tot:,} bytes to fetch")
    with open(out, "r+b") as fh:
        for name, s, e in wanted:
            print(f"{name} [{s:,}..{e:,}]")
            fetch_range(url, s, e, fh)
    print("done")


if __name__ == "__main__":
    main()
