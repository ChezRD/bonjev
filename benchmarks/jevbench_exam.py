#!/usr/bin/env python3
"""Score the public JevBench set (easy + original + hard, 231) on BonJev.

Downloads the three splits from a pinned jevbench commit into
benchmarks/cache/jevbench, then POSTs each task to /v1/systemone.
If that port is closed, starts target/release/bonjev and stops it afterwards.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

COMMIT = "3749b4fc1b88e4f5f02a3c0b9766c4ffa57891c0"
RAW = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public"
SPLITS = {
    "easy.jsonl": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original.jsonl": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard.jsonl": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "jevbench"
REPO = ROOT.parent


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def download_splits(refresh):
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expect in SPLITS.items():
        path = CACHE / name
        if path.is_file() and not refresh and sha256(path) == expect:
            paths.append(path)
            continue
        url = f"{RAW}/{name}"
        print(f"download {name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        path.write_bytes(data)
        got = sha256(path)
        if got != expect:
            path.unlink()
            raise SystemExit(f"{name} sha256 {got} != {expect}")
        paths.append(path)
    return paths


def tasks(paths):
    rows = []
    for path in paths:
        split = path.name.removesuffix(".jsonl")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_split"] = split
            rows.append(row)
    return rows


def pred_of(body, kind):
    answer = body["answers"]["decision"]
    if kind == "choice":
        return str(answer.get("choice"))
    if kind == "noul":
        value = answer.get("noul")
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return "yes" if float(value) >= 0.5 else "no"
        prob = answer.get("probability")
        if prob is None:
            probs = answer.get("probabilities") or {}
            prob = probs.get("true", probs.get("yes"))
        return "yes" if float(prob) >= 0.5 else "no"
    if kind == "score":
        probs = answer.get("probabilities") or {}
        if probs:
            return max(probs, key=lambda key: float(probs[key]))
    return None


def origin_of(url):
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.hostname}:{port}", port


def server_up(origin):
    try:
        with urllib.request.urlopen(f"{origin}/v1/models", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def start_server(port, model, ctx=16384):
    binary = REPO / "target" / "release" / "bonjev"
    if not binary.is_file():
        raise SystemExit(f"missing {binary}; run cargo build --release")
    log_path = CACHE / "serve.log"
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(binary),
            "serve",
            "--model",
            model,
            "--port",
            str(port),
            "--ctx",
            str(ctx),
            "--no-vision",
        ],
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}; see {log_path}")
        if "listening on" in log_path.read_text(errors="replace"):
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"server did not listen; see {log_path}")


def score(url, rows, out_path, dump_path=None):
    ok = 0
    times = []
    out_path.write_text("")
    if dump_path is not None:
        dump_path.write_text("")
    started_all = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        question = row["question"]
        payload = {"state": row["state"], "questions": {"decision": question}}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=float(os.environ.get("BONJEV_EXAM_TIMEOUT", "120"))) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            raise SystemExit(f"{row['id']} HTTP {err.code}: {detail}") from err
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        kind = question["type"]
        got = pred_of(body, kind)
        expected = str(row["expected"]).lower() if kind == "noul" else str(row["expected"])
        hit = got == expected
        ok += int(hit)
        with out_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "split": row["_split"],
                        "kind": kind,
                        "expected": expected,
                        "got": got,
                        "ok": hit,
                        "sec": round(elapsed, 4),
                    }
                )
                + "\n"
            )
        if index % 25 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} correct {ok}", flush=True)
        if dump_path is not None:
            answer = body.get("answers", {}).get("decision", {}) if isinstance(body, dict) else {}
            with dump_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "split": row["_split"],
                            "kind": kind,
                            "expected": expected,
                            "got": got,
                            "ok": hit,
                            "probability": answer.get("probability"),
                            "noul": answer.get("noul"),
                            "probabilities": answer.get("probabilities") or {},
                            "confidence": answer.get("confidence"),
                            "margin": answer.get("margin"),
                            "entropy": answer.get("entropy"),
                            "answer_first": answer.get("answer_first"),
                        }
                    )
                    + "\n"
                )
    wall = time.perf_counter() - started_all
    times.sort()
    mid = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    print(
        f"correct {ok}/{len(rows)} wall {wall:.1f}s median {mid:.3f}s p95 {p95:.3f}s"
    )
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8830/v1/systemone",
        help="BonJev POST /v1/systemone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CACHE / "exam.jsonl",
        help="per-task jsonl",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download the splits again",
    )
    parser.add_argument(
        "--model",
        default="ternary-bonsai-2-27b",
        help="GGUF id for bonjev serve when the port is down (bonsai-4b, bonsai-1.7b, …)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=16384,
        help="context for bonjev serve when the port is down",
    )
    parser.add_argument(
        "--dump-probs",
        type=Path,
        default=None,
        help="write per-task probabilities/margin/entropy dump",
    )
    args = parser.parse_args()
    paths = download_splits(args.refresh)
    rows = tasks(paths)
    if len(rows) != 231:
        raise SystemExit(f"expected 231 tasks, got {len(rows)}")
    origin, port = origin_of(args.url)
    proc = None
    started_here = False
    if server_up(origin):
        print(f"using server already on {origin}", flush=True)
    else:
        print(f"starting bonjev --model {args.model} ctx {args.ctx} on port {port}", flush=True)
        proc = start_server(port, args.model, args.ctx)
        started_here = True
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        score(args.url, rows, args.out, args.dump_probs)
    finally:
        if started_here and proc is not None and proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
