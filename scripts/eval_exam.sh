#!/usr/bin/env bash
# Run the JevBench-231 exam for one model + optional LoRA (deterministic) and
# print the score plus the fast-forward metrics used for selection.
#
# Usage:
#   scripts/eval_exam.sh <model> <lora|-> <out.jsonl> [--style S] [--ctx N] [--port P]
#
#   model   bonjev model id, e.g. ternary-bonsai-4b, ternary-bonsai-2-27b-ptq1
#   lora    path to a GGUF LoRA adapter, or - for the base model
#   out     where to write the exam results (231 JSONL lines)
#   --style bonjev style name (default: the per-axis default routing)
#   --ctx   context size (default 16384)
#
# Env:
#   PRISM_LLAMA_DIR  prism build bin dir (default: tooling/prism-lib-dir)
#   BONJEV_PYTHON    python for the metrics summary (default: python3)
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TMP=${TMPDIR:-/tmp}/bonjev
mkdir -p "$TMP"

model=${1:?usage: eval_exam.sh <model> <lora|-> <out.jsonl> [--style S] [--ctx N] [--port P]}
lora=${2:?}
out=${3:?}
shift 3
style=""; ctx=16384; port=8830
while [ $# -gt 0 ]; do
    case "$1" in
        --style) style=$2; shift 2 ;;
        --ctx) ctx=$2; shift 2 ;;
        --port) port=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "${PRISM_LLAMA_DIR:-}" ] && [ -f "$ROOT/tooling/prism-lib-dir" ]; then
    PRISM_LLAMA_DIR=$(cat "$ROOT/tooling/prism-lib-dir")
fi
[ -n "${PRISM_LLAMA_DIR:-}" ] || { echo "set PRISM_LLAMA_DIR (or run scripts/build-prism-llama.sh)" >&2; exit 1; }

serve_env=(env PRISM_LLAMA_DIR="$PRISM_LLAMA_DIR" BONJEV_NO_REUSE=1 BONJEV_CKPT=0 BONJEV_TOKEN_DUMP=1)
[ -n "$style" ] && serve_env+=(BONJEV_STYLE="$style")
[ "$lora" != "-" ] && serve_env+=(BONJEV_LORA="$lora")

"${serve_env[@]}" "$ROOT/target/release/bonjev" serve --model "$model" \
    --ctx "$ctx" --no-vision --port "$port" > "$TMP/eval_exam.serve.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

up=0
for _ in $(seq 1 180); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then up=1; break; fi
    sleep 2
done
[ "$up" = 1 ] || { echo "server did not start (see $TMP/eval_exam.serve.log)" >&2; exit 1; }

python3 "$ROOT/benchmarks/jevbench_exam.py" --url "http://127.0.0.1:$port/v1/systemone" \
    --ctx "$ctx" --out "$out" --dump-probs "$out.probs.jsonl" | tee "$TMP/eval_exam.log"

"${BONJEV_PYTHON:-python3}" - "$out.probs.jsonl" <<'PY'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = len(rows)
af = sum(1 for r in rows if r.get("answer_first")) / n * 100 if n else 0
m = [float(r["margin"]) for r in rows if r.get("margin") is not None]
med = statistics.median(m) if m else 0
mw = [float(r["margin"]) for r in rows if r.get("margin") is not None and not r.get("ok")]
print(f"rows={n} answer_first={af:.1f}% margin_med={med:.3f} "
      f"m_wrong={statistics.median(mw) if mw else 0:.3f}")
PY
