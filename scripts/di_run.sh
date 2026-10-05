#!/usr/bin/env bash
# Run one Decision Index configuration through the official kit:
#   bonjev server -> decision_index run --engine http -> score
#
# Usage:
#   scripts/di_run.sh <name> <model> <lora|-> <rows.jsonl.gz> [limit]
#
#   name    label for the run directory and the systemone "model" field
#   model   bonjev model id, e.g. ternary-bonsai-4b
#   lora    path to a GGUF LoRA adapter, or - for the base model
#   rows    suite rows file, e.g. .../release-v2-rebuilt/sample-300.jsonl.gz
#   limit   optional max number of rows
#
# Artifacts land in benchmarks/cache/di/<name> (results.jsonl, scores.json, index.json).
#
# Env:
#   DI_PYTHON        python with the decision-index kit installed (default: python3)
#   DI_CTX           context size (default 32768)
#   DIWORK           HF cache root for the suite (default ~/.cache/huggingface/decision-index)
#   PRISM_LLAMA_DIR  prism build bin dir (default: tooling/prism-lib-dir)
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TMP=${TMPDIR:-/tmp}/bonjev
mkdir -p "$TMP"

NAME=${1:?usage: di_run.sh <name> <model> <lora|-> <rows.jsonl.gz> [limit]}
MODEL=${2:?}
LORA=${3:?}
ROWS=${4:?}
LIMIT=${5:-}
CTX=${DI_CTX:-32768}
DIWORK=${DIWORK:-$HOME/.cache/huggingface/decision-index}
SUITE=${SUITE:-$DIWORK/artifacts/benchmark-suite/release-v2-rebuilt}
OUT=${DI_RUNS:-$ROOT/benchmarks/cache/di}/$NAME
LOG=$TMP/di/$NAME
mkdir -p "$LOG" "$OUT"

DI=${DI_PYTHON:-python3}
if [ -z "${PRISM_LLAMA_DIR:-}" ] && [ -f "$ROOT/tooling/prism-lib-dir" ]; then
    PRISM_LLAMA_DIR=$(cat "$ROOT/tooling/prism-lib-dir")
fi

# one GPU: serialize stages through a lock
exec 9>"$TMP/runner.lock"
if ! flock -w 43200 9; then
    echo "== di_run: lock wait timed out"
    exit 1
fi
echo "== di_run $NAME start $(date +%H:%M:%S)"
for _ in $(seq 1 180); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${free:-0}" -ge 10000 ] && break
    sleep 2
done

serve_env=(env PRISM_LLAMA_DIR="$PRISM_LLAMA_DIR" BONJEV_NO_REUSE=1)
[ "$LORA" != "-" ] && serve_env+=(BONJEV_LORA="$LORA")
"${serve_env[@]}" "$ROOT/target/release/bonjev" serve --model "$MODEL" \
    --ctx "$CTX" --no-vision --port 8830 > "$LOG/server.log" 2>&1 9>&- &
pid=$!
up=0
for _ in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/8830) 2>/dev/null; then up=1; break; fi
    sleep 1
done
if [ "$up" != 1 ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "== di_run $NAME: server did not start"
    kill "$pid" 2>/dev/null
    exit 1
fi

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
"$DI" -m decision_index run --engine http \
    --option base_url=http://127.0.0.1:8830 --option model="$NAME" \
    --rows "$ROWS" $LIMIT_ARG --out "$OUT" > "$LOG/run.log" 2>&1
rc=$?
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
sleep 3
if [ "$rc" != 0 ]; then
    echo "== di_run $NAME: run failed rc=$rc (see $LOG/run.log)"
    exit "$rc"
fi
"$DI" -m decision_index score --results "$OUT/results.jsonl" --suite-dir "$SUITE" > "$LOG/score.log" 2>&1
echo "== di_run $NAME done; results in $OUT (log $LOG)"
