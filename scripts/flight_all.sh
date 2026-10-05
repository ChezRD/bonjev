#!/usr/bin/env bash
# Run the Jev Flight Lab demo against OUR bonjev server, for every model x work/best LoRA.
# The flight is not reproducible (near-tie decisions cascade) -> 3 runs per combo.
#
# Env:
#   DEMO_DIR         path to the Jev Flight Lab demo (default: /tmp/typesafe-jev-drone-demo)
#   PRISM_LLAMA_DIR  prism build bin dir (default: tooling/prism-lib-dir)
#   TMPDIR           where logs go (default /tmp)
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
B=$ROOT/work/best
H=${HF_HOME:-$HOME/.cache/huggingface}/hub
DEMO=${DEMO_DIR:-/tmp/typesafe-jev-drone-demo}
OUT=${TMPDIR:-/tmp}/bonjev/flight
mkdir -p "$OUT"
if [ -z "${PRISM_LLAMA_DIR:-}" ] && [ -f "$ROOT/tooling/prism-lib-dir" ]; then
    PRISM_LLAMA_DIR=$(cat "$ROOT/tooling/prism-lib-dir")
fi

model_path() { ls "$H/models--prism-ml--Ternary-Bonsai-$1-gguf/snapshots"/*/"$2" 2>/dev/null | head -1; }
M17=$(model_path 1.7B Ternary-Bonsai-1.7B-PQ2_0.gguf)
M4=$(model_path 4B Ternary-Bonsai-4B-PQ2_0.gguf)
M8=$(model_path 8B Ternary-Bonsai-8B-PQ2_0.gguf)
M27=$(model_path 2-27B Ternary-Bonsai-2-27B-PTQ1_0.gguf)

stop_bonjev() { pkill -f "bonjev serve" 2>/dev/null; sleep 3; }

ensure_demo() {
    curl -sf http://127.0.0.1:8010/api/health >/dev/null 2>&1 && return 0
    ( cd "$DEMO" && setsid nohup uv run uvicorn backend.app:app --host 127.0.0.1 --port 8010 > "$OUT/demo.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8010/api/health >/dev/null 2>&1 && return 0; sleep 2; done
    return 1
}

run() { # label model lora
    label=$1; model=$2; lora=$3
    stop_bonjev
    if [ "$lora" = "-" ]; then
        env PRISM_LLAMA_DIR="$PRISM_LLAMA_DIR" BONJEV_NO_REUSE=1 BONJEV_CKPT=0 \
            ./target/release/bonjev serve --model "$model" --ctx 16384 --no-vision --port 8830 > "$OUT/${label}.serve.log" 2>&1 &
    else
        env PRISM_LLAMA_DIR="$PRISM_LLAMA_DIR" BONJEV_NO_REUSE=1 BONJEV_CKPT=0 BONJEV_LORA="$lora" \
            ./target/release/bonjev serve --model "$model" --ctx 16384 --no-vision --port 8830 > "$OUT/${label}.serve.log" 2>&1 &
    fi
    for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8830/v1/models >/dev/null 2>&1 && break; sleep 2; done
    echo "### $label"
    for rep in 1 2 3; do
        echo "-- run $rep"
        ( cd "$DEMO" && PYTHONPATH=. timeout 330 uv run --with websockets python "$ROOT/scripts/flight_probe.py" --port 8010 2>&1 )
    done
    stop_bonjev
}

ensure_demo || { echo "demo did not start"; exit 1; }
{
run 17_base      ternary-bonsai-1.7b "-"
run 17_tiny      ternary-bonsai-1.7b "$B/1.7b/m17_tiny25_single.gguf"
run 4_base       ternary-bonsai-4b "-"
run 4_b10        ternary-bonsai-4b "$B/4b/B10.gguf"
run 4_align      ternary-bonsai-4b "$B/4b/alt/m4b_all3_align.gguf"
run 8_base       ternary-bonsai-8b "-"
run 8_inv        ternary-bonsai-8b "$B/8b/m8_inv_b04.gguf"
run 8_r3         ternary-bonsai-8b "$B/8b/alt/R3.gguf"
run 27_base      ternary-bonsai-2-27b-ptq1 "-"
run 27_win       ternary-bonsai-2-27b-ptq1 "$B/27b/vega_clef_plumb.gguf"
run 27_at_half   ternary-bonsai-2-27b-ptq1 "$B/27b/alt/m27_clef_plumb_at_half.gguf"
run 27_vega_half ternary-bonsai-2-27b-ptq1 "$B/27b/alt/vega_clef_plumb_at_half.gguf"
run 27_vegaffn   ternary-bonsai-2-27b-ptq1 "$B/27b/alt/clef_plumb_vegaffn.gguf"
} > "$OUT/summary.txt" 2>&1
echo "FLIGHT_DONE" >> "$OUT/summary.txt"
echo "wrote $OUT/summary.txt"
