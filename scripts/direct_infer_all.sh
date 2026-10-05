#!/usr/bin/env bash
# Direct-generation check for every model x every work/best LoRA (+ base), 50 prompts each.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
B=$ROOT/work/best
H=${HF_HOME:-$HOME/.cache/huggingface}/hub
PY=${BONJEV_PYTHON:-python3}
OUT=${TMPDIR:-/tmp}/bonjev/direct
mkdir -p "$OUT"

m() { ls "$H/models--prism-ml--Ternary-Bonsai-$1-gguf/snapshots"/*/"$2" 2>/dev/null | head -1; }
M17=$(m 1.7B Ternary-Bonsai-1.7B-PQ2_0.gguf)
M4=$(m 4B Ternary-Bonsai-4B-PQ2_0.gguf)
M8=$(m 8B Ternary-Bonsai-8B-PQ2_0.gguf)
M27=$(m 2-27B Ternary-Bonsai-2-27B-PTQ1_0.gguf)

run() { echo "### $1 $2"; "$PY" "$ROOT/scripts/direct_infer.py" "$1" "$2" 8890 50 2>&1; }

{
run "$M17" -
run "$M17" "$B/1.7b/m17_tiny25_single.gguf"
run "$M4" -
run "$M4" "$B/4b/B10.gguf"
run "$M4" "$B/4b/alt/m4b_all3_align.gguf"
run "$M8" -
run "$M8" "$B/8b/m8_inv_b04.gguf"
run "$M8" "$B/8b/alt/R3.gguf"
run "$M27" -
run "$M27" "$B/27b/vega_clef_plumb.gguf"
run "$M27" "$B/27b/alt/m27_clef_plumb_at_half.gguf"
run "$M27" "$B/27b/alt/vega_clef_plumb_at_half.gguf"
run "$M27" "$B/27b/alt/clef_plumb_vegaffn.gguf"
} > "$OUT/summary.txt" 2>&1
echo "DIRECT_DONE" >> "$OUT/summary.txt"
echo "wrote $OUT/summary.txt"
