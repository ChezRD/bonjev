#!/usr/bin/env bash
# Build a final LoRA from its source deltas (parts) — one script for all recipes.
# See docs/RECIPES.md for the provenance and the table.
#
# Usage: [WORK=<dir>] scripts/build_lora.sh <recipe> [out_dir]
#   WORK      root holding the source parts (default: work/experiments)
#   out_dir   where to write <recipe>.gguf (default: $WORK/build)
#
# Recipes: B10 all3_align m8_inv_b04 R3 vega_clef_plumb vega_clef_plumb_at_half
#          clef_plumb_vegaffn m17_tiny25_single e27_early
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-$ROOT/work/experiments}
OUT=${2:-$WORK/build}
V=${BONJEV_PYTHON:-python3}
SP=$ROOT/scripts/spectral_merge3.py
GW=$ROOT/scripts/gguf_lora_writer.py
LS=$ROOT/scripts/layer_scale.py
mkdir -p "$OUT"

# source parts
POOL=$WORK/pool/parts
M8=$WORK/merges8b/parts
M17=$WORK/merges17b/parts
M=$WORK/merges/parts
CLEF=$WORK/preserve/parts/clef27b_r64

merge() { # out rank arch q8|f16 [--align --budget B --rescale] <inputs...>
    local out=$1 rank=$2 arch=$3 qt=$4; shift 4
    local parts="$OUT/$out.parts"
    "$V" "$SP" "$parts" "$rank" "$@"
    "$V" "$GW" "$parts" "$OUT/$out.gguf" "$rank" "$arch" "$qt"
}

slice() { # src dst lo hi pattern...
    local src=$1 dst=$2 lo=$3 hi=$4; shift 4
    [ $# -gt 0 ] || set -- ""
    mkdir -p "$dst"
    for i in $(seq "$lo" "$hi"); do
        for pat in "$@"; do
            for f in "$src"/blk.$i."$pat"*.safetensors; do [ -e "$f" ] && cp -n "$f" "$dst/" 2>/dev/null || true; done
        done
    done
}

recipe=${1:?usage: build_lora.sh <recipe>}
case "$recipe" in
    B10)
        slice "$POOL/kev4b_q3_parts" "$OUT/kev4b_early" 0 17 attn_ ffn_
        merge B10 64 qwen3 q8_0 --align --budget 0.4 --rescale \
            "$OUT/kev4b_early" "$POOL/candigate_parts" "$POOL/senna_parts" ;;
    all3_align)
        merge m4b_all3_align 64 qwen3 q8_0 --align --budget 0.3 --rescale \
            "$POOL/kev4b_q3_parts" "$POOL/candigate_parts" "$POOL/senna_parts" ;;
    m8_inv_b04)
        merge m8_inv_b04 128 qwen3 q8_0 --align --budget 0.4 --rescale \
            "$M8/kev8b_attn" "$M8/kev8b_ffn" "$M8/lct_attn:0.25" ;;
    R3)
        merge R3 128 qwen3 q8_0 --align --budget 0.5 --rescale \
            "$M8/kev8b_attn" "$M8/kev8b_ffn" "$M8/s_lct_down_late" ;;
    vega_clef_plumb)
        merge vega_clef_plumb 128 qwen35 q8_0 \
            "$CLEF" "$POOL/plumb_parts" "$M/vega27_parts" ;;
    vega_clef_plumb_at_half)
        merge vega_clef_plumb_at_half 128 qwen35 q8_0 \
            "$CLEF" "$POOL/plumb_parts" "$M/_at_parts_half" "$M/vega27_parts" ;;
    clef_plumb_vegaffn)
        merge clef_plumb_vegaffn 128 qwen35 q8_0 \
            "$CLEF" "$POOL/plumb_parts" "$M/vega_ffn" ;;
    m17_tiny25_single)
        "$V" "$LS" "$M17/m17_x_tiny" "$OUT/m17_tiny25_single.parts" 2.5
        "$V" "$GW" "$OUT/m17_tiny25_single.parts" "$OUT/m17_tiny25_single.gguf" 64 qwen3 q8_0 ;;
    e27_early)
        slice "$M/vega27_parts" "$OUT/e27_vega" 0 15
        slice "$POOL/plumb_parts" "$OUT/e27_plumb" 0 15
        slice "$M/_at_parts" "$OUT/e27_at" 0 15
        merge e27_early 128 qwen35 q8_0 --align --budget 0.4 --rescale \
            "$OUT/e27_vega" "$OUT/e27_plumb" "$OUT/e27_at" ;;
    *)
        echo "unknown recipe: $recipe" >&2; exit 1 ;;
esac
echo "built $OUT/$recipe.gguf"
