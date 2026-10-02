#!/usr/bin/env bash
# Build the vendored Prism llama.cpp (vendor/prism-llama.cpp, pinned at adfffbe)
# with bonjev's own flags and a selectable set of patches from patches/.
#
# Usage:
#   scripts/build-prism-llama.sh [set ...]
# Default: all 33 ada-surgery patches (the series is linear; selective sets may fail),
#          experimental is NOT included by default.
# Sets:     all | base prefill decode kv batch-invariant mtp server sampling diagnostics experimental
# Env: CUDA_ARCH (default 89), JOBS, BUILD_DIR, CMAKE_CUDA_COMPILER, APPLY_ONLY=1
#
# The vendored tree is reset to the pinned commit before patching (git am -3), so
# repeated runs are clean. Patches are applied in their original series order
# (0001..0033), then patches/experimental on top.
# Result: $BUILD_DIR/bin (point bonjev at it with PRISM_LLAMA_DIR).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/vendor/prism-llama.cpp"
BUILD_DIR="${BUILD_DIR:-$SRC/build-bonjev}"
CUDA_ARCH="${CUDA_ARCH:-89}"
JOBS="${JOBS:-$(nproc)}"
PIN="adfffbe41b2cabcd51fff326ab045662265062bb"

SETS=("$@")
if [ ${#SETS[@]} -eq 0 ]; then
    SETS=(all)
fi

if [ ! -d "$SRC/.git" ]; then
    echo "missing vendored source: $SRC" >&2
    echo "run: git submodule update --init --depth 1" >&2
    exit 1
fi

# Original series order across the role directories (experimental excluded: its
# numbers collide, it is appended last on purpose).
ordered=()
for n in $(seq 1 33); do
    f=$(find "$ROOT/patches" -maxdepth 2 -name "$(printf '%04d' "$n")-*.patch" \
        -not -path '*/experimental/*' | head -1)
    [ -n "$f" ] && ordered+=("$f")
done

selected=()
for set in "${SETS[@]}"; do
    if [ "$set" = "experimental" ]; then
        continue
    fi
    if [ "$set" = "all" ]; then
        selected=("${ordered[@]}")
        continue
    fi
    if [ ! -d "$ROOT/patches/$set" ]; then
        echo "no such patch set: $set" >&2
        exit 1
    fi
    for p in "${ordered[@]}"; do
        case "$p" in
            "$ROOT/patches/$set/"*) selected+=("$p") ;;
        esac
    done
done
for set in "${SETS[@]}"; do
    if [ "$set" = "experimental" ]; then
        for p in "$ROOT/patches/experimental"/*.patch; do
            [ -e "$p" ] && selected+=("$p")
        done
    fi
done

if [ ${#selected[@]} -eq 0 ]; then
    echo "no patches selected" >&2
    exit 1
fi

echo "== reset $SRC to $PIN"
git -C "$SRC" reset --hard -q "$PIN"
git -C "$SRC" clean -qfd -e 'build*'

echo "== apply ${#selected[@]} patches (git am -3)"
for p in "${selected[@]}"; do
    echo "   $(basename "$p")"
    git -C "$SRC" -c user.name=bonjev -c user.email=bonjev@local am -3 --quiet "$p" || {
        echo "FAILED to apply: $p" >&2
        exit 1
    }
done

if [ "${APPLY_ONLY:-0}" = "1" ]; then
    echo "APPLY_ONLY=1: stopping before cmake"
    exit 0
fi

cmake -S "$SRC" -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/opt/cuda/bin/nvcc}"

cmake --build "$BUILD_DIR" -j"$JOBS" --target llama ggml ggml-cpu ggml-cuda ggml-base

echo
echo "built: $BUILD_DIR/bin"
echo "use with bonjev:  PRISM_LLAMA_DIR=$BUILD_DIR/bin cargo build --release"
