#!/usr/bin/env bash
# Build the Prism llama.cpp fork for the local machine, with bonjev's patch sets.
#
# Ported from professorpalmer/bonsai-ada-surgery build/build_linux.sh.
# Each patch-set revision gets its own source+build tree under vendor/linux-$REV,
# so different sets coexist and repeated runs are cached. The CUDA architecture
# defaults to "native" (the GPU in this machine); no cross-platform build.
#
# Usage:
#   scripts/build-prism-llama.sh [set ...]
# Default: all ada-surgery patches (the series is linear). Sets:
#   all | base prefill decode kv batch-invariant mtp server sampling diagnostics
# Env: BONSAI_CUDA_ARCH (default native), BONSAI_BUILD_JOBS (default 4)
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BASE=adfffbe41b2cabcd51fff326ab045662265062bb
TAG=prism-b10743-adfffbe
ARCH=${BONSAI_CUDA_ARCH:-native}
JOBS=${BONSAI_BUILD_JOBS:-4}

if [[ $(uname -s) != Linux ]]; then
    echo 'This script builds on Linux.' >&2
    exit 1
fi
for tool in git cmake nvcc c++ sha256sum; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool. Install Git, CMake >=3.24, a CUDA-compatible C++ compiler, and the CUDA toolkit (nvcc on PATH)." >&2; exit 1; }
done
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo 'BONSAI_BUILD_JOBS must be a positive integer.' >&2; exit 1; }
[[ $ARCH =~ ^(native|[0-9]+(-real|-virtual)?)(\;[0-9]+(-real|-virtual)?)*$ ]] || { echo 'BONSAI_CUDA_ARCH must be native or CUDA architectures such as 89 or 86;89.' >&2; exit 1; }
version=$(cmake --version | head -n 1)
if [[ ! $version =~ ([0-9]+)\.([0-9]+) ]] || (( BASH_REMATCH[1] < 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] < 24) )); then
    echo "CMake >=3.24 required; found $version" >&2
    exit 1
fi

SETS=("$@")
if [ ${#SETS[@]} -eq 0 ]; then
    SETS=(all)
fi

# Patch list in the original series order.
ordered=()
for n in $(seq 1 33); do
    f=$(find "$ROOT/patches" -maxdepth 2 -name "$(printf '%04d' "$n")-*.patch" | head -1)
    [ -n "$f" ] && ordered+=("$f")
done
PATCHES=()
for set in "${SETS[@]}"; do
    if [ "$set" = "all" ]; then
        PATCHES=("${ordered[@]}")
        continue
    fi
    if [ ! -d "$ROOT/patches/$set" ]; then
        echo "no such patch set: $set" >&2
        exit 1
    fi
    for p in "${ordered[@]}"; do
        case "$p" in
            "$ROOT/patches/$set/"*) PATCHES+=("$p") ;;
        esac
    done
done
[ ${#PATCHES[@]} -gt 0 ] || { echo 'No patches selected.' >&2; exit 1; }

REV=$( { printf '%s\n' "$BASE"; cat "${PATCHES[@]}"; } | sha256sum)
REV=${REV%% *}
SRC="$ROOT/vendor/linux-$REV"
mkdir -p "$ROOT/vendor" "$ROOT/tooling"
if [[ ! -d $SRC ]]; then
    STAGE=$(mktemp -d "$ROOT/vendor/.linux-build.XXXXXX")
    trap 'rm -rf -- "$STAGE"' EXIT
    git -C "$STAGE" init -q
    git -C "$STAGE" remote add origin https://github.com/PrismML-Eng/llama.cpp.git
    git -C "$STAGE" fetch --depth 1 origin tag "$TAG"
    git -C "$STAGE" checkout --detach FETCH_HEAD
    [[ $(git -C "$STAGE" rev-parse HEAD) == "$BASE" ]] || { echo "tag $TAG is not $BASE" >&2; exit 1; }
    for patch in "${PATCHES[@]}"; do
        git -C "$STAGE" apply --check "$patch"
        git -C "$STAGE" apply "$patch"
    done
    git -C "$STAGE" diff --binary HEAD > "$STAGE/.git/bonsai-applied.patch"
    mv -- "$STAGE" "$SRC"
    trap - EXIT
fi
if ! git -C "$SRC" diff --binary HEAD | cmp -s - "$SRC/.git/bonsai-applied.patch"; then
    echo "Source changed since setup: $SRC. Use a different patch set or a fresh bundle checkout." >&2
    exit 1
fi

# Shared libraries and mtmd, because bonjev links libllama/libmtmd/libggml*.so.
cmake -S "$SRC" -B "$SRC/build-linux" -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$SRC/build-linux" --target llama mtmd ggml ggml-cpu ggml-cuda ggml-base -j "$JOBS"
[[ -f "$SRC/build-linux/bin/libllama.so" ]] || { echo 'Build did not produce libllama.so.' >&2; exit 1; }
printf '%s\n' "$SRC/build-linux/bin" > "$ROOT/tooling/prism-lib-dir"

# Build the Rust binary against this revision unless asked not to.
if [ "${SKIP_CARGO:-0}" != "1" ]; then
    echo "== cargo build --release (PRISM_LLAMA_DIR=$SRC/build-linux/bin)"
    (cd "$ROOT" && PRISM_LLAMA_DIR="$SRC/build-linux/bin" cargo build --release)
fi

printf '\nBuild complete. Use with bonjev:\n  PRISM_LLAMA_DIR="%s/build-linux/bin" cargo build --release\n' "$SRC"
