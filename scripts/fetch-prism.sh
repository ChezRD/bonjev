#!/bin/sh
set -eu

tag="prism-b10743-adfffbe"
dirname="llama-prism-b10743-adfffbe"

os=$(uname -s)
arch=$(uname -m)

case "$os" in
  Linux) host="linux" ;;
  Darwin) host="macos" ;;
  *)
    echo "Error: unsupported OS: $os. BonJev currently supports Linux and macOS." >&2
    exit 1
    ;;
esac

case "$arch" in
  x86_64|amd64) bits="x64" ;;
  aarch64|arm64) bits="arm64" ;;
  *)
    echo "Error: unsupported CPU: $arch" >&2
    exit 1
    ;;
esac

detect_kind() {
  if [ "$host" = "linux" ] && [ "$bits" = "x64" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      cuda_ver=$(nvidia-smi | grep -o 'CUDA.*Version: [0-9.]\+' | head -n 1 | awk '{print $NF}' || true)
      case "$cuda_ver" in
        13*) echo "cuda-13.3" ;;
        12.8*|12.9*) echo "cuda-12.8" ;;
        12*) echo "cuda-12.4" ;;
        *)   echo "cuda-12.4" ;;
      esac
      return
    fi
  fi
  echo "cpu"
}

kind="${1:-$(detect_kind)}"

case "$kind:$host:$bits" in
  cpu:linux:x64) stem="bin-ubuntu-x64" ext="tar.gz" ;;
  cpu:linux:arm64) stem="bin-ubuntu-arm64" ext="tar.gz" ;;
  cpu:macos:arm64) stem="bin-macos-arm64" ext="tar.gz" ;;
  cpu:macos:x64) stem="bin-macos-x64" ext="tar.gz" ;;

  cuda-13.3:linux:x64) stem="bin-linux-cuda-13.3-x64" ext="tar.gz" ;;
  cuda-12.8:linux:x64) stem="bin-linux-cuda-12.8-x64" ext="tar.gz" ;;
  cuda-12.4:linux:x64) stem="bin-linux-cuda-12.4-x64" ext="tar.gz" ;;

  vulkan:linux:x64) stem="bin-ubuntu-vulkan-x64" ext="tar.gz" ;;
  vulkan:linux:arm64) stem="bin-ubuntu-vulkan-arm64" ext="tar.gz" ;;
  rocm:linux:x64) stem="bin-ubuntu-rocm-7.2-x64" ext="tar.gz" ;;
  kleidiai:macos:arm64) stem="bin-macos-arm64-kleidiai" ext="tar.gz" ;;

  *)
    echo "Error: no '$kind' prebuilt archive for $host $bits." >&2
    echo "Supported backends on Linux x64: cuda-13.3, cuda-12.8, cuda-12.4, vulkan, rocm, cpu" >&2
    echo "Supported backends on macOS: cpu, kleidiai" >&2
    exit 1
    ;;
esac

asset="${dirname}-${stem}.${ext}"
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
dest="$root/vendor/prism-prebuilt"
url="https://github.com/PrismML-Eng/llama.cpp/releases/download/${tag}/${asset}"
archive=$(mktemp)

mkdir -p "$dest"
echo "Fetching Prism prebuilt ($kind): $asset"
curl -fL "$url" -o "$archive"

tar -xzf "$archive" -C "$dest"
rm -f "$archive"

libs="$dest/$dirname"
if [ -f "$libs/libllama.so" ] || [ -f "$libs/libllama.dylib" ]; then
  echo "Prism libraries ready in $libs"
else
  echo "Error: unpacked $asset, but libllama not found in $libs" >&2
  exit 1
fi
