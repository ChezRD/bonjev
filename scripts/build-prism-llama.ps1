# Build the patched PrismML llama.cpp for bonjev on Windows without the CUDA toolkit.
# Ported from professorpalmer/bonsai-ada-surgery build/build_windows.ps1 and adapted to
# bonjev's patch sets and per-revision source trees.
#
#   python -m pip install cmake ninja nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-nvrtc
#   .\scripts\build-prism-llama.ps1                      # all ada patches, arch from nvidia-smi
#   .\scripts\build-prism-llama.ps1 -Arch 86 -Server     # also build llama-server/bench/perplexity
#
# Output: vendor\win-<rev>\build\bin (shared libs for bonjev; tools with -Server).
# Untested on this machine (Linux); kept in sync with scripts/build-prism-llama.sh.
param(
    [string[]] $Sets = @('all'),
    [string] $Arch = "",
    [string] $Base = "adfffbe41b2cabcd51fff326ab045662265062bb",
    [string] $Tag = "prism-b10743-adfffbe",
    [switch] $Server,
    [switch] $Tests,
    [switch] $SkipCargo
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

# --- patch list in series order ---
$ordered = @()
for ($i = 1; $i -le 33; $i++) {
    $name = "{0:D4}-*.patch" -f $i
    $hit = Get-ChildItem (Join-Path $Root 'patches') -Recurse -Filter $name | Select-Object -First 1
    if ($hit) { $ordered += $hit.FullName }
}
$selected = @()
foreach ($set in $Sets) {
    if ($set -eq 'all') { $selected = $ordered; continue }
    $dir = Join-Path $Root "patches\$set"
    if (-not (Test-Path $dir)) { throw "no such patch set: $set" }
    foreach ($p in $ordered) { if ($p -like "$dir\*") { $selected += $p } }
}
if ($selected.Count -eq 0) { throw 'no patches selected' }

# --- per-revision source tree (same scheme as the Linux script) ---
$sha = [System.Security.Cryptography.SHA256]::Create()
$text = $Base + "`n" + (($selected | ForEach-Object { Get-Content -Raw $_ }) -join "`n")
$rev = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))) -replace '-', '').ToLower()
$Src = Join-Path $Root "vendor\win-$rev"
$Build = Join-Path $Src 'build'
if (-not (Test-Path (Join-Path $Src 'CMakeLists.txt'))) {
    $stage = Join-Path $Root 'vendor\.win-stage'
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force $stage | Out-Null
    git -C $stage init -q
    git -C $stage remote add origin https://github.com/PrismML-Eng/llama.cpp.git
    git -C $stage fetch --depth 1 origin tag $Tag
    git -C $stage checkout --detach FETCH_HEAD
    if ((git -C $stage rev-parse HEAD) -ne $Base) { throw "tag $Tag is not $Base" }
    foreach ($p in $selected) { git -C $stage apply --check $p; git -C $stage apply $p }
    New-Item -ItemType Directory -Force (Split-Path $Src -Parent) | Out-Null
    Move-Item $stage $Src
}
Write-Host "source: $Src"

# --- GPU arch ---
if (-not $Arch) {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
    if (-not $cc) { throw 'nvidia-smi not found; pass -Arch' }
    $Arch = $cc.Replace('.', '')
}
$ArchList = ($Arch -split '[;,]' | ForEach-Object { $a = $_.Trim(); if ($a -match '-') { $a } else { "$a-real" } }) -join ';'

# --- Visual Studio ---
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw 'Visual Studio Build Tools 2022 (C++ workload) not found' }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat missing under $vs" }

# --- CUDA: toolkit if present, else pip wheels ---
$Cuda = $env:CUDA_PATH
if (-not ($Cuda -and (Test-Path (Join-Path $Cuda 'bin\nvcc.exe')))) {
    $Cuda = & python -c @"
import os, site
dirs = site.getsitepackages() + [site.getusersitepackages()]
hits = [os.path.join(d, 'nvidia', 'cu13') for d in dirs if os.path.exists(os.path.join(d, 'nvidia', 'cu13', 'bin', 'nvcc.exe'))]
print(hits[0] if hits else '')
"@
    if (-not $Cuda) {
        throw 'no CUDA compiler: install the toolkit or `python -m pip install nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-nvrtc`'
    }
    $pip = $true
} else { $pip = $false }

function Find-Tool([string]$name) {
    $c = Get-Command $name -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $scripts = & python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
    $p = Join-Path $scripts "$name.exe"
    if (Test-Path $p) { return $p }
    throw "$name not found; python -m pip install $name"
}
$cmake = Find-Tool cmake
$ninja = Find-Tool ninja

New-Item -ItemType Directory -Force $Build | Out-Null
$targets = 'llama mtmd ggml ggml-cpu ggml-cuda ggml-base'
if ($Server) { $targets += ' llama-server llama-bench llama-perplexity' }
$testFlag = 'OFF'
if ($Tests) { $targets += ' test-backend-ops'; $testFlag = 'ON' }

$implibs = ''
if ($pip) {
    $implibs = @"
if not exist "$Cuda\lib\x64" mkdir "$Cuda\lib\x64"
for %%D in (cublas64_13 cublasLt64_13 cudart64_13) do (
  if exist "$Cuda\bin\x86_64\%%D.dll" if not exist "$Cuda\bin\%%D.dll" copy /y "$Cuda\bin\x86_64\%%D.dll" "$Cuda\bin\" >nul
)
if not exist "$Cuda\lib\x64\cublas.lib"   powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\make-import-lib.ps1" "$Cuda\bin\cublas64_13.dll"   "$Cuda\lib\x64\cublas.lib"
if not exist "$Cuda\lib\x64\cublasLt.lib" powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\make-import-lib.ps1" "$Cuda\bin\cublasLt64_13.dll" "$Cuda\lib\x64\cublasLt.lib"
if not exist "$Cuda\lib\x64\cudart.lib"   powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\make-import-lib.ps1" "$Cuda\bin\cudart64_13.dll"   "$Cuda\lib\x64\cudart.lib"
"@
}

$bat = Join-Path $Build 'build.bat'
@"
@echo off
call "$vcvars" >nul
set CUDA_PATH=$Cuda
set CUDAToolkit_ROOT=$Cuda
set PATH=$Cuda\bin;$Cuda\bin\x86_64;$Cuda\nvvm\bin;%PATH%
$implibs
"$cmake" -S "$Src" -B "$Build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="$ninja" -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES="$ArchList" -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=$testFlag -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DCMAKE_CUDA_COMPILER="$Cuda\bin\nvcc.exe" -DCUDAToolkit_ROOT="$Cuda"
if errorlevel 1 exit /b 1
"$cmake" --build "$Build" --target $targets -j
if errorlevel 1 exit /b 1
"@ | Set-Content -Path $bat -Encoding ASCII

Write-Host "arch=$ArchList cuda=$Cuda ($(if ($pip) {'pip wheels'} else {'toolkit'}))"
cmd /c $bat
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

# --- stage ---
$out = Join-Path $Root "tooling\win-$rev"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item (Join-Path $Build 'bin\*.exe') $out -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Build 'bin\*.dll') $out -Force
if ($pip) {
    foreach ($d in 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'nvJitLink_130_0.dll', 'nvvm64_40_0.dll') {
        $f = Get-ChildItem $Cuda -Recurse -Filter $d -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { Copy-Item $f.FullName $out -Force }
    }
}
Set-Content -Path (Join-Path $Root 'tooling\win-lib-dir') -Value $out -Encoding ASCII
Write-Host "staged $out"

if (-not $SkipCargo) {
    Write-Host "== cargo build --release (PRISM_LLAMA_DIR=$out)"
    $env:PRISM_LLAMA_DIR = $out
    & cargo build --release --manifest-path (Join-Path $Root 'Cargo.toml')
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
}
