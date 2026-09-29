# BonJev

BonJev is a fast single-step decision engine for Bonsai models on llama.cpp, written in Rust.

Like dozens of other open-source implementations on the JevBench leaderboard, BonJev reproduces the core paradigm of Jev (System One) on local consumer hardware. Instead of calling proprietary hosted APIs or generating conversational text that requires JSON parsing, it evaluates closed choices, scores, and boolean verdicts on open weights through a single forward pass over token logits.

## How It Works

Generative language models produce text one token at a time. When software only needs a choice, score, or boolean decision, text generation introduces parsing failures, nondeterminism, and high latency.

BonJev replicates the Jev decision contract locally. It formats the input state, criteria, and allowed options into a structured prompt. It evaluates the model once, extracts logits for the allowed option labels at the decision boundary, and normalizes them with softmax. The engine does not run an autoregressive generation loop.

Key capabilities:
- **Single-step inference:** Returns probabilities over candidate options from one forward pass.
- **Logit aggregation:** Extracts bare and space-prefixed label tokens, combining them with log-sum-exp.
- **Prior calibration:** Computes background option bias on a neutral prompt and calibrates output probabilities.
- **Multimodal support:** Reads images (JPEG, PNG, WebP, GIF, BMP) and encodes them via mtmd at native resolution.
- **Memory efficiency:** Runs 27B ternary weights at 32k context in ~7.9 GB VRAM, fitting within 16 GB GPUs.

## Codebase Architecture

The crate is structured as a safe Rust layer over a minimal C shim and prebuilt llama.cpp binaries:

| Path | Purpose |
|---|---|
| `src/main.rs` | CLI entry point for `serve`, `ask`, and `models` commands. |
| `src/server.rs` | Axum HTTP server exposing `POST /v1/systemone` and `GET /v1/models`. |
| `src/decision.rs` | Request deserialization, prompt dispatch, prior computation, and response formatting. |
| `src/prompt.rs` | Prompt templating, label bank assignment (up to 255 options), and media tags. |
| `src/readout.rs` | Logit extraction, label token pairing, prior adjustment, and softmax. |
| `src/features.rs` | Prompt layout routing and thinking prefill selection. |
| `src/engine.rs` | Safe wrapper for model initialization, decoding, and prior caching. |
| `src/ffi.rs` | Foreign function declarations for the C shim. |
| `src/hf.rs` | Hugging Face Hub download helpers and model registry. |
| `src/picture.rs` | Image decoding to raw RGB8 buffers using the `image` crate. |
| `c/src/shim.c` | C shim interfacing with `libllama` and `libmtmd`. |
| `build.rs` | Builds the static C shim and links dynamic libraries. |

### Dependencies

- **HTTP & Async:** `axum`, `tokio`
- **CLI & Serialization:** `clap`, `serde`, `serde_json`
- **Model & Media:** `hf-hub`, `image`, `base64`
- **Native Runtime:** Prebuilt Prism `libllama.so`, `libmtmd.so`, `libggml.so`

## Supported Models

BonJev loads GGUF models directly or downloads them through the Hugging Face Hub:

| Model ID | Repository | GGUF File | Disk Size |
|---|---|---|---|
| `q2` (default) | `prism-ml/Ternary-Bonsai-2-27B-gguf` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.2 GB |
| `q1` | `prism-ml/Bonsai-27B-gguf` | `Bonsai-27B-Q1_0.gguf` | 3.6 GB |

For `q2`, vision is enabled by default using the projector `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. Pass `--no-vision` to disable it.

## Building

### Quick Start

Prerequisites: Rust (2024 edition or newer), C compiler (`gcc` or `clang`), `curl`.

```bash
git clone https://github.com/ChezRD/bonjev.git
cd bonjev
cargo build --release
```

`build.rs` automatically runs `./scripts/fetch-prism.sh` to download and unpack the pinned Prism prebuilts (`prism-b10743-adfffbe`) into `vendor/prism-prebuilt/` if they are not already present. On Linux with an NVIDIA GPU, it auto-detects CUDA; otherwise, it falls back to the CPU archive.

### Manual / Custom Backend

To pre-download a specific backend before building:

```bash
./scripts/fetch-prism.sh               # auto-detects CUDA / CPU
./scripts/fetch-prism.sh cuda-13.3      # Linux x64 CUDA 13.3
./scripts/fetch-prism.sh cuda-12.8      # Linux x64 CUDA 12.8
./scripts/fetch-prism.sh cuda-12.4      # Linux x64 CUDA 12.4
./scripts/fetch-prism.sh vulkan         # Linux x64 Vulkan
./scripts/fetch-prism.sh rocm           # Linux x64 ROCm 7.2
./scripts/fetch-prism.sh cpu            # Linux / macOS CPU
```

Or point `PRISM_LLAMA_DIR` to existing prebuilt libraries:

```bash
PRISM_LLAMA_DIR=/path/to/prism-prebuilt cargo build --release
```

At runtime, the binary locates shared libraries using the embedded rpath or `PRISM_LLAMA_DIR`.

## Usage

### Check Cached Models

```bash
./target/release/bonjev models
```

### Run a One-Shot CLI Query

Categorical decision:

```bash
./target/release/bonjev ask --model q2 \
  --state "Customer requested a billing address update." \
  --question "Which department handles this request?" \
  --option "billing" \
  --option "technical support" \
  --option "sales"
```

Multimodal decision with an image:

```bash
./target/release/bonjev ask --model q2 \
  --image ./diagram.png \
  --state "Geometry problem diagram" \
  --question "What is the value of angle X?" \
  --option "30 degrees" \
  --option "45 degrees" \
  --option "60 degrees"
```

### Start the HTTP Server

```bash
./target/release/bonjev serve --model q2 --port 8080
```

The server binds to `127.0.0.1:8080` and exposes two endpoints:
- `GET /v1/models`: Returns active model metadata.
- `POST /v1/systemone`: Evaluates decisions.

#### Request Example

```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "User report: application crashed on startup with SIGSEGV.",
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "Classify the severity of this issue.",
        "criteria": {
          "critical": "Crashes, data loss, or security vulnerabilities.",
          "normal": "Functional bugs with available workarounds.",
          "minor": "Cosmetic or minor usability issues."
        }
      }
    }
  }'
```

#### Response Example

```json
{
  "model": "bonjev-q2",
  "answers": {
    "decision": {
      "type": "choice",
      "choice": "critical",
      "probabilities": {
        "critical": 0.942,
        "normal": 0.051,
        "minor": 0.007
      }
    }
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 1
  }
}
```

To send an image over HTTP, add a base64 string or data URL in the top-level `"image"` field, or embed it under `"image"` within the `"state"` object.

## Configuration

Behavior can be configured using environment variables:

| Variable | Default | Description |
|---|---|---|
| `PRISM_LLAMA_DIR` | from `prism-rel.txt` | Path to directory with `libllama.so` and `libmtmd.so`. |
| `BONJEV_SYSTEM` | Default system prompt | Custom system prompt text. |
| `BONJEV_PRIOR` | `1` | Set to `0` to disable prior calibration. |
| `BONJEV_PRIOR_ALPHA` | `0.10` | Prior blending strength. |
| `BONJEV_THINK_TAGS` | `1` | Set to `0` to omit `<think>` tags from the prompt. |
| `BONJEV_THINK_PREFILL` | `auto` | Prefill strategy for the think block (`auto`, `none`, or custom text). |
| `BONJEV_SUFFIX` | `Answer:` | Prompt suffix placed before the target decision token. |
| `BONJEV_SANDWICH` | `auto` | Sandwich prompt layout for long inputs (`auto`, `1`, `0`). |

## Performance

### Hardware Environment

All measurements below were run on the local testbed:
- **GPU:** NVIDIA GeForce RTX 4060 Ti 16 GB (165 W TDP)
- **Memory Bus & Bandwidth:** 128-bit GDDR6, theoretical 288 GB/s bandwidth (32 MB L2 cache)
- **Allocated Memory:** ~7.9 GB VRAM at 32k context

### Benchmark Results

| Benchmark | Model | Score | Total Time | Median Latency (All) | Short Requests Median (<300 tok) | Short Requests Range (Min – p90) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| JevBench Public (231 tasks) | `q2` (PQ2_0, 27B) | **202 / 231** (87.4%) | 186.4 s | 0.216 s | **0.190 s** | 0.131 s – 0.275 s | ~7.9 GB |


On short classification and intent tasks (<300 prompt tokens), BonJev delivers a median response time of **190 ms** on 288 GB/s memory bandwidth, without speculative decoding or prompt caching. Long-context items (up to 4k tokens) account for the tail latency up to 4.8 s.

## Limitations

- Batching (`--parallel > 1`) currently batches text-only requests. Requests containing images execute sequentially.
- Model execution is bound to the llama.cpp backend plugins available in `PRISM_LLAMA_DIR`.

## License and Attribution

- **BonJev Engine:** Licensed under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **llama.cpp / mtmd / ggml:** MIT License (Copyright (c) 2023–2026 The ggml authors).
- **Model Weights:** Ternary-Bonsai and Bonsai models are licensed under Apache 2.0 (derived from Qwen/Qwen3.8-27B). Weight downloads are subject to their respective model cards.
- **Trademarks:** Jev, System One, and TypeSafe are trademarks of their respective owners. BonJev is an independent project.
