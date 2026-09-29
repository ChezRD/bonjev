# BonJev

BonJev is a fast single-step decision engine for Bonsai models on llama.cpp, written in Rust.

Like dozens of other open-source implementations on the [JevBench](https://github.com/fstandhartinger/jevbench) leaderboard, BonJev reproduces the core paradigm of Jev (System One) on local consumer hardware. Instead of calling proprietary hosted APIs or generating conversational text that requires JSON parsing, it evaluates closed choices, scores, and boolean verdicts on open weights through a single forward pass over token logits.

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
| `src/decision/` | Request deserialization, prompt dispatch, prior computation, and response formatting. |
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
| `bonsai2` (default) | `prism-ml/Ternary-Bonsai-2-27B-gguf` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.2 GB |
| `bonsai` | `prism-ml/Bonsai-27B-gguf` | `Bonsai-27B-Q1_0.gguf` | 3.6 GB |

`q2` still selects `bonsai2`. `q1` still selects `bonsai`. They are two models, not two quants of one file.

Vision is on by default for both models. `bonsai2` loads `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. `bonsai` loads `Bonsai-27B-mmproj-Q8_0.gguf`. Pass `--no-vision` to disable it.

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
./target/release/bonjev ask --model bonsai2 \
  --state "Customer requested a billing address update." \
  --question "Which department handles this request?" \
  --option "billing" \
  --option "technical support" \
  --option "sales"
```

Multimodal decision with an image:

```bash
./target/release/bonjev ask --model bonsai2 \
  --image ./diagram.png \
  --state "Geometry problem diagram" \
  --question "What is the value of angle X?" \
  --option "30 degrees" \
  --option "45 degrees" \
  --option "60 degrees"
```

### Start the HTTP Server

```bash
./target/release/bonjev serve --model bonsai2 --port 8080
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
  "model": "bonjev-bonsai2",
  "answers": {
    "decision": {
      "type": "choice",
      "choice": "critical",
      "probabilities": {
        "critical": 0.942,
        "normal": 0.051,
        "minor": 0.007
      },
      "confidence": 0.913
    }
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 1
  }
}
```

## Wire

`POST /v1/systemone` follows the TypeSafe shapes. Send `state` and `questions`. Each question is `choice`, `score`, or `noul`. The response has `model`, `answers` under the same keys, and `usage`.

A choice answer has `type`, `choice`, `probabilities`, and `confidence`. A score answer has `type`, `score`, `legend`, `probabilities`, and `confidence`. A noul answer has `type` and `noul`. `type: "boolean"` is accepted and answered as `noul`. A missing `type` is `choice`. Choice needs 2 to 255 options. Score needs 2 to 10 string levels.

The request `model` field does not choose the GGUF. The process loaded one file at startup. The response `model` is that id: `bonjev-bonsai2` or `bonjev-bonsai`.

A validation failure is HTTP 422:

```json
{"error":{"message":"choice needs 2 to 255 options","field":"questions.decision.criteria"}}
```

Broken JSON is also 422. `field` is left out when the parser cannot name one. An engine failure is 500, same envelope, no `field`.

These are local and are not TypeSafe fields: top-level `image`, `scores_are_calibrated`, and `GET /v1/models`. The server does not check an API key. It listens on `127.0.0.1`.

To send an image over HTTP, add a base64 string or data URL in the top-level `"image"` field, or embed it under `"image"` within the `"state"` object.

Several text questions in one request are scored together, up to six at a time. Their shared prompt head is stored once, and only the question tails are decoded separately. A seventh question starts a new group. Images stay one at a time.

After the model loads, one forward reads the logit row of an equal choice: state `N/A`, question `Which option follows?`, and one option per letter, `A: Option A.` through the letter bank. A choice answer subtracts `BONJEV_PRIOR_ALPHA` (default `0.10`) times that row's label logits, then applies softmax. Omit `scores_are_calibrated`, or set it to `true`, to keep that subtraction. Set it to `false` to score the request logits with no subtraction. `BONJEV_PRIOR=0` skips the forward and the subtraction. Noul and score ignore the field.

## Configuration

Behavior can be configured using environment variables:

| Variable | Default | Description |
|---|---|---|
| `PRISM_LLAMA_DIR` | from `prism-rel.txt` | Path to directory with `libllama.so` and `libmtmd.so`. |
| `BONJEV_SYSTEM` | Default system prompt | Custom system prompt text. |
| `BONJEV_PRIOR` | `1` | Set to `0` to disable prior calibration for every request. A request can also set `scores_are_calibrated` to `false`. |
| `BONJEV_PRIOR_ALPHA` | `0.10` | Prior blending strength. |
| `BONJEV_THINK_TAGS` | `1` | Set to `0` to omit `<think>` tags from the prompt. |
| `BONJEV_THINK_PREFILL` | `auto` | Prefill strategy for the think block (`auto`, `none`, or custom text). |
| `BONJEV_SUFFIX` | `Answer:` | Prompt suffix placed before the target decision token. |
| `BONJEV_SANDWICH` | `auto` | Sandwich prompt layout for long inputs (`auto`, `1`, `0`). |

## Performance

All measurements were taken on the same hardware testbed: **NVIDIA GeForce RTX 4060 Ti 16 GB**, 32k context, using the same release binary.

| Benchmark / Primitive | Metric | `bonsai2` (PQ2_0, 27B) | `bonsai` (Q1_0, 27B) | Notes & Details |
|---|---|---|---|---|
| [JevBench Public](https://github.com/fstandhartinger/jevbench) | Accuracy (231 tasks) | **202 / 231 (87.4%)** | **163 / 231 (70.6%)** | Short median: 0.19 s vs 0.18 s; overall median: 0.22 s vs 0.21 s |
| [Persian](https://github.com/ArmanJR/Jev-Persian-Benchmark) Main: Choice | Accuracy (240 questions) | **237 / 240 (98.8%)** | **207 / 240 (86.3%)** | Brier score: 0.038 (`bonsai2`) vs 0.211 (`bonsai`) |
| Persian Main: Noul | Accuracy (160 questions) | **154 / 160 (96.3%)** | **132 / 160 (82.5%)** | Threshold p ≥ 0.5. `bonsai2` prec/rec: 0.97/0.95; `bonsai`: 0.85/0.79 |
| Persian Main: Score | Accuracy (80 questions) | **68 / 80 (85.0%)** | **44 / 80 (55.0%)** | Margin ±0.5. MAE: 0.213 (`bonsai2`) vs 0.463 (`bonsai`) |
| Persian: English Control | Combined (48 questions) | **46 / 48 (95.8%)** | **41 / 48 (85.4%)** | Choice 29/30 vs 27/30, noul 9/10 vs 7/10, score 8/8 vs 7/8 |
| Persian: Repeats Stability | Combined (96 questions) | **92 / 96 (95.8%)** | **66 / 96 (68.8%)** | Choice 48/48 vs 42/48, noul 30/32 vs 20/32, score 14/16 vs 4/16 |
| Persian Total Wall Time | 624 answers (batch = 6) | **71.3 s** | **64.0 s** | 6-sequence shared prefix; `bonsai2` serial run was 101.1 s |
| Peak VRAM (32k context) | 1 sequence / 6 sequences | ~7.9 GB / ~12.4 GB | ~4.5 GB / ~8.7 GB | Model weights: ~9 GB (`bonsai2`) vs 3.6 GB (`bonsai`) |
| [BTZSC pilot](https://github.com/AbdelStark/jev-benchmarks) AG News | Accuracy (100 examples) | **90 / 100 (90.0%)** | not run | 4 labels. Hosted Jev 1.13.0 published 0.910. Wall 22.7 s |
| BTZSC DAIR Emotion | Accuracy (100 examples) | **45 / 100 (45.0%)** | not run | 6 labels. Hosted Jev 1.13.0 published 0.480. Wall 22.6 s |
| BTZSC Banking77 | Accuracy (100 examples) | **76 / 100 (76.0%)** | not run | 72 labels. Hosted Jev 1.13.0 published 0.870. Wall 174.1 s |
| [dhruvmehra/jevbench](https://github.com/dhruvmehra/jevbench) SST-2 | Accuracy (500, seed 0) | **470 / 500 (94.0%)** | not run | Validation split. Question: which category best describes the text |
| [AY Automate](https://www.ayautomate.com/blog/jev-vs-llm-benchmark) 8-way intent | Accuracy (160, seed 42) | **131 / 160 (81.9%)** | not run | Labels only. Their hosted Jev 1.13: 83.8% |
| AY Automate 77-way intent | Accuracy (231, seed 42) | **163 / 231 (70.6%)** | not run | Their hosted Jev 1.13: 78.8% |
| AY Automate prompt injection | Accuracy (400, seed 42) | **349 / 395 (88.4%)** | not run | 5 ordinary requests left out. Hosted Jev 1.13 on all 400: 87.0% |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test | Decisions (2,000) | **1,303 / 1,964 (66.3%)** | not run | 36 invoice questions left out. Hosted Jev on all 2,000: 0.740 |
| [sys1bench](https://github.com/rssr25/system-one-bench) canary | Choice and noul | **888 / 900 (98.7%)** | not run | Score 107/159 after 41 questions were left out |
| sys1bench main, choice | Accuracy (n=500, seed 42) | **3,400 / 3,500 (97.1%)** | not run | Tickets, phishing class, best passage, multilingual queue: 500/500. Team 469, clause 480, guardrail intent 451 |
| sys1bench main, noul | Accuracy | **5,844 / 6,000 (97.4%)** | not run | Page-on-call 419/500 |
| sys1bench main, score | Argmax of the level | **2,531 / 3,093 (81.8%)** | not run | 407 questions left out |

How each run was scored, which questions were left out, and the remaining misses are in [benchmarks/](benchmarks/README.md). The 231-task public exam is `python3 benchmarks/jevbench_exam.py`. It downloads the pinned splits and scores `POST /v1/systemone`.

## Limitations

- One request scores at most six text questions at a time, because the server keeps six sequences. Further questions in that request are a second group. Images in a request are scored one at a time.
- `--parallel > 1` batches separate text-only HTTP requests. A request that contains an image is not included in that batch.
- Model execution is bound to the llama.cpp backend plugins available in `PRISM_LLAMA_DIR`.

## License and Attribution

- **BonJev:** Licensed under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **llama.cpp / mtmd / ggml:** MIT License (Copyright (c) 2023–2026 The ggml authors).
- **Model Weights:** Ternary-Bonsai and Bonsai models are licensed under Apache 2.0 (derived from Qwen/Qwen3.8-27B). Weight downloads are subject to their respective model cards.
- **Trademarks:** Jev, System One, and TypeSafe are trademarks of their respective owners. BonJev is an independent project.
