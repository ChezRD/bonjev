# BonJev

BonJev is a fast single-step decision engine for Bonsai models on llama.cpp, written in Rust. It also
ships a set of **training-free LoRA adapters** that raise the decision score of each Bonsai model.

Like dozens of other open-source implementations on the [JevBench](https://github.com/fstandhartinger/jevbench)
leaderboard, BonJev reproduces the core paradigm of Jev (System One) on local consumer hardware. Instead
of calling proprietary hosted APIs or generating conversational text that requires JSON parsing, it
evaluates closed choices, scores, and boolean verdicts on open weights through a single forward pass
over token logits.

**Contents**

- [How it works](#how-it-works)
- [Model architecture](#model-architecture)
- [Codebase architecture](#codebase-architecture)
- [Supported models](#supported-models)
- [LoRA adapters](#lora-adapters)
- [Building](#building)
- [Usage](#usage)
- [Wire](#wire)
- [Configuration](#configuration)
- [Performance](#performance)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [License and attribution](#license-and-attribution)

## How it works

Generative language models produce text one token at a time. When software only needs a choice, score,
or boolean decision, text generation introduces parsing failures, nondeterminism, and high latency.

BonJev replicates the Jev decision contract locally. It formats the input state, criteria, and allowed
options into a structured prompt, evaluates the model once, extracts the logits for the allowed option
labels at the decision boundary, and normalizes them with softmax. The engine does not run an
autoregressive generation loop.

Key capabilities:

- **Single-step inference:** returns probabilities over candidate options from one forward pass.
- **One letter:** each option is a single letter, `W` `X` `Y` first, then the rest of `A`–`Z` (26
  labels). The prompt ends with the answer lead (`Answer:` by default) and no newline after it; the
  scored token is the one piece `lead + space + letter`.
- **Multimodal support:** reads images (JPEG, PNG, WebP, GIF, BMP) and encodes them via mtmd at native
  resolution.
- **Memory efficiency:** runs 27B ternary weights at 32k context in ~7.9 GB VRAM, fitting within 16 GB
  GPUs.

## Model architecture

The Bonsai family spans two architectures: the 1.7B / 4B / 8B are dense `qwen3`, and the 27B is a
hybrid 3:1 `qwen35` stack (48 Gated DeltaNet blocks and 16 full-attention blocks, at 3, 7, …, 63).

Only the full-attention blocks carry a KV cache, so a KV tail cannot be trimmed. The GDN state
propagates through the whole stack, so early-layer edits pass through the entire recurrence. The full
layout and why it matters are in [docs/BLOCK-SEARCH.md](docs/BLOCK-SEARCH.md).

## Codebase architecture

The crate is a safe Rust layer over a minimal C shim and prebuilt llama.cpp binaries. The following
table lists the main files.

| Path | Purpose |
|---|---|
| `src/main.rs` | CLI entry point for `serve`, `ask`, and `models`. |
| `src/server.rs` | Axum HTTP server exposing `POST /v1/systemone` and `GET /v1/models`. |
| `src/decision/` | Request deserialization, prompt dispatch, response formatting, and readout calibration. |
| `src/prompt.rs` | One prompt: literal `state` block, letters `W`…`V` (26 labels), media marker. |
| `src/readout.rs` | Space-letter token ids and softmax. |
| `src/engine.rs` | Safe wrapper for model initialization, decoding, and LoRA attachment. |
| `src/ffi.rs` | Foreign function declarations for the C shim. |
| `src/hf.rs` | Hugging Face Hub download helpers and model registry. |
| `src/picture.rs` | Image decoding to raw RGB8 buffers using the `image` crate. |
| `c/src/shim.c` | C shim interfacing with `libllama` and `libmtmd` (LoRA, KV type, prefill reuse). |
| `build.rs` | Builds the static C shim and links dynamic libraries. |

### Dependencies

- **HTTP and async:** `axum`, `tokio`.
- **CLI and serialization:** `clap`, `serde`, `serde_json`.
- **Model and media:** `hf-hub`, `image`, `base64`.
- **Native runtime:** prebuilt Prism `libllama.so`, `libmtmd.so`, `libggml.so`.

## Supported models

BonJev loads GGUF models directly or downloads them through the Hugging Face Hub.

| Canonical id | Repository | Default GGUF |
|---|---|---|
| `ternary-bonsai-2-27b` (default) | `prism-ml/Ternary-Bonsai-2-27B-gguf` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| `ternary-bonsai-27b` | `prism-ml/Ternary-Bonsai-27B-gguf` | `Ternary-Bonsai-27B-PQ2_0.gguf` |
| `ternary-bonsai-8b` | `prism-ml/Ternary-Bonsai-8B-gguf` | `Ternary-Bonsai-8B-PQ2_0.gguf` |
| `ternary-bonsai-4b` | `prism-ml/Ternary-Bonsai-4B-gguf` | `Ternary-Bonsai-4B-PQ2_0.gguf` |
| `ternary-bonsai-1.7b` | `prism-ml/Ternary-Bonsai-1.7B-gguf` | `Ternary-Bonsai-1.7B-PQ2_0.gguf` |

The supported set is the **ternary** line. The 1-bit `bonsai-*` (Q1_0) packs exist upstream but are
not part of the supported set: the decision adapters and the published results cover the ternary line
only. Run `bonjev models` for cache status; legacy CLI aliases still work (`bonsai2`, `8b`, `4b`,
`1.7b`, …).

The vision mmproj applies only to `ternary-bonsai-2-27b`; the other checkpoints are text-only. Pass
`--no-vision` to skip the projector download.

If VRAM is tight, `serve` halves the context (down to 2048 tokens) until the model loads. After a
decode failure it unloads the active GGUF from VRAM and reloads at a smaller context before retrying
the same request once.

## LoRA adapters

Each model has one recommended adapter, built **training-free**: third-party decision LoRA deltas are
recombined by blocks and layers (weighted sums, shared-basis alignment, rank budgets) and written to a
single GGUF adapter. No fine-tuning, no distillation, no data.

Attach an adapter by **short name**: the engine downloads it from Hugging Face through the built-in
client and caches it like a model. `--lora` (repeatable) and `BONJEV_LORA` accept a short name or a
local `.gguf` path; each entry takes an optional `,scale` (λ, task arithmetic) and the value may be a
`;`-separated list. **Up to 8 adapters** are attached at once (additive).

```bash
bonjev loras                                    # list adapters per model
bonjev serve --model ternary-bonsai-4b --lora b10
bonjev serve --model ternary-bonsai-4b --lora b10,1.0 --lora all3_align,0.5   # two adapters
BONJEV_LORA="vega_clef_plumb,1.0;m27_clef_plumb_at_half,0.5" bonjev serve --model ternary-bonsai-2-27b
BONJEV_LORA=/path/to/B10.gguf bonjev serve --model ternary-bonsai-4b          # local file
```

Adapters are tied to their base model; `bonjev loras <model>` lists the variants. Recommended:
`m17_tiny25_single` (1.7B, with `BONJEV_STYLE=top3`), `b10` (4B), `m8_inv_b04` (8B), `vega_clef_plumb`
(27B). Build one with `scripts/build_lora.sh <recipe>`.

The per-model choice and the selection rule are in
[benchmarks/RECOMMENDATIONS.md](benchmarks/RECOMMENDATIONS.md); the pipeline and provenance are in
[docs/RECIPES.md](docs/RECIPES.md) and [docs/LORAS.md](docs/LORAS.md).

## Building

### Quick start

Prerequisites: Rust (2024 edition), a C compiler (`gcc` or `clang`), and `curl`.

```bash
git clone https://github.com/ChezRD/bonjev.git
cd bonjev
cargo build --release
```

`build.rs` runs `./scripts/fetch-prism.sh` to download and unpack the pinned Prism prebuilts
(`prism-b10743-adfffbe`) into `vendor/prism-prebuilt/` if they are not already present. On Linux with
an NVIDIA GPU it auto-detects CUDA; otherwise it falls back to the CPU archive.

### Manual or custom backend

To pre-download a specific backend before building, run one of the following.

```bash
./scripts/fetch-prism.sh               # auto-detects CUDA / CPU
./scripts/fetch-prism.sh cuda-13.3     # Linux x64 CUDA 13.3
./scripts/fetch-prism.sh cuda-12.8     # Linux x64 CUDA 12.8
./scripts/fetch-prism.sh cuda-12.4     # Linux x64 CUDA 12.4
./scripts/fetch-prism.sh vulkan        # Linux x64 Vulkan
./scripts/fetch-prism.sh rocm          # Linux x64 ROCm 7.2
./scripts/fetch-prism.sh cpu           # Linux / macOS CPU
```

Or point `PRISM_LLAMA_DIR` at existing prebuilt libraries.

```bash
PRISM_LLAMA_DIR=/path/to/prism-prebuilt cargo build --release
```

At runtime the binary locates shared libraries using the embedded rpath or `PRISM_LLAMA_DIR`.

The pinned release is [`prism-b10743-adfffbe`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10743-adfffbe)
(CUDA 12.4 / 12.8 / 13.3, Vulkan, ROCm, CPU). Run `./scripts/fetch-prism.sh` after bumping
`prism-rel.txt` to refresh.

## Usage

### Check cached models

```bash
./target/release/bonjev models
```

### Run a one-shot CLI query

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

### Start the HTTP server

```bash
./target/release/bonjev serve --model bonsai2 --port 8080
```

The server binds to `127.0.0.1:8080` and exposes two endpoints:

- `GET /v1/models` returns active model metadata.
- `POST /v1/systemone` evaluates decisions.

#### Request example

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

#### Response example

```json
{
  "model": "ternary-bonsai-2-27b",
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

### CLI flags

`serve` and `ask` share the model flags.

| flag | default | meaning |
|---|---|---|
| `--model <id>` | `bonsai2` | Model id (see [Supported models](#supported-models)). |
| `--ctx <n>` | `32768` | Context window in tokens. |
| `--threads <n>` | `4` | CPU threads. |
| `--ngl <n>` | `99` | Layers to offload to the GPU. |
| `--model-path <file>` | — | Use a local GGUF instead of the HF cache. |
| `--mmproj <file>` | — | Vision projector path (downloaded when omitted). |
| `--no-vision` | off | Do not load a vision projector. |
| `--port <n>` | `8080` | `serve` only. |

`ask` adds `--state`, `--question`, `--option` (repeatable), `--kind` (default `choice`), and `--image`.

## Wire

`POST /v1/systemone` follows the TypeSafe shapes. Send `state` and `questions`; each question is
`choice`, `score`, or `noul`. The response has `model`, `answers` under the same keys, and `usage`.

A choice answer has `type`, `choice`, `probabilities`, and `confidence`. A score answer has `type`,
`score`, `legend`, `probabilities`, and `confidence`. A noul answer has `type` and `noul`.
`type: "boolean"` is accepted and answered as `noul`; a missing `type` is `choice`. Choice needs 2 to
26 options; score needs 2 to 10 levels.

The request `model` field does not choose the GGUF; the process loaded one file at startup. The
response `model` is that checkpoint id, e.g. `ternary-bonsai-2-27b` or `ternary-bonsai-8b` (same as
`bonjev models` / `--model`).

### Prompt and readout

- `state` is a literal block (`state: |`). A string is copied as the raw source; a JSON object or array
  is rendered as YAML text inside that block, not as a nested mapping under `state`.
- Option labels are `W`, `X`, `Y`, then `Z`, `A`, `B`, and so on through `V`.
- The assistant turn ends with the answer lead, once, and nothing after it.
- The decision token is the single token of `lead + space + letter` (` W`, not a bare `W`).
- `prompt_style` (per request) and `BONJEV_STYLE` (process-wide) select the system text and answer
  lead; without either, the engine uses the per-axis default (`native_user` for choice, `answer` for
  yes/no and score). The style names and the per-axis default are in
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

### Errors and images

A validation failure is HTTP 422:

```json
{"detail":[{"loc":["body","questions","decision","criteria"],"msg":"choice needs 2 to 26 options","type":"value_error"}]}
```

Broken JSON is also 422, with `loc` of `["body"]`. An engine failure is 500 and uses the same `detail`
envelope.

To send an image over HTTP, add a base64 string or data URL in the top-level `"image"` field, or embed
it under `"image"` within the `"state"` object.

Several text questions in one request are scored in order on the same `state`; each question gets a
full `--ctx` budget (one llama sequence). Images stay one at a time.

## Configuration

Every environment variable (`BONJEV_*`, `PRISM_LLAMA_DIR`), its default, and its measured effect on the
exam are documented in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). The ones used most: `BONJEV_LORA`
(adapters), `BONJEV_STYLE` (prompt pattern), `BONJEV_KV_TYPE` (K/V cache type), and
`BONJEV_NO_REUSE=1 BONJEV_CKPT=0` (reproducible runs).

## Performance

All measurements use the same testbed: **NVIDIA GeForce RTX 4060 Ti 16 GB**, 16384 context, release
binary. The exam grades a **choice** by the top-probability option, a **yes/no** by `p(true)` vs 0.5,
and a **score** by the top level.

Best measured accuracy (prompt-pattern sweep): the flagship `ternary-bonsai-2-27b` at **195/231**; each
smaller model has its own best pattern. The per-model baselines, per-split numbers, latency, the LoRA
results, and the Decision Index are in [benchmarks/EXAM.md](benchmarks/EXAM.md),
[benchmarks/RESULTS.md](benchmarks/RESULTS.md), and [benchmarks/README.md](benchmarks/README.md).

### Latency notes

- One request is one prefill. On this GPU the engine prefills about 840 tokens/s; per-request time is
  set by the variable part of the prompt (state + question + options), not by the system text.
- The models are hybrid (attention plus recurrent layers), so a KV tail cannot be trimmed. The engine
  restores a saved prefix checkpoint per request (~17 ms) instead of re-prefilling the shared 41-token
  prefix. Up to four checkpoints are kept, one per pattern family.
- The exam p50 is set by the short `easy`/`original` tasks; p95 by the long `hard` documents (up to
  about 4k prompt tokens).

## Limitations

- One request may list many text questions. They run one after another on the single sequence, each
  with the full context window. Images in a request are scored one at a time.
- Concurrent HTTP requests are answered one after another; there is no multi-sequence batch. A failed
  picture does not shrink the context. A load that does not fit in VRAM, including the vision
  projector, retries at half the context down to 4096, and that smaller context stays for the process.
- The 27B is a hybrid stack, so the KV cache covers only the 16 full-attention blocks and the GDN state
  cannot be partially trimmed (see [Model architecture](#model-architecture)).
- Model execution is bound to the llama.cpp backend plugins available in `PRISM_LLAMA_DIR`.

## Documentation

| document | what it is |
|---|---|
| [docs/README.md](docs/README.md) | methodology index |
| [docs/LORAS.md](docs/LORAS.md) | adapter inventory, merge theory, provenance |
| [docs/RECIPES.md](docs/RECIPES.md) | how each final LoRA is built and reproduced |
| [docs/BLOCK-SEARCH.md](docs/BLOCK-SEARCH.md) | block/layer study and the 27B hybrid layout |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | every `BONJEV_*` variable and its measured effect |
| [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) | third-party models used for the adapters, with licenses |
| [benchmarks/README.md](benchmarks/README.md) | how the published numbers are produced |
| [benchmarks/RECOMMENDATIONS.md](benchmarks/RECOMMENDATIONS.md) | which LoRA to use per model |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | consolidated 231 / DI / latency tables |

## License and attribution

- **BonJev:** MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **LoRA adapters:** Apache-2.0 (derivatives of Apache-2.0 source models); see
  [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).
- **llama.cpp / mtmd / ggml:** MIT License (Copyright (c) 2023–2026 The ggml authors).
- **Model weights:** Ternary-Bonsai and Bonsai models are licensed under Apache 2.0 (derived from
  Qwen/Qwen3.8-27B). Weight downloads are subject to their respective model cards.
- **Trademarks:** Jev, System One, and TypeSafe are trademarks of their respective owners. BonJev is an
  independent project.
