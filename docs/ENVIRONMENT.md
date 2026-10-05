# Environment variables

Every `BONJEV_*` variable the engine reads, what it does, and its **measured effect on JevBench-231
relative to the base model**. Numbers are JevBench-231; "deterministic" means
`BONJEV_NO_REUSE=1 BONJEV_CKPT=0`. The base per model depends on the run mode: the style-sweep
baselines are 195 (27B) / 150 (8B) / 148 (4B) / 124 (1.7B); the deterministic bases are 194 / 149 /
145 / 124.

The variables are read in the Rust engine (`src/decision/`) and in the C shim (`c/src/shim.c`).

**Contents**

- [Summary](#summary)
- [BONJEV_LORA](#bonjev_lora)
- [BONJEV_STYLE](#bonjev_style)
- [BONJEV_ENSEMBLE_MODE](#bonjev_ensemble_mode)
- [BONJEV_KV_TYPE](#bonjev_kv_type)
- [BONJEV_THINK_TAGS](#bonjev_think_tags)
- [BONJEV_BATCH](#bonjev_batch)
- [BONJEV_TEMP](#bonjev_temp)
- [BONJEV_BIAS](#bonjev_bias)
- [BONJEV_CONTEXT_CALIB](#bonjev_context_calib)
- [BONJEV_NO_REUSE and BONJEV_CKPT](#bonjev_no_reuse-and-bonjev_ckpt)
- [BONJEV_TIMING and BONJEV_TOKEN_DUMP](#bonjev_timing-and-bonjev_token_dump)
- [PRISM_LLAMA_DIR](#prism_llama_dir)
- [See also](#see-also)

## Summary

| variable | default | purpose | effect on 231 vs base |
|---|---|---|---|
| `BONJEV_LORA` | off | attach adapter(s) | the LoRA results (4B 145 → 161, 27B 194 → 207) |
| `BONJEV_LORA_NAMESPACE` | `ChezRD` | Hugging Face namespace for the adapter repos | — |
| `BONJEV_STYLE` | per-axis | prompt pattern | up to +13 (4B 148 → 159); the per-axis default is the flagship's optimum |
| `BONJEV_ENSEMBLE_MODE` | `avg` | merge the passes of one question | with `top3` on 27B: −4…−6 |
| `BONJEV_KV_TYPE` | `q4_0` | K/V cache type | 27B +1; Q2_K clef +1; neutral elsewhere |
| `BONJEV_THINK_TAGS` | off | closed empty think block | 27B +2; 9B base −3; 9B clef −6 |
| `BONJEV_BATCH` | `1024` | prefill chunk / ubatch | 0 |
| `BONJEV_TEMP` | `1.0` | global temperature | 0 (argmax-neutral); calibration only |
| `BONJEV_BIAS` | off | per-slot bias + per-axis temperature | bias can move the argmax; per-axis T is argmax-neutral |
| `BONJEV_CONTEXT_CALIB` | off | subtract a content-free prior | not measured |
| `BONJEV_NO_REUSE` | off | full prefill (no prefix reuse) | determinism; 4B 148 → 145, 8B 150 → 149, 27B 195 → 194 |
| `BONJEV_CKPT` | on | `0` disables the prefix-restore path | used with `NO_REUSE`; no direct 231 effect |
| `BONJEV_TIMING` | off | per-stage decode timings | 0 |
| `BONJEV_TOKEN_DUMP` | off | one JSON line per pass | 0 (diagnostics) |
| `PRISM_LLAMA_DIR` | `prism-rel.txt` | directory with `libllama.so` / `libmtmd.so` | 0 |

## BONJEV_LORA

**Purpose:** attach N LoRA adapters. Each entry is a **short name** (see `bonjev loras`) or a local
`.gguf` path, with an optional `,scale` (default 1, amplifies the delta via task arithmetic). Entries
are separated by `;`, e.g. `b10,1.25` or `/path/B10.gguf`. Short names are resolved by the Rust engine
and downloaded through the built-in Hugging Face client (cached like a model); the resolved path list
is then read by the C shim (`c/src/shim.c`).

- `--lora` is **repeatable** and may also hold a `;`-separated list; it overrides `BONJEV_LORA`.
- Multiple adapters are attached at once and are **additive**; up to **8** (`JF_LORA_MAX`).
- Adapters are tied to their base model: `bonjev loras <model>` lists the variants.
- `BONJEV_LORA_NAMESPACE` (default `ChezRD`) changes the repository namespace.

**Effect:** the whole point of the release — one adapter per model. Deterministic 231:

| model | base | adapter | 231 |
|---|---|---:|---:|
| 1.7B | 124 | `m17_tiny25_single` (+ `BONJEV_STYLE=top3`) | 143 (147) |
| 4B | 145 | `B10` | **161** |
| 8B | 149 | `m8_inv_b04` (default is base) | 159 |
| 27B | 194 | `vega_clef_plumb` | **207** |

The same 27B adapters also load on the v1 `ternary-bonsai-27b` (same `qwen35` architecture): 176 → 187
(+11). See [../benchmarks/EXAM.md](../benchmarks/EXAM.md).

## BONJEV_STYLE

**Purpose:** force prompt pattern(s) for every request, overriding the per-axis default (`native_user`
for choice, `answer` for yes/no and score). One name, `top3`, or a comma list of up to three. Read in
`src/prompt.rs`.

**Effect:** the largest non-LoRA lever. Default (per-axis) vs best measured pattern (style sweep,
non-deterministic baselines):

| model | default | best pattern | Δ |
|---|---:|---:|---:|
| 27B (2-27b) | 195 | 195 (`default`) | 0 |
| 27B (v1) | 177 | 179 (`answer`) | +2 |
| 27B (1-bit) | 151 | 159 (`answer`) | +8 |
| 8B | 150 | 155 (`answer`) | +5 |
| 8B (1-bit) | 146 | 152 (`w3_chosen`) | +6 |
| 4B | 148 | 159 (`w3_chosen`) | +11 |
| 4B (1-bit) | 145 | 153 (`native_user_theanswer`) | +8 |
| 1.7B | 124 | 138 (`w2_strict2_desc`); 142 as an ensemble | +14 / +18 |
| 1.7B (1-bit) | 115 | 127 (`w3_chosen`) | +12 |

The per-axis default is the flagship's optimum; every other model has its own best pattern, so set
`BONJEV_STYLE` per model.

## BONJEV_ENSEMBLE_MODE

**Purpose:** how the passes of one question are merged when more than one pass runs (a `top3` or
comma-list style): `avg` (mean probabilities), `vote` (majority of each pass's argmax; ties fall back
to the mean), `weighted` (mean weighted by each pass's top1−top2 margin). Read in
`src/decision/pack.rs`.

**Effect:** on 27B, `top3` ensembles lose to the single style; the merge mode barely matters.

| run (base 195) | avg | vote | weighted |
|---|---:|---:|---:|
| `ternary-bonsai-2-27b` `top3` | 189 | 190 | 189 |
| `ternary-bonsai-2-27b-ptq1` `top3` | 190 | 191 | 191 |

Confirmed real ensembles: they beat the best single style only on 1.7B, tie it on 27B / 8B, and lose on
the rest. The offline combination estimate is optimistic (near-ties flip in a real batched decode).

## BONJEV_KV_TYPE

**Purpose:** K/V cache type: `q4_0` (default), `q8_0`, or `f16`. Read in `c/src/shim.c`.

**Effect:** small and positive on 27B, neutral elsewhere.

| run | base | `q8_0` | `f16` |
|---|---:|---:|---:|
| `ternary-bonsai-2-27b` | 195 | 196 | 196 |
| `ternary-bonsai-2-27b-ptq1` | 195 | 196 | 196 |
| 9B Q2_K clef | 83 | 84 | — |
| `ternary-bonsai-27b` + `r64-q8` | 177 | 177 | 178 |

## BONJEV_THINK_TAGS

**Purpose:** `1` inserts one closed empty ` thinking` block before the answer lead. Read in
`src/prompt.rs`.

**Effect:** helps the 27B, hurts the 9B and the small adapters.

| run | base | `THINK_TAGS=1` |
|---|---:|---:|
| `ternary-bonsai-2-27b` / `-ptq1` | 195 | **197** (+2) |
| 9B PTQ1_0 base | 74 | 71 (−3) |
| 9B PTQ1_0 clef-flash | 75 | 69 (−6) |
| 9B PTQ1_0 openjev | 69 | 69 (0) |
| 9B PTQ1_0 merged | 69 | 74 (+5) |
| Qwen3.5-4B PTQ1_0 base | 75 | 74 (−1) |
| Qwen3.5-4B PTQ1_0 Kahn1 | 72 | 69 (−3) |
| 9B Q2_K clef-flash | 83 | 72 (−11) |

## BONJEV_BATCH

**Purpose:** prefill chunk and ubatch size, 32 to 8192. Read in `c/src/shim.c`.

**Effect:** 0 on the score; 1024 is the fastest.

| run | 512 | 1024 | 4096 |
|---|---:|---:|---:|
| `ternary-bonsai-2-27b` | 195 | 195 | 195 |
| `ternary-bonsai-2-27b-ptq1` | 195 | 195 | 195 |

## BONJEV_TEMP

**Purpose:** global temperature: `softmax(vals / temp)`. Read in `src/decision/calib.rs`. **Argmax is
unchanged**, so the 231 score does not move; only the probabilities (calibration) change.

**Effect on 231:** 0. `m17_tiny25_single` stays **143** with `BONJEV_TEMP=0.5`.

**Effect on calibration** (ECE before → after; a fitted per-axis temperature, see
[TRAINING-FREE-HOOKS.md](TRAINING-FREE-HOOKS.md)):

| run | choice | noul | score |
|---|---|---|---|
| 27B `vega_clef_plumb` | .085 → .067 | .098 → .107 ✗ | .093 → .093 |
| 4B base | .141 → .063 | .254 → .080 | .231 → .202 |
| 8B base | .123 → .087 | .231 → .056 | .203 → .178 |
| 1.7B | .072 → .104 ✗ | .287 → .036 | .182 → .198 ✗ |

The fitted temperature depends on the model and the dataset, so fit your own by cross-validation. For
`choice` and small-model `noul` it fixes the calibration; 27B `noul` is already calibrated and degrades.

## BONJEV_BIAS

**Purpose:** path to a JSON calibration file: a bare `[b0, b1, …]` array, or
`{"bias": […], "temp": 1.2, "per_axis": {"choice": 1.1, "noul": 1.2, "score": 1.3}}`. Read in
`src/decision/calib.rs`.

- `bias` is a per-slot logit bias; it **can move the argmax** and is the lever that changes the 231
  score. It needs a calibration split to fit honestly (our `dev-probe` was a subset of 231, so it is
  not a holdout).
- `temp` / `per_axis` apply a per-axis temperature: **argmax-neutral**, calibration only.

**Effect on 231:** not measured for a fitted `bias` (no holdout). Per-axis temperature is config-only
and does not change the score.

## BONJEV_CONTEXT_CALIB

**Purpose:** `1` enables contextual calibration: the readout subtracts a content-free (null-prompt)
prior from the slot logits (`src/readout.rs`).

**Effect on 231:** not measured.

## BONJEV_NO_REUSE and BONJEV_CKPT

**Purpose:** reproducibility. By default the engine reuses the llama sequence cache and a saved prefix
checkpoint between requests. `BONJEV_NO_REUSE=1` forces a full prefill; `BONJEV_CKPT=0` disables the
prefix-restore path. Read in `c/src/shim.c`.

**Effect:** the canonical, reproducible mode. It removes the cross-request state sensitivity (a single
prior unrelated request dropped the 1.7B exam from 143 to 136) and slightly lowers the base numbers:

| model | reuse | deterministic |
|---|---:|---:|
| 1.7B | 124 | 124 |
| 4B | 148 | **145** |
| 8B | 150 | **149** |
| 27B | 195 | **194** |

For the hybrid 27B the reuse path also had a real bug (stale GDN state); `NO_REUSE` covers the
multi-sequence path too. Use both flags for any reported number.

## BONJEV_TIMING and BONJEV_TOKEN_DUMP

**Purpose:** diagnostics. `BONJEV_TIMING=1` prints per-stage decode timings to stderr;
`BONJEV_TOKEN_DUMP=1` prints one JSON line per scored pass (top raw tokens and per-label slot logits).

**Effect on 231:** 0. They do not change the decision; they only add output.

## PRISM_LLAMA_DIR

**Purpose:** directory with the prebuilt Prism `libllama.so` / `libmtmd.so`. Read by `build.rs` and at
runtime.

**Effect on 231:** 0 (the same libraries are used either way).

## See also

- [RECIPES.md](RECIPES.md) — how the adapters are built.
- [TRAINING-FREE-HOOKS.md](TRAINING-FREE-HOOKS.md) — the per-axis temperature experiment.
- [../benchmarks/EXAM.md](../benchmarks/EXAM.md) — the flag matrix and the cross-line checks.
- [../benchmarks/RESULTS.md](../benchmarks/RESULTS.md) — consolidated 231 / DI tables.
