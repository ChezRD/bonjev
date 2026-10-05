# Training-free hooks from `strands-decider`

What can be borrowed from the external project `StrandsAgents/strands-decider-2B-hobson-v19` (a LoRA on
`Qwen/Qwen3.5-2B-Base` plus a pointer head; **167/231** on JevBench) **without training**. Our project
is strictly training-free.

**Contents**

- [Glossary](#glossary)
- [What can be hooked](#what-can-be-hooked)
- [Per-axis temperature: measurements](#per-axis-temperature-measurements)
- [How to enable per-axis temperature](#how-to-enable-per-axis-temperature)
- [What else can be tested](#what-else-can-be-tested)
- [Bottom line](#bottom-line)

## Glossary

- **LoRA** — a low-rank weight add-on (`B·A`) applied on top of a frozen model.
- **Pointer head** — a small trained head (theirs: `q`, `k` of size `256×2048` and a bilinear score
  `q·k` over options) that turns the model's hidden state into option probabilities.
- **Readout** — how we get the answer. Ours is a **single-token letter argmax**: the model picks the
  option's letter, the engine takes the max logit over letters. Fast.
- **Calibration** — whether a model's confidence matches its real accuracy.
  - **ECE** (expected calibration error) — the average gap between confidence and accuracy; lower is
    better.
  - **Brier** — a quadratic score over probabilities; lower is better.
- **Temperature (T)** — divide logits by a number before softmax. **Argmax is unchanged** (same
  answer); only the confidence changes. `T>1` smooths (fixes over-confidence).
- **Per-axis** — a separate temperature per question kind: `choice` (pick 1 of N), `noul` (yes/no),
  `score` (an ordered level).

## What can be hooked

| hook | what it is | code status | affects accuracy? |
|---|---|---|---|
| **E1. Per-axis temperature** | a separate `T` for `choice` / `noul` / `score` | **already implemented** (`src/decision/calib.rs`: `per_axis` from `BONJEV_BIAS`; `src/decision/pack.rs`) — a **config, not code** | no (argmax-neutral); calibration only |
| **E2. Ordinal confidence for `score`** | post-processing `1−σ/σmax` | a small readout field / formula | no |
| **E3 / E4. strands prompt styles** | numbered options + descriptions, noul false/true | edit `src/prompt.rs` (`STYLES`) + **rebuild** | **yes** (the only accuracy hook) |
| **E6. Permutation averaging** | N option permutations + averaging | engine / harness change | possibly (robustness) |
| **E5. Their eval sets** | MuSiQue / ContractNLI / hotpotqa | diagnostics, no code | no |
| pointer head | a trained head | **incompatible**: hidden 2048 only at 1.7B, but trained on Qwen3.5-2B hidden states; our engine does not compute `h(option)` | — |
| their LoRA | targets are **GDN modules** (`in_proj_qkv/z/a/b/out_proj`) | **does not transfer**: qwen3 (1.7/4/8B) has no such modules; 27B hidden 5120 ≠ 2048 | — |
| transplant / `jf_embeddings` score | — | failed 4× / not needed | — |

## Per-axis temperature: measurements

The script takes the **stored probabilities** (231 dumps and DI `results.jsonl`), fits `T` by NLL on a
CV half, and recomputes Brier / ECE. **Argmax is unchanged → 231 (accuracy) is unaffected.**

231 (excerpt; ECE before→after):

| run | choice | noul | score |
|---|---|---|---|
| 27B `vega_clef_plumb` | .085→.067 | .098→.107 ✗ | .093→.093 |
| 4B base | .141→.063 | .254→.080 | .231→.202 |
| 8B base | .123→.087 | .231→.056 | .203→.178 |
| 1.7B | .072→.104 ✗ | .287→.036 | .182→.198 ✗ |

DI sample-300 (choice): 4B base `.120→.048`, `B10` `.132→.083`, champ125 `.180→.091`, λ0.75
`.131→.083`; 8B base `.114→.141` ✗.

**Conclusion:** `T` depends on the model and the dataset, so fit your own by CV (their values choice
.734 / noul .911 / score 1.328 do not transfer). For `choice` and small-model `noul` the calibration is
fixed; 27B `noul` is already calibrated and degrades at `T>3`.

## How to enable per-axis temperature

```bash
# per-axis.json (values are an example for 4B on 231; fit your own by CV)
{ "per_axis": { "choice": 2.20, "noul": 11.5, "score": 2.23 } }

BONJEV_BIAS=/path/per-axis.json BONJEV_LORA=<...> ./target/release/bonjev serve ...
```

Caveats:

- `T` **depends on the model and the set**: a fit on 231 need not calibrate DI (and vice versa) —
  measure separately per benchmark.
- Do not enable an axis if it hurts (27B `noul`, 1.7B `choice` / `score`).
- Temperature **does not change the answers** — only the confidence.

## What else can be tested

| # | check | how | cost |
|---|---|---|---|
| 1 | **E1 live**: apply a `per_axis` JSON via `BONJEV_BIAS` and re-score DI with the kit | config + DI run | GPU |
| 2 | **E2**: ordinal confidence for `score` (231 has 18) | readout edit | rebuild |
| 3 | **E3 / E4**: strands prompt styles (accuracy) | `prompt.rs` edit + rebuild | rebuild + GPU |
| 4 | **E6**: permutation averaging | harness | GPU |
| 5 | **E5**: their eval sets as diagnostics | download / run | CPU/GPU |
| 6 | head / LoRA compatibility | already checked: incompatible | — |

## Bottom line

- **Training and their artifacts need no code** — the training-free path stays as is.
- **Without a rebuild**, only per-axis temperature is available (via `BONJEV_BIAS`); it moves
  **calibration**, not decisions.
- **Accuracy** can only be moved by the prompt styles — a `src/prompt.rs` edit plus a rebuild.
