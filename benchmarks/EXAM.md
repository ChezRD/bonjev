# Exam report: patterns, ensembles, and error classes

How the JevBench-231 exam is scored, which serving styles and ensembles were searched, and how the
misses classify. The consolidated results are in [RESULTS.md](RESULTS.md); the per-model choice is in
[RECOMMENDATIONS.md](RECOMMENDATIONS.md).

**Contents**

- [Setup](#setup)
- [Models](#models)
- [Baselines](#baselines)
- [Style and ensemble search](#style-and-ensemble-search)
- [Error classes](#error-classes)
- [Flights](#flights)
- [Invariants per run and flag](#invariants-per-run-and-flag)
- [Notes](#notes)

## Setup

Every run scores the pinned public JevBench exam (231 tasks: easy 48, original 72, hard 111, commit
`3749b4fc`) through `POST /v1/systemone` on one `target/release/bonjev` server at a time, context
16384, `--no-vision`. Hardware: RTX 4060 Ti 16 GB (shared with other sessions). Run artifacts are kept
outside the repository.

Grade: argmax for choice and score, `p(true) >= 0.5` for yes/no (the JevBench rule).

## Models

The supported set is the ternary line.

| id | file | arch | blocks | GiB |
|---|---|---:|---:|---:|
| `ternary-bonsai-2-27b` | Ternary-Bonsai-2-27B-PQ2_0 | qwen35 | 64 | 6.71 |
| `ternary-bonsai-2-27b-ptq1` | Ternary-Bonsai-2-27B-PTQ1_0 | qwen35 | 64 | 5.54 |
| `ternary-bonsai-27b` | Ternary-Bonsai-27B-PQ2_0 | qwen35 | 64 | 6.67 |
| `ternary-bonsai-8b` | Ternary-Bonsai-8B-PQ2_0 | qwen3 | 36 | 2.03 |
| `ternary-bonsai-4b` | Ternary-Bonsai-4B-PQ2_0 | qwen3 | 36 | 1.00 |
| `ternary-bonsai-4b-q2g64` | Ternary-Bonsai-4B-Q2_0_g64 | qwen3 | 36 | 1.06 |
| `ternary-bonsai-1.7b` | Ternary-Bonsai-1.7B-PQ2_0 | qwen3 | 28 | 0.43 |

## Baselines

Default per-axis: choice → `native_user`, yes/no and score → `answer`.

| model | total | easy | original | hard | median | p95 |
|---|---:|---:|---:|---:|---:|---:|
| `ternary-bonsai-2-27b` | 195 | 48/48 | 69/72 | 78/111 | 0.161 s | 3.17 s |
| `ternary-bonsai-2-27b-ptq1` | 195 | 48/48 | 69/72 | 78/111 | 0.157 s | 3.18 s |
| `ternary-bonsai-27b` | 177 | 48/48 | 61/72 | 68/111 | 0.147 s | 2.94 s |
| `ternary-bonsai-8b` | 150 | 47/48 | 55/72 | 48/111 | 0.039 s | 0.82 s |
| `ternary-bonsai-4b` | 148 | 48/48 | 53/72 | 47/111 | 0.027 s | 0.53 s |
| `ternary-bonsai-4b-q2g64` | 150 | 48/48 | 54/72 | 48/111 | 0.030 s | 0.51 s |
| `ternary-bonsai-1.7b` | 124 | 45/48 | 37/72 | 42/111 | 0.016 s | 0.23 s |

### Cross-line: the 27B adapters on the v1 model

The final 27B adapters (built for `ternary-bonsai-2-27b`, Qwen3.8) also load on the v1
`ternary-bonsai-27b` (Qwen3.6) — same `qwen35` architecture. Deterministic, default-style:

| model | LoRA | 231 | easy | original | hard | answer_first |
|---|---|---:|---:|---:|---:|---:|
| `ternary-bonsai-27b` | — | 176 | 48/48 | 63/72 | 65/111 | 10.4% |
| `ternary-bonsai-27b` | `vega_clef_plumb` | 187 | 48/48 | 70/72 | 69/111 | 91.3% |

The adapter adds **+11** (original +7, hard +4) and lifts `answer_first` from 10.4% to 91.3%. This is a
cross-base application (a Qwen3.8 adapter on a Qwen3.6 base) within the same architecture; it is a
check, not part of the published set. The published 27B result is `ternary-bonsai-2-27b` (194 → 207).

## Style and ensemble search

Each sweep scores 12–15 styles in one server session; the probability dumps are then replayed offline
for average / vote / weighted fusion of every subset of one to three styles, plus a gated fallback.

| model | best single (real) | best ensemble (simulated → confirmed) | oracle |
|---|---|---|---|
| `ternary-bonsai-2-27b` | `answer` 194, `native_user_answer` 194, `strict2_answer` 194 | avg `native_user_answer+strict2_answer` 195 → **194**; per-axis default **195** (real) | 207 |
| `ternary-bonsai-27b` | `answer` **179**, `w2_word3_correct` 177, `strict2_code` 176 | avg `answer+strict_answer+strict2_code` 180 → **175** | 195 |
| `ternary-bonsai-8b` | `answer`/`w2_word3_correct`/`strict2_answer` 155 | avg `answer+native_user+strict2_answer` 158 → **155** | 189 |
| `ternary-bonsai-4b` | `w3_chosen` 159 | vote `answer+w3_chosen+native_option` 163 → **158** | 188 |
| `ternary-bonsai-1.7b` | `w2_strict2_desc` 138 | avg `w2_strict2_desc+strict2_answer+strict2_code` 143 → **142** | 198 |

Reading the table:

- The offline simulation merges per-style probabilities from separate runs; the real ensemble decodes
  the styles together and near-ties flip (4–9% of per-style picks on the small models).
- Confirmed ensembles therefore score 1–9 points below the simulation.
- The best single style beats the confirmed ensemble on `ternary-bonsai-4b` and `ternary-bonsai-27b`,
  ties it on `ternary-bonsai-8b` and `ternary-bonsai-2-27b`, and only `ternary-bonsai-1.7b` improves
  on its best single.

**Packing variants.**

- `ternary-bonsai-2-27b-ptq1` (PTQ1_0) matches PQ2_0 exactly — 195, same splits, 1.2 GiB smaller.
- `ternary-bonsai-4b-q2g64` (`Q2_0` group-64) peaks at 158 (`answer` / `w2_word3_desc` /
  `native_option`) where PQ2_0 peaks at 159 (`w3_chosen`); its simulated best vote trio
  (`answer+w2_word3_correct+native_option`, 164) confirmed at **156** (avg 154), so the single style
  still wins.

## Error classes

`BONJEV_TOKEN_DUMP=1` prints the top raw tokens and per-label slot logits for every pass; the misses
are classified as follows.

- **READOUT:** the packed answer never differs from the argmax of the slot logits. 0/231 on every
  model.
- **PROMPT:** the expected option is top-1 under at least one other style (the lead / pattern can
  extract it).
- **MODEL:** no style extracts it.

| model (config) | errors | READOUT | echo | rank-2 | rank-3 | 4+ | margin | PROMPT | MODEL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ternary-bonsai-2-27b` (default, 195) | 36 | 0 | 36 | 11 | 9 | 16 | 0.234 | 12 | 24 |
| `ternary-bonsai-2-27b` (pair avg, 194) | 37 | 0 | 37 | 14 | 7 | 16 | 0.226 | 13 | 24 |
| `ternary-bonsai-27b` (triple avg, 175) | 56 | 0 | 56 | 21 | 5 | 30 | 0.216 | 20 | 36 |
| `ternary-bonsai-8b` (trio avg, 155) | 76 | 0 | 2 | 22 | 19 | 35 | 0.514 | 34 | 42 |
| `ternary-bonsai-4b` (vote trio, 158) | 73 | 0 | 3 | 20 | 15 | 36 | 0.482 | 30 | 43 |
| `ternary-bonsai-1.7b` (avg trio, 142) | 89 | 0 | 28 | 23 | 17 | 49 | 0.238 | 56 | 33 |
| `ternary-bonsai-4b-q2g64` (vote trio, 156) | 75 | 0 | 6 | 18 | 18 | 38 | 0.477 | 30 | 45 |

On both 27B models every miss has a special token as the raw top-1 (an empty piece — the model wants
to close its think block), so the dump analyzer counts it as echo; the label slots still rank normally
and the readout stays intact. `ternary-bonsai-1.7b` is the only small model that echoes the format on
many misses (28): it loses the single-letter contract most often.

## Flights

Jev Flight Lab, own demo instance on 8011; demo → `bonjev` on 8849. "Complete" is the demo's own
flight completion: decisions, elapsed, distance, interventions, median decision latency.

| model | config | normal | barrier |
|---|---|---|---|
| `ternary-bonsai-2-27b` | `default` | complete: 21 dec, 128.2 s, 57.7 m, 0, 5.00 s | complete: 23 dec, 136.5 s, 63.1 m, 0, 4.87 s |
| `ternary-bonsai-2-27b-ptq1` | `default` | complete: 21 dec, 122.5 s, 57.7 m, 0, 4.78 s | complete: 23 dec, 130.2 s, 62.5 m, 0, 4.58 s |
| `ternary-bonsai-27b` | `answer` | complete: 22 dec, 116.9 s, 57.7 m, 1, 4.23 s | failed: six consecutive holds |
| `ternary-bonsai-8b` | `answer` | complete: 22 dec, 36.9 s, 58.5 m, 0, 0.61 s | complete: 23 dec, 39.6 s, 60.5 m, 1, 0.61 s |
| `ternary-bonsai-4b` | `w3_chosen` | failed: six consecutive holds | failed: six consecutive holds |
| `ternary-bonsai-4b-q2g64` | `w3_chosen` | pending | pending |
| `ternary-bonsai-1.7b` | avg trio | failed: six consecutive holds | failed: six consecutive holds |

"Six consecutive holds" is the demo's safety pause after six `hold` decisions in a row — model
behavior on the flight prompts, not an engine error. The barrier flight inserts an obstacle before
start; the 27B models clear it, the small ones hold.

With the per-axis default (no `BONJEV_STYLE`), `ternary-bonsai-27b` completes both flights (21 dec /
116.8 s / 0 int. / 4.52 s and 25 dec / 136.5 s / 1 int. / 4.40 s) and `ternary-bonsai-4b` completes
both (27 dec / 50.0 s / 6 int. / 0.74 s and 23 dec / 41.8 s / 1 int. / 0.75 s) — so the exam-best
style, not the model, was pausing those two; `ternary-bonsai-4b-q2g64` behaves the same (23 dec /
42.6 s / 1 int. / 0.75 s and 24 dec / 44.0 s / 1 int. / 0.76 s). `ternary-bonsai-1.7b` holds under
every style tried.

## Invariants per run and flag

Every scored pass is checked for the prompt / readout invariants: the answer lead adds tokens and stays
intact at the end, every letter is exactly one token after the lead, and no two letters share a token
id. A failing row stays in the JSONL with `ok: false` and an `invariant` name; it never aborts the
sweep.

| run | flags | passes | intact | failures |
|---|---|---:|---:|---|
| all single-style sweeps (12–15 styles each) | `--ctx 16384` (sweep), `BONJEV_TOKEN_DUMP` off | 2772–3465 | 100% | none |
| confirmations (2–3 style ensembles) | `BONJEV_ENSEMBLE_MODE=avg\|vote\|weighted`, `BONJEV_TOKEN_DUMP=1` | 231 each | 231/231 | none |
| `top3` on `ternary-bonsai-4b` at 8k context | `--ctx 8192` | 231 | 194/231 | `other` 37: `decode failed rc=1` — the three prompts do not fit the shared 8k KV |
| `top3` on `ternary-bonsai-4b` at 16k context | `--ctx 16384` | 231 | 231/231 | none |
| `ternary-bonsai-2-27b-ptq1` `native_user_answer`, ada-only fork `041078f` | `--ctx 16384` | 231 | 231/231 | none |
| same, with LoRA `r64-q8` | `BONJEV_LORA=clef_lora_r64_q8.gguf` | 231 | 231/231 | none |
| same, with LoRA `r128-q8` | `BONJEV_LORA=clef_lora_r128_q8.gguf` | 231 | 231/231 | none |
| `ternary-bonsai-2-27b-ptq1` `r128-q8` 15-style sweep | `BONJEV_LORA=clef_lora_r128_q8.gguf`, `--ctx 16384` | 3465 | 3465/3465 | none |
| `ternary-bonsai-27b` 15-style sweep + `r64-q8` | `BONJEV_LORA=clef_lora_r64_q8.gguf` | 3465 | 3465/3465 | none |
| `ternary-bonsai-27b` flags + `r64-q8` (`kv_q8_0`, `kv_f16`) | `BONJEV_KV_TYPE=q8_0` / `=f16` | 462 | 462/462 | none |
| `ternary-bonsai-2-27b-ptq1` flag matrix (8 engine flags) | `BONJEV_KV_TYPE`, `BONJEV_BATCH`, `BONJEV_THINK_TAGS`, `BONJEV_ENSEMBLE_MODE` | 1848 | 1848/1848 | none |
| `ternary-bonsai-2-27b` (PQ2_0) flag matrix (8 flags) | same flags | 1848 | 1848/1848 | none |
| every run above, by invariant name | — | — | — | `lead_missing` 0, `lead_merged` 0, `lead_absent` 0, `letter_split` 0, `shared_id` 0, `empty_lead` 0, `unknown_style` 0 |

No flag (`BONJEV_THINK_TAGS`, `BONJEV_ENSEMBLE_MODE`, `BONJEV_STYLE`, `BONJEV_BATCH`, `BONJEV_KV_TYPE`,
context size) produced a prompt / readout invariant failure. The only invariant failures ever observed
come from an ensemble that exceeds the shared KV window, not from the readout.

## Notes

- `BONJEV_ENSEMBLE_MODE=avg|vote|weighted` merges the passes of one question; `BONJEV_STYLE` accepts
  one name, `top3`, or a comma list of up to three.
- The per-axis default is the flagship's optimum and hurts some models: `ternary-bonsai-4b` 148 vs its
  best confirmed 159, `ternary-bonsai-27b` 177 vs 179. Set `BONJEV_STYLE` (or `prompt_style`) per
  model.
- `PTQ1_0` and `PQ2_0` pack the same ternary weights: both score 195 with identical splits on
  `ternary-bonsai-2-27b`, and `PTQ1_0` is 1.2 GiB smaller.
- `BONJEV_THINK_TAGS=1` (closed empty think block) adds +2–3 on `ternary-bonsai-2-27b`.
- `BONJEV_BATCH` default 1024 measured fastest; `GGML_CUDA_FORCE_MMQ`, `CUBLAS_COMPUTE_TYPE=fp16`,
  CUDA graphs and fusion do not move the exam.

## See also

- [RESULTS.md](RESULTS.md) — consolidated 231 / DI / latency tables.
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) — which LoRA to use per model and the selection rule.
- [FLIGHT.md](FLIGHT.md), [DIRECT-INFER.md](DIRECT-INFER.md) — control tests.
