# Results: consolidated 231 and Decision Index

All project measurements in one place: the JevBench-231 exam, the Decision Index, latency, and the
public Clef comparison.

**Contents**

- [Definitions](#definitions)
- [JevBench-231 (deterministic)](#jevbench-231-deterministic)
- [Decision Index (DI)](#decision-index-di)
- [Latency](#latency)
- [Public comparison: `Cloudflare/clef`](#public-comparison-cloudflareclef)
- [Per-track matrix](#per-track-matrix)
- [See also](#see-also)

## Definitions

| term | meaning |
|---|---|
| **231** | JevBench, 231 tasks, **deterministic** (`BONJEV_NO_REUSE=1`). Accuracy. |
| **DI** | Decision Index 0.2.1 (the external `decision-index` kit). The arbiter is **`A_native`**: the equal-weight mean of each track's **native metric** (`scores.json.score`), with ACOS on its own line. |
| **Ship rule** | Δ231 ≥ +3 ∧ `A_native` not worse than base ∧ fast-forward (`answer_first ≥ 99%`). |

Plain accuracy and `balanced_skill` are **not** used: ACOS is ≈99% one class, and the official index is
coverage-adjusted and ≈0 on a sample.

## JevBench-231 (deterministic)

| model | LoRA | λ | style | 231 | note |
|---|---|---|---:|---:|---|
| 1.7B | base | — | — | 124 | |
| 1.7B | `m17_tiny25_single` | baked 2.5 | default | 143 | |
| **1.7B** | **`m17_tiny25_single`** | 2.5 | **top3** | **147** | **best** |
| 4B | base | — | — | 145 | |
| **4B** | **`B10`** | **1.0** | default | **161** | **best** |
| 4B | `B10` | 1.0 | top3 | 159 | |
| 4B | `B10` | 1.25 | default / top3 | 158 / 157 | |
| 4B | `B10` | 0.75 | default | 150 | λ<1 regression |
| 4B | `m4b_all3_align` (alt) | 1.0 / 1.25 | default | 159 / **163** | 231 max, but DI− |
| 4B | `m4b_all3_align` | 1.0 | top3 | 160 | |
| 8B | base | — | default / top3 | 149 / 150 | |
| **8B** | **`m8_inv_b04`** | **1.0** | default | **159** | **provisional best** |
| 8B | `m8_inv_b04` | 1.25 | default | 160 | |
| 8B | `m8_inv_b04` | 1.0 / 1.25 | top3 | 154 / 157 | |
| 8B | `kev8b_e17` | 0.75 | default | 157 | |
| 27B | base | — | — | 194 | |
| 27B | Vega alone | — | — | 200 | breaks `answer_first` (0.4%) |
| 27B | Vega-GDN / FFN / late | — | — | 194 / 198 / 195 | |
| 27B | Vega + clef | — | — | 203 | `answer_first` 100% |
| **27B** | **`vega_clef_plumb`** | **1.0** | default | **207** | **provisional best** |
| 27B | `m27_clef_plumb_at_half` (alt) | — | — | 207 | tie |
| 27B | `vega_clef_plumb_at_half` (alt) | — | — | 207 | tie |
| 27B | `clef_plumb_vegaffn` (alt) | — | — | 206 | Vega contribution ≈ FFN |
| 27B | `e27_early` (B10 analog) | — | — | 201 | early block fails on 27B |
| 27B | vega27b/c/d (modules / compression) | — | — | 192–203 | splitting / compressing loses it |
| 27B (v1) | base | — | default | 176 | `ternary-bonsai-27b`, deterministic |
| 27B (v1) | `vega_clef_plumb` (2-27b adapter) | 1.0 | default | 187 | +11; `answer_first` 10.4% → 91.3% |

## Decision Index (DI)

**Scope:** every DI run in this document was made on **`ternary-bonsai-2-27b`** (Bonsai 2, Qwen3.8).
The v1 `ternary-bonsai-27b` was not run on the Decision Index.

`A_native`, ACOS separate.

| comparison | sample | `A_native` base → LoRA | ACOS | sign | p | verdict |
|---|---|---|---|---|---|---|
| 4B base → `B10` | DI-300 | .4358 → **.4723** (+.037) | .088→.152 | 15/8 | .210 | + (CI just covers 0) |
| 4B base → `B10`×λ0.75 | DI-300 | .4358 → .4710 | .122 | 16/6 | **.052** | +, but 231 150 |
| 4B base → `all3_align` λ0.75 | DI-300 | .4358 → .4606 | .144 | 14/8 | .286 | weak + |
| 4B base → champ λ1.25 | DI-300 | .4358 → .4307 | .115 | 12/14 | .845 | − |
| **27B base → `vega_clef_plumb`** | **DI-500** | **.5772 → .6090** (+.032) | .222→.168 | **22/10** | **.050** | **+ (borderline)** |
| 27B base → `vega_clef_plumb` | DI-100 | .5748 → .5612 | .000→.226 | 9/8 | 1.0 | tie (small sample) |
| 8B base → `m8_inv_b04` | DI-100 | .5332 → .5221 | .111→.200 | 6/5 | 1.0 | tie |
| 1.7B base → `m17_tiny25_single` | DI-1k | .2987 → **.3903** (+.092) | — | 12/4 | .077 | + (CI [+.016, +.172]) |

**DI-500 (27B, per-track, base → `win`).** Gains and losses:

- Large gains: GPQA .20→**.70**, CRUXEval .40→.70, BANKING77 .87→**1.0**, HoVer .89→**1.0**,
  Amazon ESCI .49→.65, CLINC150 .64→.78, ToolRet .50→.63, MMLU-Pro .56→.67, BBH .56→.67,
  When2Call .33→.44.
- Losses: SGD .56→.24, SATA .40→.10, cfcolor .60→.40, GSM8K .60→.45, ACOS .22→.17.

### DI-1500 (sample-1500, all best LoRAs, base skipped)

| model | LoRA | `A_native` | ACOS |
|---|---|---:|---:|
| 1.7B | `m17_tiny25_single` | .3388 | .036 |
| 4B | `B10` | .4542 | .094 |
| 4B | `all3_align` | .4486 | .081 |
| 8B | `m8_inv_b04` | .4738 | .109 |
| 8B | `R3` | .4656 | .102 |
| 27B | `vega_clef_plumb` | .6143 | .164 |
| 27B | `m27_clef_plumb_at_half` | .6180 | .155 |
| 27B | `vega_clef_plumb_at_half` | **.6191** | .166 |
| 27B | `clef_plumb_vegaffn` | .6100 | .156 |

Order by size: **27B > 8B > 4B > 1.7B**. Within 4B, `B10` > `all3_align`; within 8B, `m8_inv_b04` >
`R3`; within 27B, `vega_half` ≈ `at_half` ≈ `win` > `vegaffn` (all within noise, n=43 tracks).

**DI samples.** `sample-300` has 300 rows (ACOS is 62% of questions). `sample-500-hybrid` is the kit's
`--n 500` minus 9 rows over 16k = **491 rows**, ≤16k, ACOS ~58%, kit subsets applied, context 16384.
DI-100 is underpowered (tracks n=2) and must not be used for decisions.

## Latency

Sample-1500, p50 / p95 in milliseconds.

| run | p50 | p95 |
|---|---:|---:|
| 1.7B `tiny` | 75.6 | 2 227 |
| 4B `B10` / `align` | 162 / 166 | 5 061 / 5 163 |
| 8B `inv` / `R3` | 257 / 256 | 7 892 / 7 884 |
| 27B `at_half` / `vegaffn` / `win` / `vega_half` | 889 / 922 / 937 / 968 | 26 767 / 28 235 / 28 820 / 29 494 |

Clef reports p50 209 / p95 239 ms (H200, its own head). Our p95 is inflated by long rows (BRIGHT /
HoVer, context up to 32k); by p50, 4B/8B are comparable and 27B is ~0.9 s.

## Public comparison: `Cloudflare/clef`

**Important:** the Clef card only says *"per-benchmark results from our internal run of the Decision
Index 0.2.1 suite"* — it does **not** state the run size (full suite or a sample). We do not know it,
so compare absolutes with care.

Decision Index 0.2.1, per benchmark (percent; `n` is the number of questions). Only the **base** column
is DI-500-hybrid; every other column is **sample-1500**, so base↔LoRA is not a strict comparison.

Column legend:

| column | LoRA |
|---|---|
| `base` | no LoRA (DI-500-hybrid) |
| `1.7B` | `m17_tiny25_single` |
| `4B B10` | `B10` |
| `4B align` | `all3_align` |
| `8B inv` | `m8_inv_b04` |
| `8B R3` | `R3` |
| `27B win` | `vega_clef_plumb` |
| `27B at_half` | `m27_clef_plumb_at_half` |
| `27B vega_half` | `vega_clef_plumb_at_half` |
| `27B vegaffn` | `clef_plumb_vegaffn` |

| track (n) | base† | 1.7B | 4B B10 | 4B align | 8B inv | 8B R3 | 27B win | 27B at_half | 27B vega_half | 27B vegaffn | **Clef** | Clef-flash | Jev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BFCL (29) | 90 | 41.4 | 82.8 | 79.3 | 89.7 | 82.8 | 96.5 | 96.5 | 96.5 | 96.5 | 98.5 | 98.8 | 95.8 |
| ToolRet (32) | 49.9 | 35.2 | 53.5 | 56.7 | 53.2 | 57.1 | 60.3 | 59.9 | 61.6 | 57.5 | 69.2 | 66.4 | 65.3 |
| API-Bank (29) | 90 | 0 | 48.3 | 44.8 | 48.3 | 55.2 | 96.5 | 100 | 100 | 96.5 | 91.9 | 93.1 | 88.2 |
| BANKING77 (29) | 86.7 | 37.3 | 29.1 | 32.7 | 75.5 | 74.3 | 94.8 | 94.8 | 94.8 | 94.8 | 94.2 | 90.9 | 79.7 |
| CLINC150+OOS (29) | 64.4 | 18.7 | 37.1 | 34.4 | 43.9 | 54.5 | 75 | 68.0 | 68.0 | 68.0 | 97.4 | 66.8 | 89.3 |
| Home appliance simulator (29) | 30 | 0 | 0 | 0 | 0 | 3.5 | 44.8 | 51.7 | 51.7 | 51.7 | 83.0 | 97.7 | 52.3 |
| ContractNLI (29) | 76.7 | 20.4 | 44.0 | 45.9 | 46.5 | 50.9 | 78.6 | 79.1 | 78.8 | 78.4 | 81.4 | 84.3 | 71.7 |
| BPoMP (130) | 88.5 | 56.1 | 66.9 | 60.0 | 65.4 | 51.5 | 95.4 | 94.6 | 95.4 | 95.4 | 96.9 | 95.4 | 90.6 |
| MMLU (29) | 70 | 51.7 | 69.0 | 62.1 | 69.0 | 69.0 | 82.8 | 86.2 | 82.8 | 82.8 | 90.3 | 91.8 | 91.7 |
| GPQA Diamond (29) | 20 | 27.6 | 24.1 | 27.6 | 27.6 | 24.1 | 51.7 | 44.8 | 48.3 | 44.8 | 48.0 | 51.0 | 78.3 |
| ARC-Easy (29) | 100 | 72.4 | 82.8 | 86.2 | 93.1 | 96.5 | 100 | 100 | 96.5 | 100 | 99.0 | 99.5 | 99.3 |
| ARC-Challenge (29) | 100 | 65.5 | 75.9 | 75.9 | 79.3 | 79.3 | 89.7 | 96.5 | 96.5 | 93.1 | 97.7 | 98.3 | 97.8 |
| WinoGrande (29) | 80 | 69.0 | 69.0 | 69.0 | 75.9 | 72.4 | 86.2 | 89.7 | 89.7 | 89.7 | 93.5 | 97.5 | 92.0 |
| HellaSwag (29) | 90 | 44.8 | 65.5 | 72.4 | 65.5 | 65.5 | 96.5 | 96.5 | 96.5 | 96.5 | 98.2 | 98.6 | 94.5 |
| GSM8K (58) | 60 | 22.4 | 34.5 | 32.8 | 46.6 | 46.6 | 51.7 | 51.7 | 53.4 | 53.4 | 80.8 | 67.3 | 79.9 |
| ChessBench (29) | 0 | 10.3 | 17.2 | 10.3 | 6.9 | 6.9 | 17.2 | 17.2 | 6.9 | 13.8 | 24.7 | 23.0 | 17.2 |
| MuSR (29) | 60 | 44.8 | 62.1 | 69.0 | 58.6 | 51.7 | 69 | 69.0 | 69.0 | 69.0 | 83.5 | 86.0 | 66.1 |
| SATA-Bench (29) | 40 | 0 | 13.8 | 13.8 | 20.7 | 17.2 | 24.1 | 27.6 | 31.0 | 27.6 | 33.8 | 36.7 | 26.4 |
| BRIGHT (29) | 37 | 14.1 | 28.4 | 28.6 | 32.9 | 32.1 | 36 | 34.3 | 34.0 | 35.6 | 45.9 | 39.3 | 47.5 |
| ACOS (124) | 22.1 | 3.6 | 9.4 | 8.1 | 10.9 | 10.2 | 16.4 | 15.5 | 16.6 | 15.6 | 33.3 | 25.9 | 29.5 |
| FinEntity (29) | 85.5 | 48.6 | 64.2 | 68.9 | 85.7 | 86.3 | 91.2 | 91.2 | 91.2 | 91.2 | 96.2 | 97.1 | 87.0 |
| CRUXEval (28) | 40 | 25.0 | 21.4 | 21.4 | 32.1 | 32.1 | 57.1 | 57.1 | 64.3 | 57.1 | 86.7 | 86.1 | 73.0 |
| CLadder (28) | 70 | 53.6 | 67.9 | 67.9 | 71.4 | 67.9 | 82.1 | 89.3 | 85.7 | 82.1 | 94.0 | 97.7 | 72.6 |
| Habermas Machine (28) | 50 | 25.0 | 32.1 | 39.3 | 32.1 | 32.1 | 42.9 | 32.1 | 39.3 | 35.7 | 68.7 | 71.8 | 45.9 |
| PhishNChips phishing decisions (28) | 77.8 | 67.9 | 39.3 | 39.3 | 39.3 | 39.3 | 78.6 | 67.9 | 67.9 | 78.6 | 79.6 | 75.0 | 62.5 |
| MMLU-Pro (28) | 55.6 | 14.3 | 21.4 | 17.9 | 28.6 | 25.0 | 57.1 | 50.0 | 42.9 | 50.0 | 65.9 | 65.3 | 82.7 |
| BBH fixed-option tasks (28) | 55.6 | 21.4 | 39.3 | 46.4 | 42.9 | 46.4 | 60.7 | 67.9 | 60.7 | 67.9 | 73.7 | 68.9 | 92.9 |
| HoVer claim verification (28) | 88.9 | 67.9 | 67.9 | 67.9 | 64.3 | 67.9 | 75 | 82.1 | 82.1 | 78.6 | 65.2 | 61.2 | 72.9 |
| When2Call MCQ (28) | 33.3 | 57.1 | 60.7 | 50.0 | 60.7 | 57.1 | 71.4 | 67.9 | 67.9 | 67.9 | 72.4 | 65.6 | 81.0 |
| New Yorker caption matching (28) | 77.8 | 21.4 | 60.7 | 46.4 | 46.4 | 42.9 | 53.6 | 53.6 | 46.4 | 53.6 | 69.5 | 66.1 | 70.1 |

† `base` is DI-500-hybrid; all other columns are sample-1500.

**Careful:**

- Our tracks have **n = 6–40** (sample-1500), so individual deltas are noisy (one question = 10 pp at
  n=10).
- Our best run is sometimes "above" Clef (GPQA 51.7 vs 48.0), but that is n=29 plus base degeneracy (the
  base spams `B` on MCQ) — not a conclusion.
- Only the **sign test over 43 tracks** (22/10, p=.050) and the large tracks (ACOS n=40, ToolRet n=6)
  are reliable.

The full per-track matrix for all runs is in [DI-TRACKS.md](DI-TRACKS.md).

The full Clef table is in the `Cloudflare/clef` card. Our `clef` LoRA is distilled from Cloudflare/clef,
so `win` closes part of the gap, but Clef is still ahead in absolute reasoning terms (at its full n).

## Per-track matrix

[DI-TRACKS.md](DI-TRACKS.md) holds the track × run matrix for the sample-1500 runs (best value per
track in bold).

## See also

- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) — which LoRA to use per model and the selection rule.
- [RECIPES.md](../docs/RECIPES.md) — how each final LoRA was produced and how to reproduce it.
- [FLIGHT.md](FLIGHT.md), [DIRECT-INFER.md](DIRECT-INFER.md) — control tests (flight simulation,
  direct text generation).
