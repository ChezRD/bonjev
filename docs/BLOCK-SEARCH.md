# Block/layer LoRA search for Bonsai

How the LoRA deltas were dissected into blocks (modules) and layers, how combinations were assembled,
and what came out. The method is to pick **concrete blocks and layers, not global percentages of a
source**.

The final per-model guidance is in [RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md); the recipes
are in [RECIPES.md](RECIPES.md); the adapter inventory is in [LORAS.md](LORAS.md).

**Contents**

- [Why](#why)
- [Glossary](#glossary)
- [How a combination is built](#how-a-combination-is-built)
- [8B results](#8b-results)
- [4B result](#4b-result)
- [1.7B](#17b)
- [Per-model bottom line](#per-model-bottom-line)
- [27B per-block source profile](#27b-per-block-source-profile)
- [Technical facts](#technical-facts)
- [Tool files](#tool-files)

## Why

Bonsai is a base model (1.7B / 4B / 8B / 27B); we do not change it. We assemble **one LoRA** for it
from third-party LoRA deltas (taken from trained models), training-free. The goal is not just to fit
JevBench-231 but to raise decision quality in general (Decision Index, DI).

## Glossary

### Delta sources

All under `work/deltas` or `work/experiments/*/parts`:

- `kev8b`, `kev4b` — the "Kev" decision LoRA for 8B / 4B.
- `LCT-8b` — Jev-LCT 8B (a late / cross-base delta).
- `typed-8b` — typed-decisions 8B.
- `candigate-4B`, `senna-4B`, `mogita-4B` — 4B decision deltas.
- `tiny-jev-1.7B`, `xuhao-1.7B` — 1.7B deltas.

### Modules

GGUF tensor names: `attn_q`, `attn_k`, `attn_v`, `attn_output` (= o_proj), `ffn_gate`, `ffn_up`,
`ffn_down` (= down_proj).

Groups we use:

- `attn` = q, k, v, o
- `qkv` = q, k, v
- `o` = attn_output
- `ffn` = gate, up, down
- `gu` = gate, up
- `down` = ffn_down

### Layer windows

| model | blocks | `early` | `mid` | `late` |
|---|---:|---|---|---|
| 36-layer 4B / 8B (qwen3) | 36 | 0–17 | — | 18–35 |
| 28-layer 1.7B (qwen3) | 28 | 0–13 | — | 14–27 |
| 64-layer 27B (qwen35) | 64 | 0–15 | 16–39 | 40–63 |

For the 27B, the clef core is **40–47** (62–73% depth). Relative-depth mapping to the smaller
models: 9B ~20–23, 8B / 4B ~22–26, 1.7B ~17–20.

### 27B hybrid layout

The 27B (`qwen35`) is a **hybrid 3:1** stack: three **Gated DeltaNet** (linear-attention,
recurrent-state) blocks per one **full gated-attention** block. The full-attention interval is 4.

| property | value |
|---|---|
| blocks / hidden | 64 / 5120 |
| full-attention blocks | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 (16 of 64) |
| GDN blocks | the other 48 |
| per-block weights | GDN ~98.1 MiB, full ~94.3 MiB, block 63 ~104.9 MiB; file 6.71 GiB |
| KV cache at 16384 ctx | 288 MiB over the 16 full blocks (K q4_0 144 + V q4_0 144) |
| recurrent state | 448.88 MiB (3 cells × 64 layers) |

Why it matters:

- Only the 16 full-attention blocks carry a **KV cache**; the GDN blocks carry a **recurrent state**.
  A KV tail therefore cannot be trimmed — the engine restores a prefix checkpoint instead.
- The GDN state propagates through the whole stack, so **early edits pass through the entire
  recurrence**. That is why early-layer deltas (openjev, Kahn1) are risky on 27B, while the late
  window works (clef 40–63).
- The full-attention blocks are the ones that carry the KV and are usually the riskiest; the late
  full blocks **43 / 47 / 51 / 55 / 59 / 63** sit inside the clef window.

The layout comes from the GGUF metadata (`general.architecture = qwen35`, `full_attention_interval = 4`)
and the loader log. The 4B / 8B / 1.7B models are **dense qwen3** (no GDN blocks).

### Merge operations

- `align+budget α` — `spectral_merge3 --align --budget α --rescale`: a shared basis plus a
  rank-direction budget.
- `λ` — runtime amplification of a finished LoRA (`BONJEV_LORA=file,λ`).
- `CAT` — exact packing of a sum into one file (factor concatenation).

### Metrics

- `231` — JevBench (deterministic, `BONJEV_NO_REUSE=1`).
- `margin` — median top1−top2 gap (confidence).
- `near-tie` — share of tasks with margin < 0.10.
- `DI` — Decision Index, mean native per-track score (sample-100 or sample-300).

### Artifact names in the logs

- `m_*` / `s3_*` — 8B module / add-on scan.
- `L01–L16` — 8B window × module combinations.
- `R1–R6`, `s5_*`, `s6_*` — safe-block combinations for 8B.
- `B01–B14` — the 4B grid; `B10a–h` — 4B refinement.
- `C01–C12` — the 1.7B grid.

## How a combination is built

1. Slice the source into `(layer, module)` (a copy by mask).
2. Merge the chosen blocks.

   ```bash
   python scripts/spectral_merge3.py out_dir 128 --align --budget 0.4 --rescale <blocks...>
   ```

3. Write the GGUF (r128 `q8_0`; rank < 32 → `f16`).

   ```bash
   python scripts/gguf_lora_writer.py <parts> <out.gguf> 128 qwen3 q8_0
   ```

4. Run the 231 exam with a probability dump, and read margin / near-tie.

   ```bash
   scripts/eval_exam.sh <model> <lora|-> out.jsonl
   ```

## 8B results

Existing champion: `m8_inv_b04` = `kev8b(attn+ffn) + LCT_attn×0.25`, align+budget 0.4 → **159**; at
λ=1.25 → **160**.

What the block search gave:

- **Individual kev8b modules are weak:** `attn` 152, `qkv` 152, `o` 151, `ffn` 150, `gu` 149, `down`
  145 (base 149).
- **LCT is a dangerous late source:** at weight 1.0, late `attn` / `o` / `gu` / `ffn` **break the
  readout** (49–116 of 231; `answer_first` drops to 0–28%). The champion therefore uses LCT at dose
  0.25.
- **Safe:** LCT **early attn** (156) and LCT **late down** (156, margin .989, near-tie 2.2%).
- **Safe-block combinations (S6):** the best is `kev attn+ffn + LCT-late-down`, budget 0.5, λ=1.25 →
  **159**, **margin .995, near-tie 1.3%** (the most confident 8B variant).
- No set beat the champion on 231. 8B bottom line: max **160** (champion λ=1.25); for "confidence",
  `R3` (159, margin .995).

## 4B result

Champion: `m4b_all3_align` = `kev4b + candigate + senna` (align+budget 0.3) → 159; λ=1.25 → 163;
DI-100 .450.

The block search (grid B01–B14, weight 1.0, align+budget 0.4) found:

- **`B10` = `kev4b` early layers 0–17 only (attn+ffn) + candigate + senna** → **231 = 161** (λ=1.0),
  **DI-300 = .476** (base .438, **+3.8 pp**).
- Refinement (B10a–h): narrower window (0–11) 154, wider (0–23) 156, attn only 156, ffn only 152,
  without senna 153, without candigate 159 — **`B10` (window 0–17, attn+ffn, all three sources) is
  best**.
- Meaning: **choosing the early layers** of kev4b makes the LoRA *generalize* (DI rises), not just fit
  the exam. The same trick did not work for 8B (early kev 153–156).
- **DI-300 comparison (294 rows):** base `.438`; champion `all3_align` λ=1.25 (231=163) — **`.433`
  (−0.5 pp, harmful)**; **`B10` λ=1.0 — `.476` (+3.8 pp)**. `B10` is the best product: 231 almost the
  champion's, but clearly better general decisions.
- Budget confirmed: `B10` at `budget 0.4` = 161; at 0.3 — 156/157 (λ1.0/1.25); at 0.5 — 158/155. The
  optimum is 0.4.

## 1.7B

Champion: `m17_tiny25_single` (tiny×2.5, one file) = **143**; runtime tiny,2.5 = 144; with the `top3`
style ensemble — **147**.

The block/layer grid (C01–C12: tiny/xuhao × attn/ffn × early/late) at weight 1.0 (no λ):

- Best is **C06 (full tiny + xuhao attn) = 133**, C12 (tiny + xuhao attn + xuhao ffn) = 132, C07
  (tiny + xuhao ffn) = 130; the rest 120–126 (base 124).
- So for 1.7B it is **λ amplification of the full tiny, not block selection**, that wins.
- C06/C12 at λ=2.0/2.5 were checked: C06 λ=2.0 = 138, C06 λ=2.5 = 139, C12 λ=2.5 = 141 — they **did
  not beat** tiny×2.5 (144).

1.7B conclusion: the win is **λ amplification of the full tiny**; block/layer selection does not help.

## Per-model bottom line

- **4B** — the block/layer winner is **`B10`** (`kev4b` early 0–17 + `candigate` + `senna`,
  align+budget 0.4): 231 **161**, DI-300 **.476** (+3.8 pp). The champion `all3_align` λ=1.25 gives 231
  163 but DI .433 (below base) — exam overfitting. `B10` is the best product.
- **8B** — the winner is the champion `kev8b(attn+ffn) + LCT_attn×0.25`: 231 159 (λ=1.25 → 160). The
  block search did not beat it; a "confident" variant `R3` (159, margin .995) was found.
- **1.7B** — the winner is `tiny×2.5`: 143 (one file) / 144 (runtime), 147 with `top3`. Block selection
  does not help.

Final confidence (margin, deterministic): 1.7B `m17_tiny25_single` 143 / margin .641 / near-tie 11.3%;
4B `B10` 161 / .937 / 5.2%; 8B `m8_inv_b04` 159 / .878 / 4.8%.

## 27B per-block source profile

Fraction of each source's mass per depth window, plus module split.

| source | window 0–15 | 16–39 | 40–63 | top layers | FFN / attn / GDN |
|---|---:|---:|---:|---|---|
| **Vega-27B** (Qwen3.8-27B, r512) | .25 | .39 | .36 | 63, 28, 29, 25, 26, 48 | .54 / .11 / **.35** |
| clef-27B | .00 | .00 | 1.00 | 40, 63, 62, 61, 41 | .53 / .37 / .10 |
| plumb | .25 | .39 | .36 | 24, 22, 29, 30, 28 | .55 / .34 / .10 |
| autotrust | .27 | .41 | .32 | 25, 26, 20, 27, 28, 62 | .55 / .37 / .08 |
| at-mid | .00 | 1.00 | .00 | 25, 26, 20, 27, 28 | .57 / .36 / .07 |
| canopy | 1.00 | .00 | .00 | 1, 0, 2, 3 | .59 / .33 / .08 |
| openjev | .29 | .39 | .32 | 7, 3, 15, 23, 19 | .00 / .74 / .26 |
| simplejev | .27 | .42 | .32 | 6, 20, 8, 29, 41 | .57 / .34 / .09 |
| novel2 | .25 | .37 | .38 | 62, 2, 59, 3, 61 | .58 / .31 / .11 |

Key: **Vega is all-layer, mid+late (25–29, 48, 63), GDN-heavy (.35)** — unlike clef (late, FFN/attn).
Combination hypotheses: Vega-late × clef (different modules), Vega-GDN × clef-FFN, Vega-mid ×
autotrust-mid (overlap). Tool: `scripts/lora_block_profile.py`.

### Layer study

- clef-27B is **r256, eff_rank(ΔW)=147**; r64 loses ~12% energy → use r128+ (an r64 projection is not a
  valid basis).
- Clef invariant: **FFN ~53–59% / attn ~9–12% / GDN ~29–38%** (FFN carries the signal). The depth
  window is not invariant: 27B touches 62–98%, flash the whole depth; "late-only" is likely an
  artifact of the base diff.
- **Two regimes:** late (62–98%) improves 231 / format; **early (0–47%) improves DI / generalization**
  (`B10`).
- clef × vega overlap: **attention is 2–4× higher** (O_B .134 vs FFN .043 / GDN .051) → separate the
  sources by module and minimize shared attention.
- `at` (+3) is concentrated (r16, eff_rank 12.6), Vega is diffuse (eff_rank 89) → **compress Vega to
  r16–32**.
- **Fast-forward gate:** `answer_first` ≥ 99%, special/empty ≈ 0, margin not below base.
- clef-flash impact ≈ 0 on a healthy quant (Q4_K_M: 184 = 184); the weights do not transfer
  (cross-arch).

### Dump analysis

- Working adapters keep **`answer_first` 99–100%**, special ≈ 0, and improve margin / near-tie (4B
  .743→.782, 8B .691→.727).
- **Late-attention LCT breaks the readout:** `answer_first` 0–28% and **`rank3%` → 42–52%** (the
  expected option drops out of the top-3), not merely "the wrong top-1".
- **`answer_first` ≠ quality** (the engine reads slot logits); the best predictor of 231 is **`rank3%`**,
  then margin / near-tie; **`m_wrong`** (confidence on wrong answers) predicts DI harm.
- Modules: FFN/down and kev8b attn/qkv/o are safe; late LCT-attn at 1.0 is forbidden.
- Fast-forward gate: **`answer_first` ≥ 99% AND `rank3%` not below base**; 1.7B — `top3`; compress Vega
  to r16–32.

## Technical facts

- **Determinism:** all numbers use `BONJEV_NO_REUSE=1`. Without it, results depend on request history
  (CUDA non-invariance: near-tie flips); for dense models this is not a bug, for the hybrid 27B it is a
  real bug (stale GDN state). Hence the honest bases: 1.7B 124, 4B 145, 8B 149.
- **Run DI only when 231 improved** (otherwise too expensive and noisy).
- **Cross-size transplants are useless** (27B→8B, kev8b→4B, 4B→8B, 0.6B→1.7B — 4 failures).
- **Failed:** per-layer and per-module scaling, TIES/DARE, Pico/CAT on small models, orthogonal tricks,
  calibration for argmax (improves ECE but hurts the score).

## Tool files

- `scripts/spectral_merge3.py` (align+budget/pico/do/rescale), `scripts/spectral_concat.py` (CAT),
  `scripts/layer_scale.py` (layer scaling), `scripts/gguf_lora_writer.py`.
- Build a recipe: `scripts/build_lora.sh <recipe>`; run the 231 exam: `scripts/eval_exam.sh`.
- Dumps / analysis: `benchmarks/jevbench_exam.py --dump-probs`.
