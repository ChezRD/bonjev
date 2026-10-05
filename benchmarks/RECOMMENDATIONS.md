# Recommendations: which LoRA to use for each Bonsai model

One LoRA per model (the "default"); everything else is an alternate or is not recommended. Build the
files with `scripts/build_lora.sh` (see [RECIPES.md](../docs/RECIPES.md)).

## Defaults

| Model | Default LoRA | 231 | DI (native) |
|---|---|---:|---:|
| 1.7B | `m17_tiny25_single` (+ `top3`) | **147** | not claimed |
| 4B | `B10` | **161** | **.476** |
| 8B | base (no LoRA) | 149 | **.536** |
| 27B | `vega_clef_plumb` | **207** | **.609** |

Serve with `--lora <name>` or `BONJEV_LORA=<name>`; the engine downloads the adapter by short name.
Append `,λ` to amplify (task arithmetic). The names (also in `bonjev loras`):

| model | default | alternates |
|---|---|---|
| 1.7B | `m17_tiny25_single` | — |
| 4B | `b10` | `all3_align` (DI−) |
| 8B | `m8_inv_b04` | `r3` (over-confident) |
| 27B | `vega_clef_plumb` | `m27_clef_plumb_at_half`, `vega_clef_plumb_at_half`, `clef_plumb_vegaffn` |

**Multiple adapters** attach additively (`--lora a,λ --lora b,λ`, up to 8) but are **research-only**:
tested runtime composites did not beat the single adapter (1.7B `tiny,2.5 + xuhao_ffn,0.25` 142 < 144;
8B `kev,1.5 + lct_attn,0.25` 159 < 160; 4B `kev4b + candigate + senna` 158 < 161). Ship one adapter per
model.

## Selection rule

The Decision Index (DI) is the arbiter; the 231 exam is the guardrail. Keep a candidate only if all of
the following hold:

1. **231 did not drop** — target Δ231 ≥ +3.
2. **DI not worse than base** — paired McNemar over DI questions, not just the mean.
3. **Fast-forward preserved** — `answer_first` ≥ 99%, special/empty ≈ 0, `rank3%` not below base,
   `m_wrong` without a sharp rise.

Why DI arbitrates: the 231 exam cannot separate close variants (top 4B results are statistically
indistinguishable: `B10` 161 vs the champion 163, p=0.79), while DI can (`B10` is significantly better
than base and champion, p<0.0001).

Two facts to keep in mind:

- **Within one model, 231 and DI are almost uncorrelated** (4B: +0.33 track / +0.12 field; 8B: −0.59 /
  +0.01). Across sizes they correlate (r≈+0.8), but the choice is made within a model.
- **Where the DI signal is** (stratified by option count, sample-300): almost all of it is in **2–6
  options** (1727 of 1805 scored). The 7–26 (49) and >26 (29) strata are too small to conclude.
  **ACOS is 1206 questions with exactly 2 options** (binary sentiment), not many-option.

Divergence example:

- 4B `all3_align` at λ=1.25 gives the maximum 231 (163) but DI **.433** (below base).
- 4B `B10` gives 161 but DI **.476**.

The arbiter (DI) picks `B10`.

## 1.7B — `m17_tiny25_single`

- **What it is:** the full Tiny-Jev delta, scaled ×2.5, baked into one file (rank 64, q8_0).
- **Result:** 231 **143** (base 124); with `BONJEV_STYLE=top3` — **147**, `answer_first` 99.6%.
- **DI:** not claimed. The apparent gain is **not robust** (ACOS per-review F1 does not change; the gain
  comes from 12 tiny tracks with n=1–6).
- **Conclusion:** for 1.7B the win comes from **amplification (λ) + the `top3` ensemble**, not from
  block or layer selection.

## 4B — `B10`

- **What it is:** `kev4b` **early layers 0–17 only** (attention+FFN) + `candigate` + `senna`, merged
  with `align+budget 0.4` (rank 64).
- **Result:** 231 **161** (base 145), DI-300 **.476** (+3.8 pp), margin .937, `m_wrong` .729.
- **Alternate:** `all3_align` (all `kev4b` layers) gives the maximum 231 (163 at λ=1.25) but DI **.433**
  — **below base**. Do not use it; take `B10`.
- **Conclusion:** the **early layers** of `kev4b` make the LoRA generalize (DI rises); the all-layer
  variant only fits the exam format.

## 8B — base without LoRA

- **The 8B base is the DI maximum** (track .5358). All 8B adapters are DI-negative: `m8_inv_b04` .5268,
  λ0.75 .5159, λ1.25 .4405.
- **Do not attach a LoRA by default.** Take `m8_inv_b04` only if 231 is the priority and a DI drop is
  acceptable: 231 **159** (base 149; λ=1.25 → 160), margin .878, `rank3%` 64.1%, `m_wrong` .688.
- **Do not take:** `R3` (over-confident: `m_wrong` .831) or λ=1.25 (DI .4405).
- **LCT rule:** LCT is allowed only as late `ffn_down`; **late-attention LCT is forbidden** (it breaks
  the readout).

## 27B — `vega_clef_plumb`

- **What it is:** Vega (Decision 2.0, rank 128) + `clef` + `plumb`, merged with a plain sum (rank 128).
- **Result:** 231 **207** (base **194**). There is a tie at 207 with `m27_clef_plumb_at_half` and
  `vega_clef_plumb_at_half`.
- **Cross-line:** the same adapter also loads on the v1 `ternary-bonsai-27b` (Qwen3.6, same `qwen35`
  architecture) and gives 176 → **187** (+11); `answer_first` 10.4% → 91.3%. See [EXAM.md](EXAM.md).
- **DI:** on the DI-500 hybrid sample the winner is better than base on the native per-track mean
  (`A_native` .5772 → .6090, sign 22/10, p=.050). Full-coverage confirmation is pending.
- **Vega is diffuse:** splitting it by module or compressing to r16 loses the gain (Vega-GDN 194,
  Vega-FFN 198, compressed 197–203). Keep the full Vega delta.
- **Fast-forward:** `vega_alone` breaks the readout (`answer_first` 0.4%, 99.6% echo); **clef restores
  it** (100%). The merged default keeps `answer_first` 100%, special/empty 0%.

## What worked and what did not

| Works | Does not work / dangerous |
|---|---|
| `align+budget` (shared basis + rank budget) | Per-layer and per-module scaling |
| λ>1 on full-FT deltas (1.7B) | TIES / DARE |
| Early-layer selection for generalization (4B `B10`) | Pico / CAT on small models |
| CAT for exact packing | Cross-size transplants (27B→8B, kev8b→4B, 4B→8B, 0.6B→1.7B) |
| — | Late-attention LCT (breaks the readout) |

Two regimes: **early layers → generalization (DI)**; **late layers → format (231)**.

Metrics: the best predictor of 231 is `rank3%`; DI harm is predicted by `m_wrong` (confidence on wrong
answers); `answer_first` is a readout-breakage indicator, not a quality signal.

## How to test a new LoRA

1. Run 231 deterministically with a probability dump.

   ```bash
   scripts/eval_exam.sh <model> <lora|-> out.jsonl
   ```

2. Read `rank3%`, `m_wrong`, `answer_first`, and margin; compare with base.
3. If 231 improved, run DI and check ΔDI (native per-track, ACOS separately).

   ```bash
   DI_CTX=16384 scripts/di_run.sh <name> <model> <lora|-> sample-300.jsonl.gz
   ```

4. Keep only if the selection rule above holds; otherwise discard.

## See also

- [RECIPES.md](../docs/RECIPES.md) — how each final LoRA was produced and how to reproduce it.
- [RESULTS.md](RESULTS.md) — the consolidated 231 / DI / flight / direct-inference tables.
- [BLOCK-SEARCH.md](../docs/BLOCK-SEARCH.md) — the block/layer study and glossary.
- [LORAS.md](../docs/LORAS.md) — the adapter map, merge methods, and research survey.
- [TRAINING-FREE-HOOKS.md](../docs/TRAINING-FREE-HOOKS.md) — what can be borrowed without training.
