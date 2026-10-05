# Direct generation: garbled-text check

Goal: verify that the models and LoRAs produce **coherent text**, not garbage, under **direct
generation** — not the decision readout, which only reads label logits.

**Contents**

- [Setup](#setup)
- [Results](#results)
- [Conclusion](#conclusion)

## Setup

- Engine: `llama-server` built from the Prism fork (target built locally).
- Harness: `scripts/direct_infer.py`; matrix: `scripts/direct_infer_all.sh`.
- **50 prompts** per combo (facts, arithmetic, haiku, translation, lists), `temp=0`, `n_predict=40`.
- Metrics: `empty` (empty output), `garbled` (share of non-printable > 2%), `rep≥6` (a run repeated
  ≥6 times), `uniq_word_ratio` (word diversity).

## Results

All LoRAs are `work/best/` recipes.

| model | LoRA | empty | garbled | rep≥6 | uniq_word |
|---|---|---:|---:|---:|---:|
| 1.7B | base | 0 | 0 | 0 | .52 |
| 1.7B | `m17_tiny25_single` | 0 | 0 | 0 | .54 |
| 4B | base | 0 | 0 | 0 | .64 |
| 4B | `B10` | 0 | 0 | 0 | .65 |
| 4B | `all3_align` (alt) | 0 | 0 | 0 | .63 |
| 8B | base | 0 | 0 | 0 | .60 |
| 8B | `m8_inv_b04` | 0 | 0 | 0 | .61 |
| 8B | `R3` (alt) | 0 | 0 | 0 | .63 |
| 27B | base | 0 | 0 | 0 | .63 |
| 27B | `vega_clef_plumb` | 0 | 0 | 0 | .65 |
| 27B | `m27_clef_plumb_at_half` (alt) | 0 | 0 | 0 | .66 |
| 27B | `vega_clef_plumb_at_half` (alt) | 0 | 0 | 0 | .65 |
| 27B | `clef_plumb_vegaffn` (alt) | 0 | 0 | 0 | .65 |

## Conclusion

- **No `empty` / `garbled`:** every model and every `work/best` LoRA generates **coherent text**
  (examples: "The capital of France is Paris.", "Au, which comes from the Latin word \"aurum\"…", a
  haiku about the sea). The LoRAs **do not break** the base language ability.
- The only quirk is **repetition** on some prompts for the smaller / base configs (1.7B and 27B-base,
  e.g. "2 + 2 = 4 2 + 2 = 4…"), but that is coherent text, not garbage.
- So the decision readout does not hide a degradation: the models remain language models.

Raw outputs and server logs are kept outside the repository.
