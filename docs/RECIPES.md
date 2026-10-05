# Recipes: how the final LoRAs were produced

Training-free recombination of LoRA deltas. Pipeline: **source model → delta (parts) → merge → GGUF**.
The files in `work/best/` are copies; this document reproduces them. All commands run from the repo
root.

**Contents**

- [Tools](#tools)
- [Source deltas](#source-deltas)
- [Recipes](#recipes)
- [Consolidated build script](#consolidated-build-script)
- [Reproduce from scratch](#reproduce-from-scratch)
- [Method and research references](#method-and-research-references)

## Tools

| step | script |
|---|---|
| PEFT adapter → parts | `scripts/peft_to_parts.py` (downloads `adapter_model.safetensors` + `adapter_config.json` at a pinned revision, folds α/r into the delta) |
| merged fine-tune (two models) → delta parts | `scripts/extract_delta.py <base> <ft> --rank R` (fp32 weight diff, randomized SVD) |
| Clef model → delta parts | `scripts/extract_clef.py` (rank 256; env `CLEF_*`; text tensors in `parts_text`) |
| merge deltas | `scripts/spectral_merge3.py <out_parts> <rank> [--align --budget B --rescale] [--pico X] [--do R] <input[:scale]>...` |
| scale by layer / window | `scripts/layer_scale.py <in_parts> <out_parts> <scale>` |
| concat without SVD | `scripts/spectral_concat.py` |
| parts → GGUF LoRA | `scripts/gguf_lora_writer.py <parts> <out.gguf> <rank> <arch> <q8_0\|f16>` |

Notes on `spectral_merge3.py`:

- With no flags it is a plain **sum**.
- `--align --budget B --rescale` is the working merge (shared basis over B matrices + rank-direction
  budget).
- `input:scale` sets a per-source weight (e.g. `lct_attn:0.25`).
- `--rescale` renormalizes the merged delta to the plain-sum norm.

> **Determinism note:** `rsvd` inside `spectral_merge3.py` uses an unseeded `torch.randn`, so
> byte-identical rebuilds are not guaranteed. The recipe is validated by the effective delta norm
> `‖dW‖_F` (matches to ~1e-6).

## Source deltas

Model → parts.

Method terms: **delta** = `W_ft − W_base` re-factored as a low-rank LoRA; **PEFT** = the HF adapter
format (`lora_A` / `lora_B` + α/r); **suffix adapter** = the plumb/2 `suffix_adapter.safetensors`
layout (converted with a `×2` factor); **slice** = a partial delta cut by `(layer, module)`. Full
definitions are in [LORAS.md § Sources and extraction](LORAS.md#sources-and-extraction).

| parts dir | source model | base | method | license |
|---|---|---|---|---|
| `kev4b_q3_parts` | `jaredpalmer/kev-4b` (branch `qwen3`) | Qwen3-4B-Base | PEFT | Apache-2.0 |
| `candigate_parts` | `CullenYap/CandiGate-Qwen3-4B` | Qwen3-4B | PEFT | Apache-2.0 |
| `senna_parts` | `sennaLLMLearner/qwen3-4b-system-one-lora` | Qwen3-4B-Instruct-2507 | PEFT | Apache-2.0 |
| `kev8b_attn`, `kev8b_ffn` | `jaredpalmer/kev-8b` (attn / ffn slices) | Qwen3-8B-Base | PEFT | Apache-2.0 |
| `lct_attn` | `CaoHaoWei/Jev-LCT-Qwen3-8B` (layers 34–35, attention) | Qwen3-8B | delta r128 | Apache-2.0 |
| `m17_x_tiny` | `lostargon/Tiny-Jev-1.7B` | Qwen3-1.7B | delta | Apache-2.0 |
| `clef27b_r64` | `Cloudflare/clef` (r256 → r64) | Qwen3.8-27B | `extract_clef.py` | Apache-2.0 |
| `plumb_parts` | `totum-labs/Qwen3.5-27B-plumb` (plumb/2 `suffix_adapter`, r32/α64, ×2) | Qwen3.5-27B | suffix adapter | Apache-2.0 |
| `_at_parts_half` | `autotrust/JEV-27B` (`lora_b ×0.5`) | Qwen3.8-27B | PEFT | Apache-2.0 |
| `vega27_parts` | `vllm-sr/Decision-2.0-Vega-27B` (r512/α1024 → r128) | Qwen3.8-27B | PEFT | Apache-2.0 |
| `vega_ffn` | slice of `vega27_parts` (ffn_gate/up/down) | — | slice | Apache-2.0 |

All source models are Apache-2.0; see [ATTRIBUTION.md](ATTRIBUTION.md).

## Recipes

Parts → final LoRA. `rank` is the output LoRA rank; `arch` is `qwen3` (1.7B/4B/8B) or `qwen35` (27B).

| # | output (`work/best/`) | method | rank | inputs | reproduce | verified |
|---|---|---|---|---|---|---|
| 1 | `1.7b/m17_tiny25_single.gguf` | `layer_scale 2.5` | 64 | `m17_x_tiny` | `build_lora.sh m17_tiny25_single` | ✓ |
| 2 | `4b/B10.gguf` | `align+budget 0.4` | 64 | `kev4b_q3` (layers 0–17, attn+ffn) + `candigate` + `senna` | `build_lora.sh B10` | ✓ |
| 3 | `4b/alt/m4b_all3_align.gguf` | `align+budget 0.3` | 64 | `kev4b_q3` + `candigate` + `senna` | `build_lora.sh all3_align` | ✓ |
| 4 | `8b/m8_inv_b04.gguf` | `align+budget 0.4` | 128 | `kev8b_attn` + `kev8b_ffn` + `lct_attn:0.25` | `build_lora.sh m8_inv_b04` | ✓ `‖dW‖` |
| 5 | `8b/alt/R3.gguf` | `align+budget 0.5` | 128 | `kev8b_attn` + `kev8b_ffn` + `s_lct_down_late` | `build_lora.sh R3` | ✓ |
| 6 | `27b/vega_clef_plumb.gguf` | `sum` | 128 | `clef27b_r64` + `plumb_parts` + `vega27_parts` | `build_lora.sh vega_clef_plumb` | ✓ |
| 7 | `27b/alt/m27_clef_plumb_at_half.gguf` | **unpinned** (ad-hoc) | — | `clef27b_r64` + `plumb_parts` + `_at_parts_half` (align+budget 0.3 approx.) | — | ✗ |
| 8 | `27b/alt/vega_clef_plumb_at_half.gguf` | `sum` | 128 | `clef27b_r64` + `plumb_parts` + `_at_parts_half` + `vega27_parts` | `build_lora.sh vega_clef_plumb_at_half` | ✓ |
| 9 | `27b/alt/clef_plumb_vegaffn.gguf` | `sum` | 128 | `clef27b_r64` + `plumb_parts` + `vega_ffn` | `build_lora.sh clef_plumb_vegaffn` | ✓ |

`s_lct_down_late` = `Jev-LCT-Qwen3-8B` late `ffn_down` (layers 34–35).

Recipe #6 is the published 27B adapter. It also loads on the v1 `ternary-bonsai-27b` (Qwen3.6, same
`qwen35` architecture): deterministic 231 176 → **187** (+11), `answer_first` 10.4% → 91.3%.

Example (recipe #2, `B10`):

```bash
WORK=work/experiments
# kev4b_q3 is sliced to layers 0–17 (attn_ + ffn_) into $WORK/build/kev4b_early first
python scripts/spectral_merge3.py $WORK/build/B10.parts 64 --align --budget 0.4 --rescale \
    $WORK/build/kev4b_early $WORK/pool/parts/candigate_parts $WORK/pool/parts/senna_parts
python scripts/gguf_lora_writer.py $WORK/build/B10.parts $WORK/build/B10.gguf 64 qwen3 q8_0
```

The `kev4b_q3` layers 0–17 are sliced first into `$WORK/build/kev4b_early` (the consolidated script
does this automatically).

## Consolidated build script

`scripts/build_lora.sh <recipe>` builds any recipe from the table (slicing + merge + GGUF) into
`$WORK/build/<recipe>.gguf`. Set `WORK` (default `work/experiments`) to where the source parts live.

```bash
WORK=work/experiments scripts/build_lora.sh B10
WORK=work/experiments scripts/build_lora.sh vega_clef_plumb
```

Recipe #7 (`m27_clef_plumb_at_half`) is an alternate; its exact merge parameters are not pinned, so
`build_lora.sh` does not build it. The 27B default is recipe #6. The script also builds `e27_early`
(the 27B early-block analog of `B10`, layers 0–15, 201/231) which is not shipped.

## Reproduce from scratch

1. Download the trained sources and base models into the HF cache, and print the slice commands.

   ```bash
   python scripts/fetch_sources.py            # add --check to only verify paths
   ```

2. Slice each source into delta parts (slow; run per source as printed above).

   ```bash
   python scripts/peft_to_parts.py <snapshot>/adapter_model.safetensors <out_parts>
   python scripts/extract_delta.py <base_snapshot> <ft_snapshot> <out_parts> --rank 128
   CLEF_QWEN_DIR=<base> CLEF_CLEF_DIR=<clef> CLEF_OUT=<out> python scripts/extract_clef.py
   ```

3. Build any final LoRA from its parts (slicing + merge + GGUF).

   ```bash
   WORK=work/experiments scripts/build_lora.sh B10
   ```

`scripts/fetch_sources.py --check` prints, for every source, whether the adapter/base are in the HF
cache and whether the delta `parts` target already exists, so paths can be verified before the slow
slicing. One source (the plumb/2 suffix adapter, recipes #6/#8) has no pinned org and is flagged
`UNPINNED` — pin it before reproducing those recipes.

## Method and research references

The merge is a **training-free** recombination of LoRA deltas (no fine-tuning, no distillation). The
operators implemented by `scripts/spectral_merge3.py` follow published data-free merging methods:

| flag | method | reference |
|---|---|---|
| `--align` | shared basis via joint SVD of the `B` matrices (**KnOTS**) | arXiv:2410.19735 |
| `--budget` | rank-direction budget by a net-utility score | arXiv:2609.22237 |
| `--pico` | calibrate the output `B` space before merging (**Pico**) | arXiv:2604.16826 |
| `--do` | magnitude/direction decoupling for cross-base deltas (**DO-Merging**) | arXiv:2505.15875 |
| `plain sum`, `:scale` | task arithmetic / TIES / DARE families | see [LORAS.md](LORAS.md) |

The design of the merged decisions is benchmark-driven: the external **Decision Index** kit
([github.com/apolinario/decision-index](https://github.com/apolinario/decision-index), edition 0.2.1)
is the arbiter, JevBench-231 is the guardrail — see [RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md).

Source models were trained by their authors (we only recombine their deltas):

- **Clef / Clef-flash** (Cloudflare): frozen Qwen3.8-27B / Qwen3.5-9B + a routing head and a rank-256
  LoRA; label-smoothed CE + **Brier** loss, with **RLCD = Reinforcement Learning for Calibrated
  Decisions** as a secondary objective.
- **Jev / TypeSafe** and the community deciders (kev, candigate, senna, LCT, tiny, autotrust, Vega):
  typed decision adapters; autotrust distils Jev output distributions (KL ≈ 0.017), Vega is a
  Decision-2.0 PEFT adapter (r512/α1024). Details in [LORAS.md](LORAS.md).

## See also

- [RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md) — which LoRA to use per model and the selection rule.
- [RESULTS.md](../benchmarks/RESULTS.md) — consolidated 231 / DI / latency tables.
- [LORAS.md](LORAS.md) — adapter map, provenance, and merge theory.
