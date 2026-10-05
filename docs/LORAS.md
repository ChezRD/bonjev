# LoRA adapters: inventory, mass, merging, and provenance

How the third-party LoRA deltas were inventoried and combined. The recombination is **training-free**
and **data-free**: we only add, rescale, and re-factor deltas, then write one GGUF adapter.

The block/layer search (choosing modules and layers, not percentages) is in
[BLOCK-SEARCH.md](BLOCK-SEARCH.md). The final per-model recipes are in [RECIPES.md](RECIPES.md) and
[RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md).

**Contents**

- [Adapter inventory](#adapter-inventory)
- [Where the mass is](#where-the-mass-is)
- [Merging](#merging)
- [Measured results](#measured-results)
- [Provenance](#provenance)
- [Findings](#findings)
- [Research references](#research-references)
- [How the sources were trained](#how-the-sources-were-trained)
- [See also](#see-also)



## Adapter inventory


| adapter                       | base                         | who / what                               | rank / alpha      | modules                                | layers                 | `‖dW‖` | `rel` med/max |
| ----------------------------- | ---------------------------- | ---------------------------------------- | ----------------- | -------------------------------------- | ---------------------- | ------ | ------------- |
| **clef-27B** (our delta)      | Qwen3.8-27B                  | Cloudflare Clef − base                   | 256 → 64          | all 12: GDN + attn + FFN               | **40–63**              | 99.6   | 0.077 / 0.221 |
| **Vega-27B**                  | Qwen3.8-27B                  | vllm-sr/Decision-2.0-Vega-27B            | 512 / α1024 → 128 | all 12: GDN + attn + FFN               | 0–63                   | 7.2    | 0.004 / 0.009 |
| **plumb-27B**                 | Qwen3.5-27B                  | totum-labs/Qwen3.5-27B-plumb (suffix)    | 32 / α64          | all 12                                 | 0–63                   | 19.9   | 0.010 / 0.019 |
| **autotrust-27B**             | Qwen3.8-27B                  | autotrust, KL≈0.017 to Jev 1.13          | 16 / α32 (×2)     | GDN qkv/gate/out + attn + FFN          | 0–63 (mass 20–28 + 62) | 49.8   | 0.024 / 0.055 |
| **openjev-27B-v1.1**          | Qwen3.8-27B `1d4bf0f2…`      | ZefanCai Open-Jev                        | 8 / α16 (×2)      | GDN qkv/out + attn q/k/v/o, **no FFN** | 0–63                   | 56.1   | 0.044 / 0.074 |
| **simplejev-27B**             | Qwen3.8-27B                  | SimpleJev/JevAny-Qwen3.8-27B-LoRA        | 8 / α16           | all 12                                 | 0–63                   | 94.9   | 0.045 / 0.108 |
| **sargedev-r2-27B**           | Huihui-3.8-27B (abliterated) | SargeDev/Jev_Qwen3.8-27B-r2-LoRA         | 64 / α128         | FFN + attn, **no GDN**                 | 0–63                   | 29.1   | 0.016 / 0.034 |
| **canopy-27B**                | Qwen3.8-27B                  | Camellia86/Canopy-Jev-27B                | 16                | GDN + attn + FFN                       | **0–3**                | 2.8    | 0.005 / 0.012 |
| **autojev-27B** (delta)       | Qwen3.8-27B                  | denis-pplx/autojev-27b                   | 128               | 496 tensors                            | 0–63                   | 1.8    | 0.001 / 0.002 |
| **novel-2-27B** (delta)       | Qwen3.8-27B                  | aikexue170/jev-novel-2-27b-bf16          | 128               | 496 tensors                            | 0–63                   | 15.1   | 0.008 / 0.015 |
| **clef-flash-9B** (our delta) | Qwen3.5-9B                   | Cloudflare Clef-Flash − base             | 256 → 64          | all 12                                 | **0–31**               | 82.7   | 0.068 / 0.244 |
| **openjev-9B**                | Qwen3.5-9B `c2022362…`       | ZefanCai Open-Jev                        | 8 / α16 (×2)      | GDN qkv/out + attn q/k/v/o, **no FFN** | 0–31                   | 21.1   | 0.031 / 0.046 |
| **JevK5-9B** (delta)          | Qwen3.5-9B                   | alibiserikbay/JevK5-9B                   | 128               | attn + GDN, **no FFN**                 | 0–31                   | 3.8    | 0.005 / 0.025 |
| **kev-8b**                    | Qwen3-8B-Base                | jaredpalmer/kev-8b                       | 16 / α32          | attn q/k/v/o + FFN                     | 0–35                   | 14.3   | 0.007 / 0.011 |
| **Jev-LCT-8B** (delta)        | Qwen3-8B                     | CaoHaoWei/Jev-LCT-Qwen3-8B               | 128               | attn + FFN                             | **34–35**              | 250.4  | 0.528 / 0.848 |
| **kahn1-4B**                  | Qwen3.5-4B                   | Okura66 Kahn1                            | 16 / α32 (×2)     | attn + GDN, **no FFN**                 | 0–31                   | 2.5    | 0.005 / 0.008 |
| **kev-4b**                    | Qwen3-4B-Base                | jaredpalmer/kev-4b (branch `qwen3`)      | 16 / α32          | attn + FFN                             | 0–35                   | 10.8   | 0.008 / 0.013 |
| **candigate-4B**              | Qwen3-4B                     | CullenYap/CandiGate-Qwen3-4B             | 16 / α32          | attn only                              | 0–35                   | 3.4    | 0.005 / 0.008 |
| **senna-4B**                  | Qwen3-4B-Instruct-2507       | sennaLLMLearner/qwen3-4b-system-one-lora | 16 / α32          | attn + FFN                             | 0–35                   | 4.8    | 0.004 / 0.006 |
| **JevK5-4B** (delta)          | Qwen3.5-4B                   | alibiserikbay/JevK5                      | 128               | attn + GDN, **no FFN**                 | 0–31                   | 2.5    | 0.005 / 0.012 |
| **Tiny-Jev-1.7B**             | Qwen3-1.7B                   | lostargon/Tiny-Jev (full fine-tune)      | —                 | all                                    | 0–27                   | 42.0   | 0.035 / 0.069 |
| **xuhao-1.7B**                | Qwen3-1.7B                   | xuhaodev/Qwen3-1.7B-Jev                  | 16 / α32          | attn + FFN                             | 0–27                   | 1.8    | 0.001 / 0.002 |


`‖dW‖` = `‖B·A‖_F` of the effective delta, computed with the r×r identity in
`scripts/lora_block_profile.py` (PEFT adapters include the α/r factor). `rel` = `‖B·A‖_F / ‖W_base‖_F`
per tensor, measured against the base
bf16 checkpoint (median / max over tensors). Clef-Flash-9B is reported from its unquantized per-category
masses (the extracted parts are q8-only, which lose ~10% of the norm). All 22 rows are measured against the
matching base checkpoint.

## Where the mass is



### By tensor type

`‖dW‖_F` per category (our profiler):

- **clef-27B** — `ffn_gate 56.3`, `ffn_up 42.4`, `gdn_gate 37.4`, `gdn_qkv 34.8`, `ffn_down 30.1`,
`attn_q 27.7`, `gdn_out 20.4`, `attn_o 12.2`, `attn_k 5.7`, `attn_v 5.7`, `gdn_a 4.8`, `gdn_b 3.5`.
Half the mass is FFN, the other half GDN; attention is weaker.
- **clef-flash-9B** — same picture: `ffn_gate 44.3`, `ffn_up 38.0`, `gdn_qkv 31.9`, `gdn_gate 27.0`,
`ffn_down 25.1`, `attn_q 22.0`, `gdn_out 19.1`, `attn_o 10.4`, `attn_k 6.2`, `attn_v 5.9`,
`gdn_a 3.4`, `gdn_b 2.9`.
- **openjev-9B / 27B** — only `gdn_qkv` (15.3 / 39.8), `gdn_out` (10.0 / 24.8), `attn_q` (8.6 / 25.9),
`attn_o` (4.5 / 13.7), `attn_k` (2.9 / 6.6), `attn_v` (2.9 / 7.0). FFN and gates are not touched.
- **autotrust** — `ffn_gate 26.5`, `ffn_up 22.5`, `gdn_qkv 19.2`, `gdn_gate 17.4`, `ffn_down 17.0`,
`attn_q 12.1`, `gdn_out 10.4`, `attn_o 6.0`, `attn_v 3.3`, `attn_k 3.2` (the ×α/r factor included).
- **kahn1-4B** — attention / GDN projections only, no FFN.

Key: our deltas are **heavy and omnivorous** (FFN + GDN + attention, `rel` 0.077); Open-Jev is **light
and pointwise** (`rel` 0.03–0.04, no FFN); autotrust is light but omnivorous.

### By layer

The depth profile (the concrete top/low layer lists are in the full table below):

- **clef-27B** — rises toward the last layers; the cut starts exactly at 40.
- **clef-flash-9B** — monotone rise to the end, with a small bump around 18.
- **openjev-9B / 27B** — **early layers**, decaying at the end.
- **autotrust-27B** — middle (20–28) plus a bump at 62.
- **kahn1-4B** — middle, decaying at the end.


The Clef deltas push into the **end** of the stack, while Jev-like adapters (Open-Jev, Kahn1) push into
the **start/middle**. These are different edit regimes: the Clef post-train changed the output layers;
decision adapters learn to extract features earlier.

### Measured profile (all adapters)

Depth and module distribution of the mass, measured from the extracted parts / scaled PEFT adapters
(`scripts/lora_block_profile.py`). `‖dW‖` itself is in the [inventory](#adapter-inventory). The group
shares are the L2 fraction per group, so FFN² + attn² + GDN² = 1.


| adapter          | FFN / attn / GDN | top layers             | low layers             |
| ---------------- | ---------------- | ---------------------- | ---------------------- |
| clef-27B         | .77 / .31 / .56  | 40, 63, 62, 61, 41, 48 | 43, 57, 44, 45, 51, 47 |
| Vega-27B         | .77 / .32 / .56  | 63, 29, 28, 25, 26, 48 | 55, 59, 56, 57, 54, 60 |
| plumb-27B        | .79 / .29 / .54  | 24, 22, 29, 30, 28, 21 | 43, 16, 14, 42, 15, 41 |
| autotrust-27B    | .78 / .29 / .56  | 62, 25, 26, 20, 27, 28 | 55, 51, 63, 47, 57, 54 |
| openjev-27B-v1.1 | .00 / .55 / .83  | 7, 3, 15, 6, 26, 5     | 53, 56, 52, 54, 51, 55 |
| simplejev-27B    | .81 / .28 / .52  | 39, 20, 41, 6, 36, 25  | 56, 61, 51, 53, 59, 57 |
| sargedev-r2-27B  | .94 / .35 / .00  | 59, 61, 57, 60, 62, 3  | 36, 38, 37, 28, 25, 40 |
| canopy-27B       | .79 / .34 / .51  | 1, 0, 2, 3             | 3, 2, 0, 1             |
| autojev-27B      | .86 / .27 / .44  | 30, 31, 29, 28, 27, 32 | 63, 58, 60, 57, 61, 56 |
| novel-2-27B      | .83 / .29 / .48  | 62, 59, 2, 3, 61, 0    | 38, 7, 37, 6, 40, 8    |
| clef-flash-9B    | .75 / .33 / .57  | 30, 31, 29, 18, 23, 17 | 11, 15, 0, 12, 10, 13  |
| openjev-9B       | .00 / .50 / .87  | 6, 5, 9, 8, 7, 3       | 27, 20, 21, 25, 24, 23 |
| JevK5-9B         | .00 / .45 / .89  | 12, 13, 9, 10, 8, 17   | 31, 30, 23, 27, 26, 24 |
| kev-8b           | .87 / .50 / .00  | 6, 9, 10, 11, 12, 7    | 31, 32, 30, 27, 28, 34 |
| Jev-LCT-8B       | .80 / .60 / .00  | 34, 35                 | 35, 34                 |
| kahn1-4B         | .00 / .48 / .88  | 13, 14, 10, 9, 12, 8   | 31, 30, 29, 25, 28, 27 |
| kev-4b           | .85 / .53 / .00  | 11, 10, 6, 12, 7, 9    | 30, 32, 28, 27, 31, 29 |
| candigate-4B     | .00 / 1.00 / .00 | 5, 15, 7, 8, 0, 12     | 28, 26, 27, 23, 25, 24 |
| senna-4B         | .85 / .52 / .00  | 4, 3, 12, 11, 6, 10    | 35, 34, 26, 28, 25, 32 |
| JevK5-4B         | .00 / .45 / .89  | 13, 14, 10, 12, 8, 9   | 31, 26, 25, 27, 23, 24 |
| Tiny-Jev-1.7B    | .86 / .52 / .00  | 9, 6, 1, 7, 3, 4       | 25, 23, 20, 22, 21, 27 |
| xuhao-1.7B       | .84 / .54 / .00  | 8, 7, 9, 10, 12, 11    | 21, 19, 16, 20, 15, 24 |


Reading:

- **Depth regimes.** The Clef deltas sit in the late window (40–63); the Jev-like qwen3 deltas are
early-weighted (kev, candigate, senna, Tiny-Jev, xuhao all peak in the first third).
- **Module.** clef / plumb / autotrust / simplejev / Vega are FFN-dominant with a real GDN share;
Open-Jev, Kahn1 and JevK5 are GDN + attention with **no FFN**; the qwen3 small deltas are FFN +
attention with **no GDN** (qwen3 has no GDN blocks); sargedev has no GDN either.
- **Amplitude.** Jev-LCT-8B is by far the heaviest (`‖dW‖` 250, `rel` 0.53–0.85): a cross-base
post-train delta, not a task delta.
- **Relative strength.** `rel` separates the regimes more sharply than `‖dW‖`. The Clef post-trains edit
the base hardest (`rel` 0.077 / 0.068); the Jev decision adapters are 10–100× weaker (Open-Jev
0.031–0.044, autotrust 0.024, simplejev 0.045, JevK5 0.005, Kahn1 0.005, xuhao 0.001), and Vega /
autojev are near-zero (0.004 / 0.001) — they move the model least.



## Merging



### Can adapters be merged

Yes, if the base matches. All deltas live in the same weight space, so

```
dW_merged = Σ_i scale_i · B_i · A_i        (scale = alpha/rank)
```

The result is re-factored into a LoRA of the desired rank with randomized SVD:

```
dW = U·S·Vh  →  lora_b = U·S (out, r),  lora_a = Vh (r, in)
```

The rank of the sum is at most the sum of the ranks (Clef delta 256 + Open-Jev 8 = 264), so merging
into 64 is an SVD projection of the sum onto rank 64. For light add-ons (Open-Jev, autotrust) the loss
is small; for two heavy deltas at once it is noticeable.

Limitations:

- Only adapters of **one base** can be merged (9B with 9B, 27B with 27B). Kahn1-4B is incompatible with
anything but Qwen3.5-4B.
- llama.cpp can hold several adapters at once (`llama_set_adapters_lora` with a list of scales), but
our shim loads one, so we do an offline merge into one GGUF.



### Merge modes

The first option is **addition** (`sum`, task arithmetic): `ΔW = s₁·ΔW₁ + s₂·ΔW₂`, then SVD to rank 64.
This is not averaging: both deltas enter in full.


| mode    | what it does                                                      | why                                     |
| ------- | ----------------------------------------------------------------- | --------------------------------------- |
| `sum`   | `Σ sᵢ·ΔWᵢ`                                                        | full contribution of both deltas        |
| `mean`  | `(ΔW₁+ΔW₂)/2`                                                     | an averaged variant: halves both        |
| `norm`  | normalize each delta to the mean `‖ΔW‖`, then add                 | equal vote regardless of scale          |
| `ties`  | trim the smallest 80%, majority sign, average over agreeing signs | removes sign conflicts                  |
| `dare`  | random drop of 50% with `1/(1−p)` rescale, then add               | thins the overlap                       |
| `slerp` | spherical interpolation of the two deltas (t=0.5)                 | smooth blend without inflating the norm |




### Assembled adapters

Rank 64, `q8_0`, built locally (merge scripts + GGUF writer); they are not in the repository.


| file                                            | composition                        | method | `‖dW‖` | tensors |
| ----------------------------------------------- | ---------------------------------- | ------ | ------ | ------- |
| `merged9b_clefflash64_openjev8_q8.gguf`         | clef-flash-9B + openjev-9B         | sum    | 77.0   | 248     |
| `merged9b_clefflash64_openjev8half_q8.gguf`     | clef-flash-9B + 0.5·openjev-9B     | sum    | 74.9   | 248     |
| `merged9b_clefflash64_openjev8_mean_q8.gguf`    | same                               | mean   | 65.8   | 248     |
| `merged9b_clefflash64_openjev8_norm_q8.gguf`    | same                               | norm   | 75.9   | 248     |
| `merged9b_clefflash64_openjev8_ties_q8.gguf`    | same                               | ties   | 69.7   | 248     |
| `merged9b_clefflash64_openjev8_dare_q8.gguf`    | same                               | dare   | 77.6   | 248     |
| `merged9b_clefflash64_openjev8_slerp_q8.gguf`   | same                               | slerp  | 69.7   | 248     |
| `merged27b_clef64_openjev8_q8.gguf`             | clef-27B + openjev-27B             | sum    | 109.0  | 286     |
| `merged27b_clef64_openjev8_autotrust16_q8.gguf` | clef-27B + openjev-27B + autotrust | sum    | 119.2  | 436     |
| `merged27b_clef64_autotrust16_q8.gguf`          | clef-27B + autotrust               | sum    | 105.2  | 436     |
| `merged27b_clef64_openjev8_mean_q8.gguf`        | clef-27B + openjev-27B             | mean   | 97.7   | 286     |
| `merged27b_clef64_openjev8_norm_q8.gguf`        | same                               | norm   | 108.1  | 286     |
| `merged27b_clef64_openjev8_ties_q8.gguf`        | same                               | ties   | 102.0  | 286     |
| `merged27b_clef64_openjev8_slerp_q8.gguf`       | same                               | slerp  | 101.6  | 286     |
| `merged27b_clef64_autotrust16_ties_q8.gguf`     | clef-27B + autotrust               | ties   | 80.1   | 436     |




## Measured results


| configuration                                        | status                                                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| clef-27B r64-q8 on Bonsai 2 27B (PQ2_0 / PTQ1_0)     | **201/231** (default); PTQ1_0 sweep: best 202, oracle 206 (r64) / 207 (r128)           |
| clef-flash-9B r64-q8 on Qwen3.5-9B Q2_K              | **83/231** (base 73, +10)                                                              |
| clef-flash-9B r64-q8 on Qwen3.5-9B PTQ1_0            | **75/231** (base 74, +1)                                                               |
| openjev-9B r8-f16 on Qwen3.5-9B Q2_K / PTQ1_0        | **82/231** (+9) / **69/231** (−5)                                                      |
| openjev-27B r8-f16 on Bonsai 2 PQ2_0 / PTQ1_0        | **191/231** on both (base 195, −4)                                                     |
| Kahn1 r16-f16 on Qwen3.5-4B PTQ1_0                   | **72/231** (base 75, −3)                                                               |
| merged9b (clef-flash + openjev, r64-q8) on 9B PTQ1_0 | sum 69 / mean 69 / ties 73 / dare 75 / norm 80 / slerp 80 / half 82 (base 74, clef 75) |
| merged27b (clef + openjev, r64-q8) on Bonsai2 PTQ1_0 | sum 197 / mean 197 / slerp 197 / norm 196 / ties 196 (base 195, clef 201)              |
| **clef + autotrust (sum, r64-q8) on Bonsai2 PTQ1_0** | **204/231**                                                                            |
| `think+kv_f16` flags on 9B PTQ1_0                    | clef 69 / openjev 80 / merged 82 (default 75/69/69) — the ranking flips                |
| Q2_K: merged9b 77, clef repeat 83                    | the +10 anomaly reproduces; merged is below both singles                               |




These are the headline results. The complete grid of historical fragment-assembly configurations — the
composition (source fragments, blocks, layers, merge) and 231 score of every build — is in
[BUILD-HISTORY.md](BUILD-HISTORY.md).

## Provenance



### Sources and extraction

The `method` column uses a few terms:

- **delta** — the weight-space difference `ΔW = W_ft − W_base`, re-factored as a low-rank LoRA
  (`lora_a` / `lora_b`) so it can be merged.
- **PEFT adapter** — Hugging Face's Parameter-Efficient Fine-Tuning format
  (`adapter_model.safetensors`): stores `lora_A` / `lora_B` plus `alpha` and `rank`; the effective
  delta is `(α/r)·B·A`, so the α/r factor is folded in on conversion.
- **rank (r)** — the LoRA rank: how many directions the delta can use. r64 / r128 / r256 are the rank
  the delta is re-factored into after extraction.
- **suffix adapter** — the plumb/2 on-disk layout (`suffix_adapter.safetensors`), a different format
  that is converted (with a `×2` factor) before use.
- **slice** — a partial delta cut by a `(layer, module)` mask (e.g. attention-only, FFN-only), not a
  full-model delta.
- **merged fine-tune delta** — a delta taken from a full fine-tune of an already-merged model
  (`extract_delta.py`).

| adapter / delta     | source                                                                                                             | diff base / revision                       | method                                                                      | license    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------- | ---------- |
| clef-27B            | `Cloudflare/clef` (HF)                                                                                             | Qwen/Qwen3.8-27B                           | post-train delta: `extract_clef.py`, r256; only layers 40–63 changed        | Apache-2.0 |
| clef-flash-9B       | `Cloudflare/clef-flash` (HF)                                                                                       | Qwen/Qwen3.5-9B                            | same, r256; layers 0–31                                                     | Apache-2.0 |
| openjev-9B          | `ZefanCai/Open-Jev-9B`, rev `47e9668`                                                                              | Qwen/Qwen3.5-9B                            | PEFT adapter, `peft_to_parts.py`, ×α/r=2                                    | Apache-2.0 |
| openjev-27B-v1.1    | `ZefanCai/Open-Jev-27B-v1.1`, rev `28cf730`                                                                        | Qwen/Qwen3.8-27B                           | PEFT, ×α/r=2                                                                | Apache-2.0 |
| autotrust-27B       | `autotrust/JEV-27B`, rev `962701f`                                                                                 | Qwen/Qwen3.8-27B                           | PEFT (`adapter/`), ×α/r=2                                                   | Apache-2.0 |
| kahn1-4B            | `Okura66/Kahn1-Qwen3.5-4B-LoRA`, rev `c793a22`                                                                     | Qwen/Qwen3.5-4B                            | PEFT, ×α/r=2                                                                | Apache-2.0 |
| kev-8b              | `jaredpalmer/kev-8b`                                                                                               | Qwen/Qwen3-8B-Base                         | PEFT, ×α/r=2                                                                | Apache-2.0 |
| kev-4b@qwen3        | `jaredpalmer/kev-4b`, branch `qwen3`                                                                               | Qwen/Qwen3-4B-Base                         | PEFT, ×α/r=2                                                                | Apache-2.0 |
| candigate-4B        | `CullenYap/CandiGate-Qwen3-4B`                                                                                     | Qwen/Qwen3-4B                              | PEFT, ×α/r=2                                                                | Apache-2.0 |
| senna-4B            | `sennaLLMLearner/qwen3-4b-system-one-lora`                                                                         | Qwen/Qwen3-4B-Instruct-2507                | PEFT                                                                        | Apache-2.0 |
| xuhao-1.7B          | `xuhaodev/Qwen3-1.7B-Jev` (`adapter/`)                                                                             | Qwen/Qwen3-1.7B                            | PEFT                                                                        | Apache-2.0 |
| simplejev-27B       | `SimpleJev/JevAny-Qwen3.8-27B-LoRA`                                                                                | Qwen/Qwen3.8-27B                           | PEFT (r8/α16)                                                               | Apache-2.0 |
| sargedev-r2-27B     | `SargeDev/Jev_Qwen3.8-27B-r2-LoRA`                                                                                 | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | PEFT (r64/α128)                                                             | Apache-2.0 |
| plumb-27B           | **plumb/2** format (`suffix_adapter.safetensors`); public repo `totum-labs/Qwen3.5-27B-plumb` (org was not pinned) | Qwen/Qwen3.5-27B                           | suffix adapter r32/α64, ×2 on conversion                                    | Apache-2.0 |
| canopy-27B          | custom **Canopy-Jev-27B** (24 MB); public repo `Camellia86/Canopy-Jev-27B` (org was not pinned)                    | Qwen/Qwen3.8-27B rev `1d4bf0f2`            | custom A/B, layers 0–3 only, variant `rl64_weight01_step_1200`              | Apache-2.0 |
| JevK5-4B / JevK5-9B | `alibiserikbay/JevK5`, `alibiserikbay/JevK5-9B`                                                                    | Qwen/Qwen3.5-4B / 9B                       | delta from merged-FT: `extract_delta.py`, r128, 152 tensors (FFN untouched) | Apache-2.0 |
| Jev-LCT-8B          | `CaoHaoWei/Jev-LCT-Qwen3-8B`                                                                                       | Qwen/Qwen3-8B                              | delta r128, 14 tensors (layers 34–35)                                       | Apache-2.0 |
| autojev-27B         | `denis-pplx/autojev-27b`                                                                                           | Qwen/Qwen3.8-27B                           | delta r128, 496 tensors, `‖dW‖=1.8`                                         | Apache-2.0 |
| novel-2-27B         | `aikexue170/jev-novel-2-27b-bf16`                                                                                  | Qwen/Qwen3.8-27B                           | delta r128, 496 tensors, `‖dW‖=15.1`                                        | Apache-2.0 |


All source adapters and base models are **Apache-2.0**. The full list (including the experimental
sources) and the notices (research-only `HopitAI/hopper`, non-commercial `dhtocks/malkuth-4b`) are in
[ATTRIBUTION.md](ATTRIBUTION.md).

Extraction methods:

- **Delta from a merged fine-tune** — `scripts/extract_delta.py <base> <ft> --rank R`: fp32 tensor
diff, bit-identical tensors skipped; 2D weights → randomized SVD (`lora_a` / `lora_b`); norm vectors,
`A_log`, bias, conv1d are metrics only; canonical HF → GGUF name mapping (prefixes
`model.language_model.` / `backbone.`, sharded index). Output: `parts_text/` + `metrics.jsonl`.
- **Clef deltas** — `extract_clef.py` (same algorithm, r256, env `CLEF_`*; the vision part goes to
`parts_visual`, text modules to `parts_text`).
- **PEFT adapters** — downloaded directly
(`https://huggingface.co/<repo>/resolve/<rev>/adapter_model.safetensors` + `adapter_config.json`),
converted by `peft_to_parts.py` (folds α/r) into `parts/`; GGUF adapters are written by
`gguf_lora_writer.py`.



## Findings

- **27B is a hybrid 3:1 stack** (48 GDN + 16 full-attention blocks; full = 3, 7, …, 63). The GDN state
propagates through the whole stack, so early edits pass through the entire recurrence — early-layer
deltas are risky, late layers work. See [BLOCK-SEARCH.md](BLOCK-SEARCH.md).
- **Two depth regimes.** Late layers improve the exam format / 231; **early layers improve
generalization (DI)** — the 4B `B10` result. For clef-27B the core is layers 40–47; 56–63 is harmful
on its own.
- **FFN carries most of the Clef signal.** On 9B Q2_K the FFN part gives +8 of +10; on 27B FFN and
attention share (+2 each), GDN is zero. A "universal" delta must include FFN; GDN is optional.
- **Middle + FFN partners.** autotrust (mass 20–28 + 62) is clef's best partner; mid adapters without
FFN (Kahn1) are a minus.
- **Merge rule.** Sum over **agreeing** deltas; a **weight** (0.5 or 0.25) for conflicting ones; concat
only for conflicting components. A plain sum "for everything" is not good.
- **Multi-adapter is additive but not better.** The runtime `BONJEV_LORA="p1,s1;p2,s2"` path attaches
up to 8 deltas without rank truncation; tested composites matched or fell just below the baked
adapters (1.7B 142 < 144, 8B 159 < 160, 4B 158 < 161). Ship one adapter per model.
- **Weights beat modes.** On 9B PTQ1_0, `0.5·openjev` gives +13 over the plain sum. The third-component
dose is an inverted U: 0.25 → 204, **0.5 → 206**, 1.0 → 200.
- **Rank policy.** r128 is primary; a single needs only r64 (201 at r64 = r128 = r256); r256 does not
grow. Rank does not heal a conflict — you need a direction budget, not volume.
- **Amplification (λ>1)** works only on a full fine-tune delta (1.7B `tiny` λ=2.5). On Clef, λ=2 drops
the 27B score from 201 to 195: the delta is already at its optimal scale.
- **Cross-base late deltas are destructive.** Jev-LCT (Qwen3-8B → 8B, layers 34–35) scores 79/231 on
Bonsai-8B (base 150). Late layers are output-specific to a base. A small scale removes the harm but
creates no benefit.
- **Cross-size transplants fail** (27B→8B, kev8b→4B, 4B→8B, 0.6B→1.7B — 4 failures).
- **Quant dependence.** The 9B PTQ1_0 test bed is damaged: from the same BF16 GGUF, Q4_K_M scores base
184 and clef r64 gives no gain (184 → 184), while PTQ1_0 scores 74/75 and Q2_K 73/83. Measure the 9B
line only on a healthy quant.
- **Empty findings.** `bonzi-27b-*-jev` has no weights; `sgoedecke/system-one` is a method, not weights.



## Research references

Data-free merging methods that informed `scripts/spectral_merge3.py`.

### Projections and orthogonalization


| method                        | idea                                                                                                                | what we take                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **AlphaMerging** (CVPR 2026)  | move each task vector into the orthogonal complement of the other tasks' subspaces; compatible with TA/DoGE/WUDI    | pre-merge projection for conflicting pairs (openjev) and the triple (`--do`) |
| **PACT** (arXiv 2606.18627)   | find a task's load-bearing subspace and cut foreign projections with an "orthogonal shield"; randomized-SVD variant | protect the clef load-bearing weights when adding plumb/autotrust            |
| **DOP** (NeurIPS 2025)        | data-free double orthogonal projection for continual merging                                                        | projection onto `span(τ)` when adding the third component                    |
| **Iso-C** (arXiv 2502.04959)  | shared + task-specific subspaces, spectrum alignment; an alignment metric predicts quality                          | align the spectrum when plumb dominates; use as a diagnostic                 |
| **OSRM** (ACL 2025)           | constrain the LoRA subspace *before* fine-tuning (needs data)                                                       | not applicable: we do not train adapters                                     |
| OrthoMerge (arXiv 2602.05943) | merge on a manifold, neuron-level conflict voting                                                                   | a TIES replacement idea                                                      |




### Shared basis and rank budget


| method                              | idea                                                                                         | what we take                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **KnOTS** (arXiv 2410.19735)        | joint-SVD of adapters into a shared space, merge there; up to +4.3%                          | the align path: `spectral_merge3.py --align`                                |
| **Net Utility** (arXiv 2609.22237)  | data-free "usefulness minus interference" score, global selection of K under a budget        | `--budget`; later a per-layer budget (40–47 more, 56–63 less)               |
| **Pico** (arXiv 2604.16826)         | the main interference source is the output matrix B; calibrate B + rescale; +3.4–8.3 pp      | `--pico`; explains why 0.5·autotrust heals the triple                       |
| **DO-Merging** (arXiv 2505.15875)   | decouple magnitude and direction; orthogonal add-on                                          | `--do`: normalization + projection; candidate for LCT (250) and novel2 (15) |
| **CtM** (arXiv 2606.03723)          | compress to rank r *before* merging: shared r-subspaces, merge in r×r coordinates            | a strict r64 triple                                                         |
| **TSPA** (ACL Findings 2026)        | two-stage parameter alignment, quadratic→linear cost                                         | r128 merges                                                                 |
| **CUR+SVD** (ACL Findings 2026)     | SVD captures the shared, CUR the task-specific/local                                         | keep tail layers 56–63 via a CUR part instead of dropping them              |
| **Mergeability** (arXiv 2606.19549) | per-layer Frobenius-cos, over-amplification, sign conflicts; predict conflict before merging | diagnose the pairs (clef/plumb/autotrust/openjev/canopy)                    |
| SSR-Merge (arXiv 2606.10617)        | OLS-optimal "signal allocation" instead of merging                                           | coefficient fitting on a dev split                                          |




### Depth and layers


| method                             | idea                                                                                        | what we take                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **LiNeS** (ICLR 2025)              | λ(l) grows linearly with depth: early ≈0, late ≈1; plug-and-play                            | a layer ramp for clef/openjev instead of hard windows; consistent with the 40–47 core |
| **FedTreeLoRA** (arXiv 2603.13282) | when sources diverge, deep layers are a negative-transfer zone; the "safe depth" is dynamic | explains the LCT late-delta failure; damp the late part in cross-base merges          |
| **LOT Merging** (NeurIPS 2025)     | feature drift grows with depth; layer-optimal merge in closed form                          | a closed form per layer with statistics                                               |
| **CoM** (arXiv 2508.21421)         | merging covariate shift: sequential layer merging with updated statistics                   | explains why independent per-layer merging diverges                                   |
| **aTLAS** (NeurIPS 2024)           | learnable anisotropic per-block coefficients, concentrated on deep layers                   | learned λ on a dev split, initialized from LiNeS                                      |
| DHF (IJCAI 2026), AdaMerging       | gradient optimization of intra/inter-block coefficients                                     | compare with the grid + dev                                                           |




### Quants and ternary bases


| method                               | idea                                                                                                               | what we take                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **ternative** (GitHub 2026)          | engine for I2_S + LoRA: merging a LoRA into I2_S with requant zeroes the delta; keep the adapter separately in F32 | confirms our runtime path; a ban on "quantize after merge" for a ternary base  |
| **LoTA-QAF** (NeurIPS 2025)          | ternary adaptation inside the quantization grid; lossless merge; +5.14% at 2-bit                                   | the key to the 9B PTQ1_0 anomaly: a 16-bit adapter + low-bit base loses signal |
| **LoraQuant** (arXiv 2510.26690)     | SVD reparametrization → mixed precision: important directions more precise, the rest 1-bit                         | a recipe for PTQ5 / self-quant without imatrix                                 |
| **QA-LoRA** (ICLR 2024)              | group-wise quantization makes the adapter mergeable in low-bit losslessly                                          | a fallback if runtime adapters are undesirable                                 |
| Unified GGUF eval (arXiv 2601.14277) | quality/memory tables for Q3–Q8 on Llama-3.1-8B                                                                    | a reference for choosing quants for 9B/27B                                     |


TQ1_0 / TQ2_0 bases are broken on the fork (no CUDA kernels, CPU fallback) and are excluded.

### Cross-size and distillation


| method                               | idea                                                                                | what we take                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Cross-LoRA** (arXiv 2508.05232)    | data-free SVD base alignment + Frobenius projection of ΔW, ~20 min on 8 GB          | 27B→9B (all modules including GDN match); for qwen3 smalls only attn + FFN |
| **ULD** (arXiv 2402.12030)           | Wasserstein logit distillation across different tokenizers                          | 27B→9B/8B, 4B→1.7B: Qwen3 and Qwen3.5 have different vocabularies          |
| **Minitron** (arXiv 2407.14679)      | logit-only for small depth cuts, +intermediate for large; prune to the nearest size | 1.7B←4B (28 vs 36 layers) and 4B←27B: logit first                          |
| **BitDistill** (arXiv 2510.13998)    | distill straight to 1.58-bit: SubLN + attention-KD + CPT                            | the "native" path for ternary Bonsai                                       |
| Token-scaled logit KD (NeurIPS 2023) | KD for ternary GLMs: mask confident tokens, scale per token                         | a loss for ternary distillation                                            |
| AdaKD (AAAI 2026)                    | adaptive temperature and per-token focus                                            | if we run KD                                                               |




## How the sources were trained


| source                             | training method                                                                                                                             | key details                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Jev (TypeSafe)**                 | **RLCD = Reinforcement Learning for Calibrated Decisions** — their own method                                                               | the goal is calibration: stated probability = real frequency; typed outputs, a parallel sampler                     |
| **Clef / Clef-flash (Cloudflare)** | frozen Qwen3.8-27B / Qwen3.5-9B + a **routing head + rank-256 LoRA**; label-smoothed CE + **Brier loss**; **RLCD as a secondary objective** | synthetic data with field/prompt/schema permutations; prefill-only, non-autoregressive, two-stage attention routing |
| **autojev-27b (denis-pplx)**       | **not RLCD**: full-weight SFT, 73k examples, 286 steps, cross-entropy + separate temperature calibration                                    | "inspired by Jev"                                                                                                   |
| **autotrust JEV-27B**              | **Blocks of Experts**: frozen Qwen3.8-27B + a 108.9M decision block (0.4%), **distilling Jev 1.13 output distributions** (KL 0.017)         | the generation path is untouched (HumanEval byte-identical)                                                         |
| **autotrust JEV-9B**               | distillation from `SargeDev/jev-distill-corpus-v3` (Jev 1.13 outputs), KL 0.021                                                             | ~4,750 steps                                                                                                        |


Note: academic RLCD (arXiv 2307.12950, ICLR 2024) is **different** — Reinforcement Learning from
**Contrastive Distillation** (contrastive prompts → preference pairs → PPO). At TypeSafe and
Cloudflare, RLCD means Reinforcement Learning for **Calibrated Decisions**.

Clef is "a frozen backbone + an r256 LoRA + a head", i.e. exactly our runtime approach (an adapter on
top of a quantized base); the calibration objectives (Brier + RLCD) explain why its delta is
"task-like" and merges well.

## See also

- [BLOCK-SEARCH.md](BLOCK-SEARCH.md) — the block/layer study and glossary.
- [RECIPES.md](RECIPES.md) — how each final LoRA was produced and how to reproduce it.
- [RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md) — which LoRA to use per model and the selection rule.
- [RESULTS.md](../benchmarks/RESULTS.md) — consolidated 231 / DI / latency tables.

