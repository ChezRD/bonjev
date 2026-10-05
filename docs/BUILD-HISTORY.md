# Build history: fragment-assembly configurations

Every training-free adapter that was assembled from source-delta fragments and scored on JevBench-231
(deterministic, `BONJEV_NO_REUSE=1`). Each row is one built adapter, with its **composition** — the source
fragments, the module blocks and the layer window — and its 231 score. These are the experiments behind
the final per-model adapters; the shipped ones are in [RECIPES.md](RECIPES.md) and
[RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md).

The `composition` column is decoded from the build name, the build logs and the archived build scripts:
fragments (source deltas), blocks (`FFN` / `attn` / `GDN`, or the finer `qkv` / `gate+up` / `down` /
`o` / `q` / `k` / `v`), the layer window (`L40-47`, `mid-16`, `late`, …), and the merge parameters
(`align`, `budget`, `pico`, `sum` / `ties` / `dare` / `slerp`, `r64` / `r128`, `λ`). Base size: `b2` =
Bonsai 2 27B, `f27` = 27B, `f9b` / `q2k` = 9B testbed, `b8` / `b4` / `b17` = 8B / 4B / 1.7B. Where a
name is a grid code (`L01`, `B10`, `R3`, `C06`, …) the composition is taken from the archived build
scripts. The module/layer footprint of every build is in the appendix at the end.

**Contents**

- [27B — Ternary Bonsai 2 (Qwen3.8-27B, hybrid 3:1)](#27b-ternary-bonsai-2-qwen3-8-27b-hybrid-3-1)
- [9B — Qwen3.5-9B testbed (not a Bonsai model)](#9b-qwen3-5-9b-testbed-not-a-bonsai-model)
- [8B — Ternary Bonsai 8B (Qwen3-8B)](#8b-ternary-bonsai-8b-qwen3-8b)
- [4B — Ternary Bonsai 4B (Qwen3.5-4B / Qwen3-4B)](#4b-ternary-bonsai-4b-qwen3-5-4b-qwen3-4b)
- [1.7B — Ternary Bonsai 1.7B (Qwen3-1.7B)](#1-7b-ternary-bonsai-1-7b-qwen3-1-7b)
- [Built compositions (modules × layers)](#built-compositions-modules--layers)

## Summary

| model | configurations | best |
|---|---:|---|
| 27B | 77 | final_27_win_125_d — 207/231 |
| 9B | 66 | f9b_merged_kv — 85/231 |
| 8B | 99 | b4_8_kev_lcta_typed_f — 160/231 |
| 4B | 90 | m4_all3 — 164/231 |
| 1.7B | 84 | ens_17_top3 — 147/231 |

## 27B — Ternary Bonsai 2 (Qwen3.8-27B, hybrid 3:1)

### `b4` — cross-size transplants (quantised)

| configuration | composition | 231 |
|---|---|---:|
| `b4_27_atmid_half` | autotrust(mid) ×0.5 | 185 |

### `cfg` — module ablations (noGdn / noAttn / FFN-only)

| configuration | composition | 231 |
|---|---|---:|
| `f27_clef_at_canopy` | clef+autotrust+canopy | 203 |
| `f27_clef_ffn_at` | clef+autotrust FFN | 202 |
| `f27_clef_noGdn` | clef | 197 |

### `delta` — raw source deltas as-is

| configuration | composition | 231 |
|---|---|---:|
| `b2_autojev` | autojev | 197 |
| `b2_novel2` | novel-2 | 160 |

### `di500` — 27B finalists re-scored under DI-500

| configuration | composition | 231 |
|---|---|---:|
| `di500_231_vegaffn` | Vega(FFN) | 206 |
| `di500_231_e27` | e27 | 201 |

### `final` — final per-model grid (default / top3 / λ1.25)

| configuration | composition | 231 |
|---|---|---:|
| `final_27_win_125_d` | Vega+clef+plumb | 207 |
| `final_27_win_125_t3` | Vega+clef+plumb | 207 |
| `final_27_win_d` | Vega+clef+plumb | 207 |
| `final_27_win_t3` | Vega+clef+plumb | 206 |

### `flags` — think / kv_f16 runtime flags

| configuration | composition | 231 |
|---|---|---:|
| `f27b_clef_thinkkv` | clef | 201 |
| `f27b_merged_at_thinkkv` | autotrust | 199 |
| `f27b_merged_thinkkv` | clef+Open-Jev | 198 |

### `gate27` — Vega-vs-clef gate on the 27B readout

| configuration | composition | 231 |
|---|---|---:|
| `gate_vega_clef` | $CLEF+$VP | 203 |
| `gate_mid16_clef256` | clef gate Lmid-16 | 202 |
| `gate_vega_alone` | Vega gate | 200 |
| `gate_gdn16_clef256` | clef gate LGDN-16 | 197 |

### `infl` — layer-window slices + amplification (λ2)

| configuration | composition | 231 |
|---|---|---:|
| `f27_clef_40_48` | clef L40-47 | 200 |
| `preserve27b_r256` | preserve r256 | 200 |
| `preserve27b_r64` | preserve r64 | 198 |
| `f27_clef_48_56` | clef L48-55 | 197 |
| `f27_clef_lambda2` | clef | 195 |
| `oj27b_late` | Open-Jev Llate | 195 |
| `oj27b_early` | Open-Jev Learly | 193 |
| `f27_clef_56_64` | clef L56-63 | 191 |

### `m27champ` — 27B champion (clef + plumb + autotrust half)

| configuration | composition | 231 |
|---|---|---:|
| `m27champ_clef_plumb_at_half` | clef+plumb+autotrust ×0.5 | 207 |

### `mod` — clef module slices (attn / ffn / gdn)

| configuration | composition | 231 |
|---|---|---:|
| `f27_clef_attn` | clef+autotrust | 197 |
| `f27_clef_ffn` | clef FFN | 197 |
| `f27_4048_ffn` | FFN L40-48 | 196 |
| `f27_4048_gdn` | GDN L40-48 | 195 |
| `f27_clef_gdn` | clef GDN | 195 |
| `f27_4048_attn` | autotrust L40-48 | 193 |

### `mrg2` — 27B pair merges + 4B pair merges

| configuration | composition | 231 |
|---|---|---:|
| `m27_clef_plumb` | clef+plumb | 205 |
| `m27_clef_canopy` | clef+canopy | 202 |
| `m27_clef_plumb_at` | clef+plumb+autotrust | 200 |
| `m27_plumb_at` | plumb+autotrust | 198 |

### `openjev` — Open-Jev on Bonsai 2 PQ2_0

| configuration | composition | 231 |
|---|---|---:|
| `b2pq2_none` | base | 195 |

### `pool` — single-source baselines per size

| configuration | composition | 231 |
|---|---|---:|
| `b2_plumb` | plumb | 201 |
| `b2_canopy` | canopy | 196 |
| `b2_simplejev` | simplejev | 195 |
| `b2_sargedev_r2` | sargedev | 180 |

### `runner` — merge-mode variants and quant testbeds

| configuration | composition | 231 |
|---|---|---:|
| `merged_b2ptq1_clef_at` | clef+autotrust | 204 |
| `merged_b2ptq1_merged27b_at` | autotrust | 200 |
| `mergevariants_mv27b_clef_at_ties` | clef+autotrust ties | 200 |
| `merged_b2ptq1_merged27b` | clef+Open-Jev | 197 |
| `mergevariants_mv27b_mean` | mean | 197 |
| `mergevariants_mv27b_slerp` | slerp | 197 |
| `mergevariants_mv27b_norm` | norm | 196 |
| `mergevariants_mv27b_ties` | ties | 196 |
| `openjev_b2ptq1_none` | base | 195 |
| `openjev_b2pq2_openjev` | Open-Jev | 191 |
| `openjev_b2ptq1_openjev` | Open-Jev | 191 |

### `scale` — single-source scaling

| configuration | composition | 231 |
|---|---|---:|
| `s27_novel2_q` | novel-2 q | 189 |
| `s27_novel2_h` | novel-2 | 185 |

### `vega27` — Vega alone, its module/layer decompositions, and clef/plumb partners

| configuration | composition | 231 |
|---|---|---:|
| `vega27_clef_plumb` | Vega+clef+plumb | 207 |
| `vega27_clef` | Vega+clef | 203 |
| `vega27_alone` | Vega | 200 |
| `vega27_ffn` | Vega FFN | 198 |
| `vega27_late` | Vega Llate | 195 |
| `vega27_base` | base | 194 |
| `vega27_gdn` | Vega GDN | 194 |

### `vega27b` — Vega core + clef/GDN/attention sub-blocks

| configuration | composition | 231 |
|---|---|---:|
| `vega27b_gdn_ccore` | Vega GDN | 200 |
| `vega27b_gdn_cffn_catt` | Vega GDN | 200 |
| `vega27b_4863_ccore` | Vega L48-63 | 199 |
| `vega27b_gdn_cffn` | Vega GDN | 199 |
| `vega27b_mid_atmid` | Vega+autotrust(mid) Lmid | 199 |

### `vega27c` — Vega layer windows (mid-16 / GDN-16) + clef r256

| configuration | composition | 231 |
|---|---|---:|
| `vega27c_all_clef256` | Vega+clef | 203 |
| `vega27c_mid16_clef256` | Vega+clef Lmid-16 | 202 |
| `vega27c_earlygdn_clef256` | Vega+clef Learly-GDN | 199 |
| `vega27c_gdn16_clef256` | Vega+clef LGDN-16 | 197 |

### `vega27d` — Vega layer windows compressed to r16

| configuration | composition | 231 |
|---|---|---:|
| `vega27d_mid_r16` | Vega Lmid | 199 |
| `vega27d_all_r16` | Vega | 197 |
| `vega27d_gdn_r16` | Vega GDN | 192 |

### `vega27e` — Vega compressed (CPA) + autotrust half

| configuration | composition | 231 |
|---|---|---:|
| `vega27e_cpa_half` | Vega+CPA ×0.5 | 207 |

### `vega27win` — winning 27B build (Vega + clef + plumb)

| configuration | composition | 231 |
|---|---|---:|
| `vega27win` | Vega | 207 |

## 9B — Qwen3.5-9B testbed (not a Bonsai model)

### `cfg` — module ablations (noGdn / noAttn / FFN-only)

| configuration | composition | 231 |
|---|---|---:|
| `f9b_clef_noAttn` | clef | 78 |
| `q2k_clef_noAttn` | clef q | 78 |

### `fixups` — 9B post-fix checks

| configuration | composition | 231 |
|---|---|---:|
| `q2k_openjev` | Open-Jev q | 82 |

### `fl4` — 9B flag checks

| configuration | composition | 231 |
|---|---|---:|
| `q2k_preserve_kv` | preserve q/k | 83 |
| `f9b_half_kv` | k ×0.5 | 81 |
| `f9b_preserve_kv` | preserve k | 79 |
| `q2k_merged_kv` | q/k | 76 |

### `flags` — think / kv_f16 runtime flags

| configuration | composition | 231 |
|---|---|---:|
| `q2k_clef_rep` | clef q | 83 |
| `f9b_merged_thinkkv` | clef-flash+Open-Jev | 82 |
| `f9b_openjev_thinkkv` | Open-Jev | 80 |
| `q2k_merged9b` | q | 77 |
| `f9b_probe_batinv` | clef-flash | 74 |
| `f9b_probe_think_kvf16` | k | 73 |
| `f9b_probe_think` | clef-flash | 71 |
| `f9b_clef_thinkkv` | clef | 69 |

### `flags2` — kv_f16 / think flag matrix

| configuration | composition | 231 |
|---|---|---:|
| `f9b_merged_kv` | k | 85 |
| `q2k_clef_kv` | clef q/k | 84 |
| `f9b_merged_think` | clef-flash+Open-Jev | 74 |
| `f9b_clef_think` | clef | 73 |
| `q2k_clef_thinkkv` | clef q | 72 |
| `f9b_openjev_think` | Open-Jev | 69 |
| `f9b_clef_kv` | clef k | 67 |
| `f9b_none_kv` | base | 66 |
| `f9b_openjev_kv` | Open-Jev k | 65 |

### `flags3` — flag repeats

| configuration | composition | 231 |
|---|---|---:|
| `f9b_merged_kv_rep` | k | 85 |
| `q2k_clef_kv_rep` | clef q/k | 84 |
| `f9b_openjev_thinkkv_rep` | Open-Jev | 80 |
| `q2k_clef_thinkkv_rep` | clef q | 72 |
| `f9b_clef_kv_rep` | clef k | 67 |

### `infl` — layer-window slices + amplification (λ2)

| configuration | composition | 231 |
|---|---|---:|
| `preserve9b_r64` | preserve r64 | 83 |
| `f9b_clef_lambda2` | clef | 77 |
| `q2k_clef9_0_16` | clef q L0-15 | 76 |
| `preserve9b_r256` | preserve r256 | 75 |
| `f9b_clef_16_32` | clef L16-31 | 71 |
| `oj9b_late` | Open-Jev Llate | 71 |
| `f9b_clef_0_16` | clef L0-15 | 69 |
| `oj9b_early` | Open-Jev Learly | 69 |
| `q2k_clef9_16_32` | clef q L16-31 | 69 |

### `mod` — clef module slices (attn / ffn / gdn)

| configuration | composition | 231 |
|---|---|---:|
| `q2k_clef9_ffn` | clef q/FFN | 81 |
| `q2k_clef9_gdn` | clef q/GDN | 78 |
| `q2k_clef9_attn` | clef+autotrust q | 72 |

### `ptq1` — 9B PTQ1_0 testbed

| configuration | composition | 231 |
|---|---|---:|
| `ptq1_9b_clef` | clef | 75 |
| `ptq1_9b_openjev` | Open-Jev | 70 |
| `ptq1_9b_merged` | clef-flash+Open-Jev | 69 |

### `runner` — merge-mode variants and quant testbeds

| configuration | composition | 231 |
|---|---|---:|
| `merged_merged9b_half` | ×0.5 | 82 |
| `openjev_q2k_openjev` | Open-Jev q | 82 |
| `mergevariants_mv9b_norm` | norm | 80 |
| `mergevariants_mv9b_slerp` | slerp | 80 |
| `mergevariants_mv9b_dare` | dare | 75 |
| `ptq1_ptq1_9b_clef` | clef | 75 |
| `ptq1_ptq1_9b_none` | base | 74 |
| `mergevariants_mv9b_ties` | ties | 73 |
| `merged_merged9b` | clef-flash+Open-Jev | 69 |
| `mergevariants_mv9b_mean` | mean | 69 |
| `ptq1_ptq1_9b_merged` | clef-flash+Open-Jev | 69 |
| `ptq1_ptq1_9b_openjev` | Open-Jev | 69 |

### `scale` — single-source scaling

| configuration | composition | 231 |
|---|---|---:|
| `s9b_clef_attn` | clef+autotrust | 78 |

### `small_now` — 9B small-build checks

| configuration | composition | 231 |
|---|---|---:|
| `u9b_clef128` | clef | 82 |
| `u9b_clef128_at9` | clef+autotrust | 71 |
| `u9b_x_clef27` | clef | 70 |
| `u9b_clef128_oj` | clef+Open-Jev | 68 |
| `u9b_x_at27` | autotrust | 58 |

### `titan2` — r64/r128/r256 q8 checks

| configuration | composition | 231 |
|---|---|---:|
| `q2k_r64q8` | q | 83 |
| `q2k_r128q8` | q | 78 |
| `q2k_r256q8` | q | 74 |
| `q2k_none` | base | 73 |

## 8B — Ternary Bonsai 8B (Qwen3-8B)

### `b4` — cross-size transplants (quantised)

| configuration | composition | 231 |
|---|---|---:|
| `b4_8_kev_lcta_typed_f` | kev+LCT+typed-decisions | 160 |

### `b5` — module splits

| configuration | composition | 231 |
|---|---|---:|
| `b5_8_inv_ffn` | FFN | 112 |
| `b5_8_inv_attn` | autotrust | 49 |

### `blk` — block (module) splits per size

| configuration | composition | 231 |
|---|---|---:|
| `blk_8_kev_lcta_q` | kev+LCT q | 157 |
| `blk_8_kev_lctf` | kev+LCT | 154 |
| `blk_8_kev_lcta_h` | kev+LCT | 139 |

### `blk2` — block splits, quantised

| configuration | composition | 231 |
|---|---|---:|
| `blk2_8_kev_lcta_typed_q` | kev+LCT+typed-decisions q | 156 |

### `bt` — 8B inv without LCT

| configuration | composition | 231 |
|---|---|---:|
| `bt_8_inv4` | kev-8b attn+kev-8b ffn+candigate→8B+senna→8B+LCT attn:0.25 | 153 |
| `bt_8_inv4_nolct` | kev-8b attn+kev-8b ffn+candigate→8B+senna→8B | 151 |

### `delta` — raw source deltas as-is

| configuration | composition | 231 |
|---|---|---:|
| `b8_lct` | LCT | 79 |
| `b8_kev_lct` | kev+LCT | 63 |

### `dmp` — base-vs-LoRA dump runs

| configuration | composition | 231 |
|---|---|---:|
| `dmp_8_inv` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 159 |
| `dmp_8_base` | base | 149 |

### `ens` — top3 ensembling

| configuration | composition | 231 |
|---|---|---:|
| `ens_8_top3_125` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 157 |
| `ens_8_top3` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 154 |
| `ns_8_kev_typed` | kev+typed-decisions | 154 |
| `ns_8_typed` | typed-decisions | 147 |

### `fd` — 8B kev λ sweep

| configuration | composition | 231 |
|---|---|---:|
| `fd_8_kev_lam15` | kev λ15 | 158 |
| `fd_8_kev_attn` | kev+autotrust | 152 |
| `fd_8_kevffn_lam15` | kev λ15 | 151 |
| `fd_8_kev_ffn` | kev FFN | 150 |

### `fin` — final check

| configuration | composition | 231 |
|---|---|---:|
| `fin_8` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 159 |

### `final` — final per-model grid (default / top3 / λ1.25)

| configuration | composition | 231 |
|---|---|---:|
| `final_8_inv_125_t3` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 157 |
| `final_8_base_t3` | base | 150 |
| `final_8_base_d` | base | 149 |

### `grid2` — 8B layer-window grid

| configuration | composition | 231 |
|---|---|---:|
| `grid2_L04` | kev-8b attn+kev-8b ffn+LCT attn (early) | 156 |
| `grid2_L08` | kev-8b attn+kev-8b ffn+LCT down (late) | 156 |
| `grid2_L12` | kev-8b attn+kev-8b ffn+typed o (late) | 156 |
| `grid2_L16` | kev-8b attn+LCT down (late)+typed o (late) | 155 |
| `grid2_L02` | kev-8b attn (early)+kev-8b ffn (early)+LCT attn (early) | 153 |
| `grid2_L11` | kev-8b attn+kev-8b ffn+typed attn (late)+typed ffn (late) | 153 |
| `grid2_L15` | kev-8b attn (late)+kev-8b ffn (late)+typed attn (late)+typed ffn (late) | 148 |
| `grid2_L13` | kev-8b attn+kev-8b ffn+LCT ffn (late) | 116 |
| `grid2_L07` | kev-8b attn+kev-8b ffn+LCT o (late) | 72 |
| `grid2_L03` | kev-8b attn+kev-8b ffn+LCT attn (late) | 69 |
| `grid2_L10` | kev-8b o (early)+kev-8b down (early)+LCT o (late) | 69 |
| `grid2_L06` | kev-8b attn (early)+kev-8b ffn (early)+LCT attn | 62 |
| `grid2_L09` | kev-8b o (late)+kev-8b down (late)+LCT o (late) | 61 |
| `grid2_L14` | LCT attn (late)+kev-8b attn+kev-8b ffn | 60 |
| `grid2_L01` | kev-8b attn (late)+kev-8b ffn (late)+LCT attn (late) | 49 |
| `grid2_L05` | kev-8b attn (late)+kev-8b ffn (late)+LCT attn | 47 |

### `grid3` — 8B module grid

| configuration | composition | 231 |
|---|---|---:|
| `grid3_s3_down` | kev-8b attn+kev-8b ffn+LCT down (late) | 154 |
| `grid3_m_attn` | kev-8b attn | 152 |
| `grid3_m_qkv` | kev-8b qkv | 152 |
| `grid3_s3_qkv` | kev-8b attn+kev-8b ffn+LCT qkv (late) | 152 |
| `grid3_m_o` | kev-8b o | 151 |
| `grid3_m_ffn` | kev-8b ffn | 150 |
| `grid3_m_gu` | kev-8b gate+up | 149 |
| `grid3_m_down` | kev-8b down | 145 |
| `grid3_s3_gu` | kev-8b attn+kev-8b ffn+LCT gate+up (late) | 99 |
| `grid3_s3_o` | kev-8b attn+kev-8b ffn+LCT o (late) | 73 |

### `inv` — 8B inv variants

| configuration | composition | 231 |
|---|---|---:|
| `inv_8` | kev-8b attn+kev-8b ffn+LCT attn:0.25 | 157 |
| `inv_8_typed` | typed-decisions | 133 |

### `iv2` — inv budget sweep

| configuration | composition | 231 |
|---|---|---:|
| `iv2_8_b04` | budget 0.4 | 159 |
| `iv2_8_b02` | budget 0.2 | 156 |

### `lam` — λ sweep per size

| configuration | composition | 231 |
|---|---|---:|
| `lam_8_klcta_125` | k | 157 |
| `lam_8_klcta_150` | k | 157 |
| `lam_8_klcta_075` | k | 148 |

### `ls` — late-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls_8_klcta_late` | k Llate layer-scale | 159 |

### `ls2` — late-12-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls2_8_klcta_late12` | k Llate-12 | 156 |

### `m` — merge modes (dare / ties)

| configuration | composition | 231 |
|---|---|---:|
| `m_8_ties` | ties | 154 |
| `m_8_dare` | dare | 152 |

### `ma` — mixed merges

| configuration | composition | 231 |
|---|---|---:|
| `ma_8_kev15_lct` | kev+LCT | 159 |
| `ma_8_kev_lct` | kev+LCT | 158 |
| `ma_8_kevfa_lct` | kev+LCT | 151 |

### `op` — λ sweep (op)

| configuration | composition | 231 |
|---|---|---:|
| `op_8_lam125` | o λ125 | 160 |
| `op_8_lam075` | o λ075 | 156 |
| `op_8_lam050` | o λ050 | 152 |

### `op2` — λ sweep (op2)

| configuration | composition | 231 |
|---|---|---:|
| `op2_8_lam175` | o λ175 | 158 |
| `op2_8_lam150` | o λ150 | 156 |
| `op2_8_lam200` | o λ200 | 147 |

### `pc` — per-source compression / pico

| configuration | composition | 231 |
|---|---|---:|
| `pc_8_cat` | cat | 158 |
| `pc_8_pico_025` | pico | 155 |
| `pc_8_pico_05` | pico | 148 |
| `pc_8_pico_10` | pico | 69 |

### `pl` — negative-weight / hybrid merges

| configuration | composition | 231 |
|---|---|---:|
| `pl_8_kev_late_lct` | kev+LCT Llate | 157 |
| `pl_8_lcta_hyb` | LCT hybrid | 157 |
| `pl_8_kev_core_lct` | kev+LCT | 156 |

### `pool` — single-source baselines per size

| configuration | composition | 231 |
|---|---|---:|
| `b8_kev8b` | kev-8b | 157 |
| `b8_none` | base | 150 |

### `rb` — raw base runs

| configuration | composition | 231 |
|---|---|---:|
| `rb_8_base` | base | 149 |

### `rc` — final check (rc)

| configuration | composition | 231 |
|---|---|---:|
| `rc_8_kevlct` | kev | 156 |

### `s5` — 8B sweep

| configuration | composition | 231 |
|---|---|---:|
| `s5_s5_d` | kev-8b attn+LCT down (late)+typed attn (late) | 156 |
| `s5_s5_c` | kev-8b o+kev-8b down+LCT down (late) | 154 |
| `s5_s5_a` | kev-8b attn+LCT down (late)+LCT qkv (late) | 138 |
| `s5_s5_b` | kev-8b attn+kev-8b ffn+LCT down (late)+LCT attn (late) | 56 |

### `s6` — 8B R-grid

| configuration | composition | 231 |
|---|---|---:|
| `s6_R3_125` | kev-8b attn+kev-8b ffn+LCT down (late) | 159 |
| `s6_R2` | kev-8b attn+kev-8b ffn+LCT attn (early)+LCT down (late)+typed o (late) | 158 |
| `s6_R1` | kev-8b attn+kev-8b ffn+LCT attn (early)+LCT down (late) | 156 |
| `s6_R4` | kev-8b attn+kev-8b ffn+LCT down (late) | 156 |
| `s6_R6` | kev-8b attn+kev-8b o+LCT down (late)+typed o (late) | 156 |
| `s6_R6_125` | kev-8b attn+kev-8b o+LCT down (late)+typed o (late) | 155 |
| `s6_R3` | kev-8b attn+kev-8b ffn+LCT down (late) | 154 |
| `s6_R5` | kev-8b attn+LCT down (late)+typed o (late) | 151 |

### `scale` — single-source scaling

| configuration | composition | 231 |
|---|---|---:|
| `s8_lct_tiny` | LCT+Tiny-Jev | 150 |

### `smalldi` — small-model DI checks

| configuration | composition | 231 |
|---|---|---:|
| `smalldi_8_e17_075` | L0-17 | 157 |

### `titan2` — r64/r128/r256 q8 checks

| configuration | composition | 231 |
|---|---|---:|
| `bonsai8_none` | base | 150 |

### `wo` — 8B wo checks

| configuration | composition | 231 |
|---|---|---:|
| `wo_8_125` | wo kev-8b o+wo kev-8b down+wo LCT o:0.25 | 157 |
| `wo_8` | wo kev-8b o+wo kev-8b down+wo LCT o:0.25 | 155 |

## 4B — Ternary Bonsai 4B (Qwen3.5-4B / Qwen3-4B)

### `.`

| configuration | composition | 231 |
|---|---|---:|
| `b4_kev4b_tq` | kev-4b | 136 |

### `b10` — 4B B10 λ sweep

| configuration | composition | 231 |
|---|---|---:|
| `b10_125` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 158 |

### `b10f` — 4B B10 budget sweep

| configuration | composition | 231 |
|---|---|---:|
| `b10f_b05_10` | budget 0.5 | 158 |
| `b10f_b03_125` | budget 0.3 | 157 |
| `b10f_b03_10` | budget 0.3 | 156 |
| `b10f_b05_125` | budget 0.5 | 155 |

### `b10r` — 4B B10 repeat grid

| configuration | composition | 231 |
|---|---|---:|
| `b10r_B10g` | kev-4b attn (early)+kev-4b ffn (early)+senna | 159 |
| `b10r_B10a` | kev-4b attn (early)+candigate+senna | 156 |
| `b10r_B10d` | kev-4b attn (e23)+kev-4b ffn (e23)+candigate+senna | 156 |
| `b10r_B10c` | kev-4b attn (e11)+kev-4b ffn (e11)+candigate+senna | 154 |
| `b10r_B10h` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna+senna (early) | 154 |
| `b10r_B10f` | kev-4b attn (early)+kev-4b ffn (early)+candigate | 153 |
| `b10r_B10b` | kev-4b ffn (early)+candigate+senna | 152 |
| `b10r_B10e` | kev-4b attn (early)+kev-4b ffn (early)+candigate (early)+senna (early) | 152 |

### `b3` — cross-size transplants (r128)

| configuration | composition | 231 |
|---|---|---:|
| `b3_4_align_r128` | align+r128 | 161 |

### `b4` — cross-size transplants (quantised)

| configuration | composition | 231 |
|---|---|---:|
| `b4_4_align_x_q` | q align | 159 |
| `b4_17_tiny_xuffn_q128` | Tiny-Jev+xuhao q | 130 |
| `b4_17_tiny_xuattn_q128` | Tiny-Jev+xuhao q | 129 |

### `b5` — module splits

| configuration | composition | 231 |
|---|---|---:|
| `b5_4_align_attn` | autotrust align | 152 |
| `b5_4_align_ffn` | FFN align | 151 |

### `blk` — block (module) splits per size

| configuration | composition | 231 |
|---|---|---:|
| `blk_4_align_mog_attn` | mogita+autotrust align | 160 |
| `blk_4_align_mog_ffn` | mogita FFN align | 160 |

### `blk2` — block splits, quantised

| configuration | composition | 231 |
|---|---|---:|
| `blk2_4_align_mog_attn_q` | mogita+autotrust q align | 160 |
| `blk2_4_align_mog_ffn_q` | mogita FFN/q align | 158 |

### `cfg` — module ablations (noGdn / noAttn / FFN-only)

| configuration | composition | 231 |
|---|---|---:|
| `b4_kevFFN_cand` | kev+candigate | 155 |

### `dmp` — base-vs-LoRA dump runs

| configuration | composition | 231 |
|---|---|---:|
| `dmp_4_align` | kev-4b+candigate+senna | 159 |
| `dmp_4_base` | base | 145 |

### `ens` — top3 ensembling

| configuration | composition | 231 |
|---|---|---:|
| `ns_4_align_mog` | mogita align | 163 |
| `ens_4_top3` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 160 |
| `ens_4_top3_125` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 160 |
| `ns_4_mogita` | mogita | 150 |

### `fin` — final check

| configuration | composition | 231 |
|---|---|---:|
| `fin_4` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 161 |

### `final` — final per-model grid (default / top3 / λ1.25)

| configuration | composition | 231 |
|---|---|---:|
| `final_4_b10_d` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 161 |
| `final_4_b10_t3` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 159 |
| `final_4_b10_125_d` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 158 |
| `final_4_b10_125_t3` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 157 |

### `flags` — think / kv_f16 runtime flags

| configuration | composition | 231 |
|---|---|---:|
| `f4b_none_think` | base | 74 |
| `f4b_kahn1_think` | Kahn1 | 69 |

### `grid4b` — 4B B-grid

| configuration | composition | 231 |
|---|---|---:|
| `grid4b_B10` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 161 |
| `grid4b_B01` | kev-4b+candigate+senna | 160 |
| `grid4b_B06` | kev-4b+candigate | 160 |
| `grid4b_B11` | kev-4b+senna ffn | 159 |
| `grid4b_B02` | kev-4b+candigate+senna+mogita | 155 |
| `grid4b_B03` | kev-4b | 155 |
| `grid4b_B08` | kev-4b+senna | 155 |
| `grid4b_B04` | candigate | 153 |
| `grid4b_B07` | candigate+senna | 152 |
| `grid4b_B09` | kev-4b attn (late)+kev-4b ffn (late)+candigate+senna | 151 |
| `grid4b_B12` | kev-4b o+kev-4b down+candigate | 151 |
| `grid4b_B14` | kev-4b+senna+mogita | 151 |
| `grid4b_B05` | senna | 148 |
| `grid4b_B13` | senna (late)+candigate | 146 |

### `infl` — layer-window slices + amplification (λ2)

| configuration | composition | 231 |
|---|---|---:|
| `kahn1_late` | Kahn1 Llate | 75 |
| `kahn1_early` | Kahn1 Learly | 71 |

### `inv` — 8B inv variants

| configuration | composition | 231 |
|---|---|---:|
| `inv_4_mog` | mogita | 153 |

### `iv2` — inv budget sweep

| configuration | composition | 231 |
|---|---|---:|
| `iv2_4_b04` | budget 0.4 | 159 |
| `iv2_4_b02` | budget 0.2 | 158 |

### `lam` — λ sweep per size

| configuration | composition | 231 |
|---|---|---:|
| `lam_4_align_150` | align | 159 |
| `lam_4_align_125` | align | 157 |
| `lam_4_align_075` | align | 153 |

### `ls` — late-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls_4_align_late` | Llate layer-scale+align | 156 |

### `ls2` — late-12-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls2_4_align_late12` | Llate-12 align | 156 |

### `m` — merge modes (dare / ties)

| configuration | composition | 231 |
|---|---|---:|
| `m_4_dare` | dare | 163 |
| `m_4_dare_ties` | dare+ties | 157 |
| `m_4_ties` | ties | 154 |

### `ma` — mixed merges

| configuration | composition | 231 |
|---|---|---:|
| `ma_4_raw` | raw | 158 |

### `mrg2` — 27B pair merges + 4B pair merges

| configuration | composition | 231 |
|---|---|---:|
| `m4_all3` | all-3 | 164 |
| `m4_kev_cand` | kev+candigate | 159 |
| `m4_kev_senna` | kev+senna | 159 |
| `m4_cand_senna` | candigate+senna | 155 |

### `op` — λ sweep (op)

| configuration | composition | 231 |
|---|---|---:|
| `op_4_lam125` | o λ125 | 163 |
| `op_4_lam075` | o λ075 | 158 |
| `op_4_lam050` | o λ050 | 153 |

### `op2` — λ sweep (op2)

| configuration | composition | 231 |
|---|---|---:|
| `op2_4_lam175` | o λ175 | 161 |
| `op2_4_lam150` | o λ150 | 158 |
| `op2_4_lam200` | o λ200 | 158 |

### `pc` — per-source compression / pico

| configuration | composition | 231 |
|---|---|---:|
| `pc_4_cat` | cat | 155 |
| `pc_4_cat_budget` | cat+budget | 155 |

### `pl` — negative-weight / hybrid merges

| configuration | composition | 231 |
|---|---|---:|
| `pl_4_neg_mogita` | mogita | 159 |
| `pl_4_align_b025` | align+budget 0.25 | 157 |
| `pl_4_align_b035` | align+budget 0.35 | 157 |

### `pool` — single-source baselines per size

| configuration | composition | 231 |
|---|---|---:|
| `b4_candigate` | candigate | 154 |
| `b4_kev4b` | kev-4b | 154 |
| `b4_senna` | senna | 152 |
| `b4_none` | base | 148 |

### `rb` — raw base runs

| configuration | composition | 231 |
|---|---|---:|
| `rb_4_base` | base | 145 |

### `rc` — final check (rc)

| configuration | composition | 231 |
|---|---|---:|
| `rc_4_align` | align | 159 |

### `runner` — merge-mode variants and quant testbeds

| configuration | composition | 231 |
|---|---|---:|
| `ptq1_ptq1_4b_none` | base | 75 |
| `ptq1_ptq1_4b_kahn1` | Kahn1 | 72 |

### `smalldi` — small-model DI checks

| configuration | composition | 231 |
|---|---|---:|
| `smalldi_4_b10_075` | kev-4b attn (early)+kev-4b ffn (early)+candigate+senna | 150 |

## 1.7B — Ternary Bonsai 1.7B (Qwen3-1.7B)

### `b3` — cross-size transplants (r128)

| configuration | composition | 231 |
|---|---|---:|
| `b3_17_tiny_r128` | Tiny-Jev r128 | 133 |
| `b3_17_tinyattn_r128` | Tiny-Jev r128 | 130 |

### `b5` — module splits

| configuration | composition | 231 |
|---|---|---:|
| `b5_17_tiny_mid` | Tiny-Jev Lmid | 127 |
| `b5_17_tiny_o` | Tiny-Jev o | 127 |
| `b5_17_inv_attn` | autotrust | 126 |
| `b5_17_tiny_up` | Tiny-Jev up | 119 |
| `b5_17_inv_ffn` | FFN | 118 |
| `b5_17_tiny_down` | Tiny-Jev down | 116 |
| `b5_17_tiny_qkv` | Tiny-Jev qkv | 115 |

### `blk` — block (module) splits per size

| configuration | composition | 231 |
|---|---|---:|
| `blk_17_attn_xuffn` | autotrust+xuhao | 131 |
| `blk_17_tiny_attn` | Tiny-Jev+autotrust | 130 |
| `blk_17_tiny_late` | Tiny-Jev Llate | 125 |
| `blk_17_tiny_ffn` | Tiny-Jev FFN | 124 |
| `blk_17_ffn_xuattn` | xuhao FFN | 120 |

### `blk2` — block splits, quantised

| configuration | composition | 231 |
|---|---|---:|
| `blk2_17_tiny_xuffn_q` | Tiny-Jev+xuhao q | 133 |
| `blk2_17_tiny_xuattn_q` | Tiny-Jev+xuhao q | 130 |
| `blk2_17_tiny_early` | Tiny-Jev Learly | 127 |

### `c06` — 1.7B candidate cuts

| configuration | composition | 231 |
|---|---|---:|
| `c12_25` | Tiny-Jev+xuhao attn+xuhao ffn | 141 |
| `c06_25` | Tiny-Jev+xuhao attn | 139 |
| `c06_20` | Tiny-Jev+xuhao attn | 138 |

### `dmp` — base-vs-LoRA dump runs

| configuration | composition | 231 |
|---|---|---:|
| `dmp_17_single` | Tiny-Jev:2.5+xuhao ffn:0.25 | 143 |
| `dmp_17_base` | base | 124 |

### `ens` — top3 ensembling

| configuration | composition | 231 |
|---|---|---:|
| `ens_17_top3` | Tiny-Jev:2.5+xuhao ffn:0.25 | 147 |
| `ns_17_tiny` | Tiny-Jev | 138 |
| `ns_17_ffn_tiny` | Tiny-Jev FFN | 128 |

### `fin` — final check

| configuration | composition | 231 |
|---|---|---:|
| `fin_17` | Tiny-Jev:2.5+xuhao ffn:0.25 | 143 |

### `grid17` — 1.7B candidate grid

| configuration | composition | 231 |
|---|---|---:|
| `grid17_C06` | Tiny-Jev+xuhao attn | 133 |
| `grid17_C12` | Tiny-Jev+xuhao attn+xuhao ffn | 132 |
| `grid17_C07` | Tiny-Jev+xuhao ffn | 130 |
| `grid17_C01` | Tiny-Jev | 126 |
| `grid17_C02` | Tiny-Jev attn | 126 |
| `grid17_C04` | Tiny-Jev attn (late)+Tiny-Jev ffn (late) | 126 |
| `grid17_C05` | Tiny-Jev attn (early)+Tiny-Jev ffn (early) | 126 |
| `grid17_C08` | Tiny-Jev attn (late)+Tiny-Jev ffn (late)+xuhao (late)+xuhao ffn (late) | 125 |
| `grid17_C10` | Tiny-Jev attn+xuhao ffn | 125 |
| `grid17_C03` | Tiny-Jev ffn | 124 |
| `grid17_C09` | Tiny-Jev attn (early)+Tiny-Jev ffn (early)+xuhao (late)+xuhao ffn (late) | 122 |
| `grid17_C11` | Tiny-Jev o+xuhao ffn | 120 |

### `inv` — 8B inv variants

| configuration | composition | 231 |
|---|---|---:|
| `inv_17` | Tiny-Jev:2.5+xuhao ffn:0.25 | 139 |

### `k06` — 0.6B-transplant checks

| configuration | composition | 231 |
|---|---|---:|
| `k06_17_sum` | 0.6B sum | 143 |
| `k06_17_inv` | 0.6B | 142 |
| `k06_17_alone` | 0.6B | 118 |

### `lam` — λ sweep per size

| configuration | composition | 231 |
|---|---|---:|
| `lam_17_tiny_150` | Tiny-Jev | 139 |
| `lam_17_tiny_125` | Tiny-Jev | 135 |
| `lam_17_tiny_075` | Tiny-Jev | 123 |

### `lam2` — 1.7B λ sweep (high)

| configuration | composition | 231 |
|---|---|---:|
| `lam2_17_tiny_200` | Tiny-Jev λ2 | 140 |
| `lam2_17_tiny_150b` | Tiny-Jev λ2 | 139 |
| `lam2_17_tiny_175` | Tiny-Jev λ2 | 138 |

### `lam3` — 1.7B λ sweep (very high)

| configuration | composition | 231 |
|---|---|---:|
| `lam3_17_tiny_250` | Tiny-Jev λ3 | 144 |
| `lam3_17_tiny_225` | Tiny-Jev λ3 | 142 |
| `lam3_17_tiny_300` | Tiny-Jev λ3 | 136 |

### `ls` — late-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls_17_tiny_late` | Tiny-Jev Llate layer-scale | 127 |

### `ls2` — late-12-layer scaling

| configuration | composition | 231 |
|---|---|---:|
| `ls2_17_tiny_late12` | Tiny-Jev Llate-12 | 136 |

### `m` — merge modes (dare / ties)

| configuration | composition | 231 |
|---|---|---:|
| `m_17_dare` | dare | 133 |
| `m_17_ties` | ties | 123 |

### `ma` — mixed merges

| configuration | composition | 231 |
|---|---|---:|
| `ma_17_tiny25_xuffn` | Tiny-Jev+xuhao | 142 |

### `pc` — per-source compression / pico

| configuration | composition | 231 |
|---|---|---:|
| `pc_17_tiny15_xuffn` | Tiny-Jev+xuhao | 141 |
| `pc_17_xu_pico` | xuhao pico | 130 |

### `pc2` — 1.7B pico variants

| configuration | composition | 231 |
|---|---|---:|
| `pc2_17_tiny15_090` | Tiny-Jev | 144 |
| `pc2_17_tiny15_xuffn_h` | Tiny-Jev+xuhao | 142 |
| `pc2_17_tiny15_110` | Tiny-Jev | 138 |
| `pc2_17_tiny175_xuffn` | Tiny-Jev+xuhao | 138 |
| `pc2_17_tiny20_xuffn` | Tiny-Jev+xuhao | 123 |

### `pc3` — 1.7B pico variants (b)

| configuration | composition | 231 |
|---|---|---:|
| `pc3_17_tiny_250b` | Tiny-Jev | 144 |
| `pc3_17_t15xuffn_166` | Tiny-Jev:2.5+xuhao ffn:0.25 | 127 |
| `pc3_17_tiny25_xuffn_h` | Tiny-Jev+xuhao | 73 |
| `pc3_17_tiny25_xuffn` | Tiny-Jev+xuhao | 70 |

### `pm` — 1.7B position cuts

| configuration | composition | 231 |
|---|---|---:|
| `pm_17_out` | o | 136 |
| `pm_17_in` | Tiny-Jev | 133 |
| `pm_17_both` | Tiny-Jev | 128 |

### `pool` — single-source baselines per size

| configuration | composition | 231 |
|---|---|---:|
| `b17_none` | base | 124 |
| `b17_xuhao` | xuhao | 122 |

### `rb` — raw base runs

| configuration | composition | 231 |
|---|---|---:|
| `rb_17_base` | base | 124 |

### `rc` — final check (rc)

| configuration | composition | 231 |
|---|---|---:|
| `rc_17_single` | Tiny-Jev:2.5+xuhao ffn:0.25 | 143 |
| `rc_17_base` | base | 124 |

### `rg` — 1.7B r128 λ checks

| configuration | composition | 231 |
|---|---|---:|
| `rg_17_r128_lam25` | r128+λ25 | 142 |
| `rg_17_r128_lam10` | r128+λ10 | 135 |

### `sg` — 1.7B single-source variants

| configuration | composition | 231 |
|---|---|---:|
| `sg_17_tiny25_xuffn_cat` | Tiny-Jev+xuhao cat | 142 |
| `sg_17_tiny25_single` | Tiny-Jev | 70 |

### `sg2` — 1.7B single-source variants (b)

| configuration | composition | 231 |
|---|---|---:|
| `sg2_17_tiny25_single` | Tiny-Jev | 143 |
| `sg2_17_ls_fix` | layer-scale | 132 |

### `sg3` — 1.7B single-source variants (c)

| configuration | composition | 231 |
|---|---|---:|
| `sg3_17_tiny25_single_s` | Tiny-Jev | 139 |
| `sg3_17_tiny25_single_a` | Tiny-Jev | 136 |

### `sg4` — 1.7B single-source f16

| configuration | composition | 231 |
|---|---|---:|
| `sg4_17_tiny25_single_f16` | Tiny-Jev f16 | 140 |

## Built compositions (modules × layers)

Coverage of every built parts dir (the actual tensors written): the layer range and the module groups
it touches. This is the concrete block/layer footprint of each assembled adapter.

| build | layers | blocks | modules |
|---|---|---|---|
| `B01` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B02` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B03` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B04` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `B05` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B06` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B07` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B08` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B09` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10_b03` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10_b05` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10a` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10b` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10c` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10d` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10e` | 0..17(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10f` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10g` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B10h` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B11` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B12` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `B13` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `B14` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C01` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C02` | 0..27(28) | attn | attn_k,attn_output,attn_q,attn_v |
| `C03` | 0..27(28) | FFN | ffn_down,ffn_gate,ffn_up |
| `C04` | 14..27(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C05` | 0..13(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C06` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C07` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C08` | 14..27(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C09` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C10` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `C11` | 0..27(28) | FFN,attn | attn_output,ffn_down,ffn_gate,ffn_up |
| `C12` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L01` | 18..35(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L02` | 0..17(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L03` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L04` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L05` | 18..35(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L06` | 0..35(20) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L07` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L08` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L09` | 18..35(18) | FFN,attn | attn_output,ffn_down |
| `L10` | 0..35(20) | FFN,attn | attn_output,ffn_down |
| `L11` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L12` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L13` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L14` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L15` | 18..35(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `L16` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `R1` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `R2` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `R3` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `R4` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `R5` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `R6` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `_at_mid` | 19..28(10) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `_at_parts` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `_at_parts_half` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `at9_parts` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `at_mid_1924` | 19..24(6) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `candigate_parts` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `canopy_parts` | 0..3(4) | FFN,GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `clef27_4048_attn_q8` | 43..47(2) | attn | attn_k,attn_output,attn_q,attn_v |
| `clef27_4048_ffn_q8` | 40..47(8) | FFN | ffn_down,ffn_gate,ffn_up |
| `clef27_4048_gdn_q8` | 40..46(6) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `clef27_40_48_q8` | 40..47(8) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef27_48_56_q8` | 48..55(8) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef27_56_64_q8` | 56..63(8) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef27_attn_q8` | 43..63(6) | attn | attn_k,attn_output,attn_q,attn_v |
| `clef27_ffn_q8` | 40..63(24) | FFN | ffn_down,ffn_gate,ffn_up |
| `clef27_gdn_q8` | 40..62(18) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `clef27b_r64` | 40..63(24) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef9b_0_16_q8` | 0..15(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef9b_16_32_q8` | 16..31(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef9b_attn_q8` | 3..31(8) | attn | attn_k,attn_output,attn_q,attn_v |
| `clef9b_ffn_q8` | 0..31(32) | FFN | ffn_down,ffn_gate,ffn_up |
| `clef9b_gdn_q8` | 0..30(24) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `clef9b_r64` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `clef_plumb_vegaffn` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `e27_at` | 0..15(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `e27_early` | 0..15(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `e27_plumb` | 0..15(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `e27_vega` | 0..15(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `g01` | 0..12(5) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `kahn1_early` | 0..15(16) | GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_alpha,ssm_beta,ssm_out |
| `kahn1_late` | 16..31(16) | GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_alpha,ssm_beta,ssm_out |
| `kahn1_parts` | 0..31(32) | GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_alpha,ssm_beta,ssm_out |
| `kev4b_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `kev4b_q3_parts` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `kev4b_q3x_parts` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `kev8b_attn` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `kev8b_e17` | 0..17(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `kev8b_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `kev8b_parts` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `lct_attn` | 34..35(2) | attn | attn_k,attn_output,attn_q,attn_v |
| `lct_ffn` | 34..35(2) | FFN | ffn_down,ffn_gate,ffn_up |
| `m17_ffn_attn25` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_ffn_attn_half` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_ffn_kev` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_ffn_tiny_h` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_tiny25_single` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_tiny_attn` | 0..27(28) | attn | attn_k,attn_output,attn_q,attn_v |
| `m17_tiny_attn_r128` | 0..27(28) | attn | attn_k,attn_output,attn_q,attn_v |
| `m17_tiny_attn_xu_ffn` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_tiny_early` | 0..13(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_tiny_ffn` | 0..27(28) | FFN | ffn_down,ffn_gate,ffn_up |
| `m17_tiny_ffn_xu_attn` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_tiny_late` | 14..27(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_kev4` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny_kev4b_tq` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny_xuattn_q` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny_xuattn_q128` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny_xuffn_q` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_x_tiny_xuffn_q128` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m17_xuhao_ffn` | 0..27(28) | FFN | ffn_down,ffn_gate,ffn_up |
| `m27_clef256_plumb_at_r128` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_at_canopy` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_canopy` | 0..63(28) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_ffn_at` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `m27_clef_noGdn` | 40..63(24) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m27_clef_plumb` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_at` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_at_half` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_at_q` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_at_ties` | 0..28(22) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_atmid` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_atmid_half` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_plumb_canopy` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_clef_simplejev` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_cp_pico64` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_cpa_pico64` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_cpa_spec128` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_cpa_spec64` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m27_plumb_at` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m4_align_mog_attn` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_align_mog_attn_q` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_align_mog_ffn` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_align_mog_ffn_q` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_all3` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_cand_senna` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_kevFFN_cand` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_kev_cand` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4_kev_senna` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_align_mogita_h` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_align` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_align015` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_align05` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_align_r128` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_do` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_palign` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_pico` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_shalf` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_x` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_all3_x_q` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_kev_cand_align` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m4b_x_mogita` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_inv_b04` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_at` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_clef` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_lct` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_lcta_h` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_lcta_q` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_lcta_typed_q` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_lctf` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_kev_typed_h` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_lcta` | 34..35(2) | attn | attn_k,attn_output,attn_q,attn_v |
| `m8_wo` | 0..35(36) | FFN,attn | attn_output,ffn_down |
| `m8_x_at27` | 10..15(6) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_x_clef27` | 22..34(13) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8_x_typed` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8b_kev_at` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8b_kev_clef` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8b_x_at27` | 11..16(6) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m8b_x_clef27` | 22..35(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `m9b_clef128` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_clef128_at9` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_clef128_oj` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_clef128_xat27` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_clef_noAttn` | 0..31(32) | FFN,GDN | attn_gate,attn_qkv,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_clef_split3` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_triple128` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_x_at27` | 9..13(5) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_out |
| `m9b_x_clef27` | 19..30(12) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m9b_x_clef27_local` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `m_attn` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `m_down` | 0..35(36) | FFN | ffn_down |
| `m_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `m_gu` | 0..35(36) | FFN | ffn_gate,ffn_up |
| `m_o` | 0..35(36) | attn | attn_output |
| `m_qkv` | 0..35(36) | attn | attn_k,attn_q,attn_v |
| `mogita_attn` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `mogita_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `mogita_parts` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `oj27b_early` | 0..39(40) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_out |
| `oj27b_late` | 40..63(24) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_out |
| `oj9_parts` | 0..31(32) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_out |
| `oj9b_early` | 0..15(16) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_out |
| `oj9b_late` | 16..31(16) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v,ssm_out |
| `parts_text` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `plumb_parts` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `preserve27b_r256` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `preserve27b_r64` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `preserve9b_r256` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `preserve9b_r64` | 0..31(32) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `s17_tiny_early_a` | 0..13(14) | attn | attn_k,attn_output,attn_q,attn_v |
| `s17_tiny_early_f` | 0..13(14) | FFN | ffn_down,ffn_gate,ffn_up |
| `s17_tiny_late_a` | 14..27(14) | attn | attn_k,attn_output,attn_q,attn_v |
| `s17_tiny_late_f` | 14..27(14) | FFN | ffn_down,ffn_gate,ffn_up |
| `s17_tiny_o` | 0..27(28) | attn | attn_output |
| `s17_xu_late` | 14..27(14) | attn | attn_k,attn_output,attn_q,attn_v |
| `s17_xu_late_f` | 14..27(14) | FFN | ffn_down,ffn_gate,ffn_up |
| `s3_down` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s3_gu` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s3_o` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s3_qkv` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s4_cand_early` | 0..17(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s4_kev_down` | 0..35(36) | FFN | ffn_down |
| `s4_kev_e11_a` | 0..11(12) | attn | attn_k,attn_output,attn_q,attn_v |
| `s4_kev_e11_f` | 0..11(12) | FFN | ffn_down,ffn_gate,ffn_up |
| `s4_kev_e23_a` | 0..23(24) | attn | attn_k,attn_output,attn_q,attn_v |
| `s4_kev_e23_f` | 0..23(24) | FFN | ffn_down,ffn_gate,ffn_up |
| `s4_kev_early_a` | 0..17(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s4_kev_early_f` | 0..17(18) | FFN | ffn_down,ffn_gate,ffn_up |
| `s4_kev_late_a` | 18..35(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s4_kev_late_f` | 18..35(18) | FFN | ffn_down,ffn_gate,ffn_up |
| `s4_kev_o` | 0..35(36) | attn | attn_output |
| `s4_senna_down` | 0..35(36) | FFN | ffn_down |
| `s4_senna_early` | 0..17(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s4_senna_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `s4_senna_late` | 18..35(18) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s5_a` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `s5_b` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `s5_c` | 0..35(36) | FFN,attn | attn_output,ffn_down |
| `s5_d` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down |
| `s_kev_down` | 0..35(36) | FFN | ffn_down |
| `s_kev_down_early` | 0..17(18) | FFN | ffn_down |
| `s_kev_down_late` | 18..35(18) | FFN | ffn_down |
| `s_kev_early_a` | 0..17(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s_kev_early_f` | 0..17(18) | FFN | ffn_down,ffn_gate,ffn_up |
| `s_kev_gu` | 0..35(36) | FFN | ffn_gate,ffn_up |
| `s_kev_late_a` | 18..35(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s_kev_late_f` | 18..35(18) | FFN | ffn_down,ffn_gate,ffn_up |
| `s_kev_o` | 0..35(36) | attn | attn_output |
| `s_kev_o_early` | 0..17(18) | attn | attn_output |
| `s_kev_o_late` | 18..35(18) | attn | attn_output |
| `s_kev_qkv` | 0..35(36) | attn | attn_k,attn_q,attn_v |
| `s_lct_down` | 34..35(2) | FFN | ffn_down |
| `s_lct_down_late` | 34..35(2) | FFN | ffn_down |
| `s_lct_gu` | 34..35(2) | FFN | ffn_gate,ffn_up |
| `s_lct_gu_late` | 34..35(2) | FFN | ffn_gate,ffn_up |
| `s_lct_late_a` | 34..35(2) | attn | attn_k,attn_output,attn_q,attn_v |
| `s_lct_late_f` | 34..35(2) | FFN | ffn_down,ffn_gate,ffn_up |
| `s_lct_o` | 34..35(2) | attn | attn_output |
| `s_lct_o_late` | 34..35(2) | attn | attn_output |
| `s_lct_qkv` | 34..35(2) | attn | attn_k,attn_q,attn_v |
| `s_lct_qkv_late` | 34..35(2) | attn | attn_k,attn_q,attn_v |
| `s_typ_down` | 0..35(36) | FFN | ffn_down |
| `s_typ_gu` | 0..35(36) | FFN | ffn_gate,ffn_up |
| `s_typ_late_a` | 18..35(18) | attn | attn_k,attn_output,attn_q,attn_v |
| `s_typ_late_f` | 18..35(18) | FFN | ffn_down,ffn_gate,ffn_up |
| `s_typ_o` | 0..35(36) | attn | attn_output |
| `s_typ_o_late` | 18..35(18) | attn | attn_output |
| `s_typ_qkv` | 0..35(36) | attn | attn_k,attn_q,attn_v |
| `sargedev_parts` | 0..63(64) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `senna_parts` | 0..35(36) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `simplejev_parts` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `tiny_attn` | 0..27(28) | attn | attn_k,attn_output,attn_q,attn_v |
| `tiny_early` | 0..13(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `tiny_ffn` | 0..27(28) | FFN | ffn_down,ffn_gate,ffn_up |
| `tiny_late` | 14..27(14) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `typed_attn` | 0..35(36) | attn | attn_k,attn_output,attn_q,attn_v |
| `typed_ffn` | 0..35(36) | FFN | ffn_down,ffn_gate,ffn_up |
| `v_4863_ccore` | 40..63(10) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_all_clef256` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_earlygdn_clef256` | 0..63(36) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_gdn16_clef256` | 0..63(51) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_gdn_ccore` | 0..54(41) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_gdn_cffn` | 0..63(51) | FFN,GDN | attn_gate,attn_qkv,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_gdn_cffn_catt` | 0..63(51) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_mid16_clef256` | 16..63(48) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `v_mid_atmid` | 19..29(11) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega27_parts` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega27_r128` | 0..56(53) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_48` | 48..48(1) | FFN,GDN | attn_gate,attn_qkv,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_63` | 63..63(1) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `vega_all_r16` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_atmid` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_attn` | 0..55(52) | GDN,attn | attn_k,attn_output,attn_q,attn_qkv,attn_v |
| `vega_clef` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_clef_plumb` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_clef_plumb_at_half` | 0..63(64) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_ffn` | 0..55(52) | FFN | ffn_down,ffn_gate,ffn_up |
| `vega_gdn` | 0..54(39) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `vega_gdn_early` | 0..14(12) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `vega_gdn_r16` | 0..54(39) | GDN | attn_gate,attn_qkv,ssm_alpha,ssm_beta,ssm_out |
| `vega_late` | 40..55(16) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_mid` | 16..39(24) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_mid2529` | 25..29(5) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `vega_mid_r16` | 16..39(24) | FFN,GDN,attn | attn_gate,attn_k,attn_output,attn_q,attn_qkv,attn_v,ffn_down,ffn_gate,ffn_up,ssm_alpha,ssm_beta,ssm_out |
| `wo_kev_down` | 0..35(36) | FFN | ffn_down |
| `wo_kev_o` | 0..35(36) | attn | attn_output |
| `wo_lct_o` | 34..35(2) | attn | attn_output |
| `xuhao17_parts` | 0..27(28) | FFN,attn | attn_k,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_up |
| `xuhao_attn` | 0..27(28) | attn | attn_k,attn_output,attn_q,attn_v |
| `xuhao_ffn` | 0..27(28) | FFN | ffn_down,ffn_gate,ffn_up |

