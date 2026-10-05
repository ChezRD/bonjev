# DI: per-track native scores (sample-1500)

Each track's native metric (`scores.json.score`), in percent. All runs use the same sample
(sample-1500), so per-track comparison is valid. **Bold = best in the track.**

**Scope:** the DI run was only on **`ternary-bonsai-2-27b`** (Bonsai 2, Qwen3.8); the v1
`ternary-bonsai-27b` was not run on DI.

Native metric per track: ACOS is per-review F1; ToolRet and BRIGHT are nDCG@10; ForecastBench is Brier
(lower is better); the rest are accuracy or macro-F1.

**Contents**

- [Column legend](#column-legend)
- [Matrix](#matrix)
- [See also](#see-also)

## Column legend

| column | LoRA |
|---|---|
| `17_tiny` | 1.7B `m17_tiny25_single` |
| `4_b10` | 4B `B10` |
| `4_align` | 4B `all3_align` |
| `8_inv` | 8B `m8_inv_b04` |
| `8_r3` | 8B `R3` |
| `27_win` | 27B `vega_clef_plumb` |
| `27_at_half` | 27B `m27_clef_plumb_at_half` |
| `27_vega_half` | 27B `vega_clef_plumb_at_half` |
| `27_vegaffn` | 27B `clef_plumb_vegaffn` |

## Matrix

| track | 17_tiny | 27_at_half | 27_vega_half | 27_vegaffn | 27_win | 4_align | 4_b10 | 8_inv | 8_r3 |
|---|---|---|---|---|---|---|---|---|---|
| ACOS | 3.6 | 15.5 | **16.6** | 15.6 | 16.4 | 8.1 | 9.4 | 10.9 | 10.2 |
| ANLI | 18.8 | **70.1** | 67.3 | 64.5 | 64.5 | 52.1 | 59.0 | 43.8 | 37.6 |
| API-Bank | 0 | **100** | **100** | 96.5 | 96.5 | 44.8 | 48.3 | 48.3 | 55.2 |
| ARC-Challenge | 65.5 | **96.5** | **96.5** | 93.1 | 89.7 | 75.9 | 75.9 | 79.3 | 79.3 |
| ARC-Easy | 72.4 | **100** | 96.5 | **100** | **100** | 86.2 | 82.8 | 93.1 | 96.5 |
| Amazon ESCI | 16.7 | 63.7 | **68.1** | 60.0 | 63.3 | 27.3 | 43.7 | 40.8 | 23.8 |
| BANKING77 | 37.3 | **94.8** | **94.8** | **94.8** | **94.8** | 32.7 | 29.1 | 75.5 | 74.3 |
| BBH fixed-option tasks | 21.4 | **67.9** | 60.7 | **67.9** | 60.7 | 46.4 | 39.3 | 42.9 | 46.4 |
| BFCL | 41.4 | **96.5** | **96.5** | **96.5** | **96.5** | 79.3 | 82.8 | 89.7 | 82.8 |
| BPoMP | 56.1 | 94.6 | **95.4** | **95.4** | **95.4** | 60.0 | 66.9 | 65.4 | 51.5 |
| BRIGHT | 14.1 | 34.3 | 34.0 | 35.6 | **36.0** | 28.6 | 28.4 | 32.9 | 32.1 |
| CLINC150+OOS | 18.7 | 68.0 | 68.0 | 68.0 | **75.0** | 34.4 | 37.1 | 43.9 | 54.5 |
| CLadder | 53.6 | **89.3** | 85.7 | 82.1 | 82.1 | 67.9 | 67.9 | 71.4 | 67.9 |
| CRUXEval | 25.0 | 57.1 | **64.3** | 57.1 | 57.1 | 21.4 | 21.4 | 32.1 | 32.1 |
| ChessBench | 10.3 | **17.2** | 6.9 | 13.8 | **17.2** | 10.3 | **17.2** | 6.9 | 6.9 |
| ContractNLI | 20.4 | **79.1** | 78.8 | 78.4 | 78.6 | 45.9 | 44.0 | 46.5 | 50.9 |
| FinEntity | 48.6 | **91.2** | **91.2** | **91.2** | **91.2** | 68.9 | 64.2 | 85.7 | 86.3 |
| ForecastBench | 44.3 | 17.7 | 17.6 | 19.1 | **17.3** | 28.2 | 26.3 | 29.1 | 58.2 |
| GPQA Diamond | 27.6 | 44.8 | 48.3 | 44.8 | **51.7** | 27.6 | 24.1 | 27.6 | 24.1 |
| GSM8K | 22.4 | 51.7 | **53.4** | **53.4** | 51.7 | 32.8 | 34.5 | 46.6 | 46.6 |
| HLE | **21.4** | 7.1 | 10.7 | 10.7 | 10.7 | 14.3 | 10.7 | 14.3 | 10.7 |
| Habermas Machine | 25.0 | 32.1 | 39.3 | 35.7 | **42.9** | 39.3 | 32.1 | 32.1 | 32.1 |
| HellaSwag | 44.8 | **96.5** | **96.5** | **96.5** | **96.5** | 72.4 | 65.5 | 65.5 | 65.5 |
| HoVer claim verification | 67.9 | **82.1** | **82.1** | 78.6 | 75.0 | 67.9 | 67.9 | 64.3 | 67.9 |
| Home appliance simulator | 0 | **51.7** | **51.7** | **51.7** | 44.8 | 0 | 0 | 0 | 3.5 |
| Humicroedit | 31.0 | 48.3 | **55.2** | 48.3 | 48.3 | **55.2** | 51.7 | 34.5 | 48.3 |
| MMLU | 51.7 | **86.2** | 82.8 | 82.8 | 82.8 | 62.1 | 69.0 | 69.0 | 69.0 |
| MMLU-Pro | 14.3 | 50.0 | 42.9 | 50.0 | **57.1** | 17.9 | 21.4 | 28.6 | 25.0 |
| MuSR | 44.8 | **69.0** | **69.0** | **69.0** | **69.0** | **69.0** | 62.1 | 58.6 | 51.7 |
| NLI4CT | 38.8 | **91.8** | **91.8** | **91.8** | **91.8** | 75.5 | 67.1 | 75.6 | 62.0 |
| New Yorker caption matching | 21.4 | 53.6 | 46.4 | 53.6 | 53.6 | 46.4 | **60.7** | 46.4 | 42.9 |
| POP909-CL | 3.5 | **13.8** | 10.3 | 10.3 | 10.3 | 0 | 0 | 0 | 0 |
| PhishNChips phishing decisions | 67.9 | 67.9 | 67.9 | **78.6** | **78.6** | 39.3 | 39.3 | 39.3 | 39.3 |
| RAGTruth response-level hallucination | 56.4 | **81.5** | **81.5** | 78.6 | 78.6 | 71.4 | 64.3 | 71.0 | 62.9 |
| RouterBench | 65.5 | 75.9 | 75.9 | 75.9 | 75.9 | **78.0** | **78.0** | 76.7 | 75.0 |
| SATA-Bench | 0 | 27.6 | **31.0** | 27.6 | 24.1 | 13.8 | 13.8 | 20.7 | 17.2 |
| SGD/SGD-X | 37.9 | 60.9 | **69.4** | 44.4 | 44.4 | 50.6 | 63.5 | 46.7 | 53.5 |
| SimpleBench | 0 | 0 | 0 | 0 | 0 | **10.0** | **10.0** | **10.0** | **10.0** |
| ToolRet | 35.2 | 59.9 | **61.6** | 57.5 | 60.3 | 56.7 | 53.5 | 53.2 | 57.1 |
| VAST | 47.1 | 54.1 | 62.4 | 56.4 | **64.1** | 44.4 | 48.1 | 44.5 | 33.9 |
| When2Call MCQ | 57.1 | 67.9 | 67.9 | 67.9 | **71.4** | 50.0 | 60.7 | 60.7 | 57.1 |
| WinoGrande | 69.0 | **89.7** | **89.7** | **89.7** | 86.2 | 69.0 | 69.0 | 75.9 | 72.4 |
| cfcolor | 41.4 | **55.2** | 51.7 | **55.2** | **55.2** | **55.2** | 51.7 | 48.3 | 37.9 |
| iSarcasmEval | **0.1** | **0.1** | **0.1** | **0.1** | **0.1** | 0.0 | **0.1** | **0.1** | **0.1** |

## See also

- [RESULTS.md](RESULTS.md) — the consolidated `A_native` summary and the public Clef comparison.
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) — which LoRA to use per model and the selection rule.
- [README.md](README.md) — how the DI suite is fetched, sampled, and run.
