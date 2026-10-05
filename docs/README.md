# Documentation

Methodology and internals: how the LoRA adapters are built, combined, and studied. Benchmark results
and the exam harness live in [`../benchmarks/`](../benchmarks/README.md); the project overview is in
the [root README](../README.md).

| file | what it is |
|---|---|
| [LORAS.md](LORAS.md) | adapter inventory, where the delta mass is, merge theory and modes, provenance, findings, and the research survey |
| [RECIPES.md](RECIPES.md) | how each final LoRA was produced and how to reproduce it from source deltas |
| [BLOCK-SEARCH.md](BLOCK-SEARCH.md) | the block/layer study (modules × layers), the glossary, and the results that led to each recipe |
| [TRAINING-FREE-HOOKS.md](TRAINING-FREE-HOOKS.md) | what can be borrowed from the `strands-decider` model without training (calibration, prompt styles) |
| [ENVIRONMENT.md](ENVIRONMENT.md) | every `BONJEV_*` variable, its purpose, and its measured effect on 231 relative to the base model |
| [ATTRIBUTION.md](ATTRIBUTION.md) | every third-party model used to build the adapters, with its license and notices |

## Where things live

| path | contents |
|---|---|
| [`../benchmarks/`](../benchmarks/README.md) | the JevBench-231 exam and the Decision Index runs, plus the results and selection rule |
| `../scripts/` | builders and runners: `build_lora.sh`, `fetch_sources.py`, `eval_exam.sh`, `di_run.sh` |
| `../src/` | the Rust decision engine (single forward pass over option-token logits) |

## See also

- [RECOMMENDATIONS.md](../benchmarks/RECOMMENDATIONS.md) — which LoRA to use per model and the
  selection rule.
- [RESULTS.md](../benchmarks/RESULTS.md) — consolidated 231 / DI / latency tables.
