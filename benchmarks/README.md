# Benchmark notes

How the published numbers are produced. There are two suites: the public **JevBench-231** exam and the
external **Decision Index** (DI). Per-model guidance is in `RECOMMENDATIONS.md`; the consolidated
tables are in `RESULTS.md`; how each LoRA was built is in [`../docs/RECIPES.md`](../docs/RECIPES.md).

## Public exam (JevBench-231)

```bash
python3 benchmarks/jevbench_exam.py
```

The script downloads the pinned JevBench public splits (`easy`, `original`, `hard`, 231 tasks, commit
`3749b4fc`) into `benchmarks/cache/jevbench/` and scores them with `POST /v1/systemone`. A file whose
sha256 already matches is not downloaded again; `--refresh` forces a new download.

The default endpoint is `http://127.0.0.1:8830/v1/systemone`. If nothing is listening there, the script
starts `target/release/bonjev serve --model ternary-bonsai-2-27b --port 8830 --ctx 16384` and stops that
process when the run ends. A server that is already up is left running; point `--url` at another server.
Per-task rows go to `benchmarks/cache/jevbench/exam.jsonl`.

Run it for one model + LoRA with `scripts/eval_exam.sh <model> <lora|-> <out.jsonl>`, which also writes a
probability dump (`--dump-probs`) and prints `answer_first` / `margin` / `m_wrong`. The selection metrics
(`rank3%`, `m_wrong`, `answer_first`) are described in `RECOMMENDATIONS.md`.

The published numbers use the release binary, **deterministic mode** (`BONJEV_NO_REUSE=1`), 16384
context, one HTTP request at a time on port 8830, no client SDK.

Grading: a **choice** is correct when the highest-probability option is the author label; a **yes/no**
question when the probability of `true` is on the same side of 0.5 as the author label; a **score** when
the highest-probability level equals the author level. The probability-weighted average is recorded but
is not the grade.

Serving styles: the engine's default routing is `native_user` for choice and `answer` for yes/no and
score (chosen by the style sweep); `BONJEV_STYLE=top3` averages three styles and is used for 1.7B.

## Decision Index (DI)

The arbiter for LoRA selection is the external **Decision Index** kit
([github.com/apolinario/decision-index](https://github.com/apolinario/decision-index), edition 0.2.1).
All DI runs here are on **`ternary-bonsai-2-27b`** (Bonsai 2, Qwen3.8); the v1 `ternary-bonsai-27b` was
not run on DI.
A copy is kept under `archived/benchmarks/decision-index/`; install the kit from upstream to reproduce:

```bash
pip install "decision-index"            # or: pip install -e <path-to-the-kit>
```

**Suite rows.** The kit reads the frozen suite. Fetch/build it with the kit:

```bash
python -m decision_index suite download --dir ~/.cache/huggingface/decision-index
# or rebuild from pinned sources: python -m decision_index suite rebuild --work work
```

Rows land in `~/.cache/huggingface/decision-index/artifacts/benchmark-suite/release-v2-rebuilt/`
(`selected-rows.jsonl.gz` + `added-rows.jsonl.gz`). Build samples with the kit's stratified sampler:

```bash
python -m decision_index suite sample --edition 0.2.1 --n 300 --out sample-300.jsonl.gz
```

Samples used here: `sample-300`; `sample-500-hybrid` (the kit's `--n 500` minus rows over 16k
`proxy_tokens` = 491 rows); `sample-1500`. The kit applies the scoring subsets at read time (ACOS is
scored on 400 of its 1399 reviews; ToolRet/BRIGHT only answerable queries; Home-appliance deduplicated).

**Run.** `scripts/di_run.sh` starts our server, runs the kit against it, and scores:

```bash
scripts/di_run.sh <name> <model> <lora|-> <rows.jsonl.gz> [limit]
# e.g.
DI_CTX=16384 scripts/di_run.sh d1500_4_b10 ternary-bonsai-4b work/best/4b/B10.gguf sample-1500.jsonl.gz
```

It calls `python -m decision_index run --engine http --option base_url=http://127.0.0.1:8830
--option model=<name> --rows <rows> --out benchmarks/cache/di/<name>` and then
`python -m decision_index score --results ... --suite-dir <suite>`. Env: `DI_PYTHON` (the kit's python),
`DI_CTX` (context, default 32768; we use 16384 for the hybrid/1500 samples), `PRISM_LLAMA_DIR`. Runs are
serialized through a lock (one GPU).

**Scoring.** The kit's `scores.json` holds each track's **native** primary metric (accuracy / macro-F1 /
nDCG@10 / per-review F1 / Brier). We compare configurations by **`A_native`** = the equal-weight mean of
the per-track native scores (ACOS reported separately); the official index is coverage-adjusted and ≈0 on
a sample. Absolute DI differences are noisy (±5–8 pp), so use paired tests over the scored questions.
Per-track results are in `benchmarks/cache/di/<name>/` (`results.jsonl`, `scores.json`,
`benchmark-summary.json`) and the consolidated matrix is `DI-TRACKS.md`.

## Documents

Benchmark docs (this directory):

| file | what it is |
|---|---|
| `RECOMMENDATIONS.md` | which LoRA to use per model, and the selection rule (DI arbiter, 231 guardrail) |
| `RESULTS.md` | consolidated 231 / DI / latency / Clef-comparison tables |
| `EXAM.md` | error taxonomy and style/ensemble analysis for the 231 exam |
| `FLIGHT.md` | flight-simulation control test (does the model drive the drone) |
| `DIRECT-INFER.md` | direct-generation control test (coherent text vs garbled) |
| `DI-TRACKS.md` | per-track DI matrix for the sample-1500 runs |

Methodology docs (`../docs/`):

| file | what it is |
|---|---|
| [`../docs/LORAS.md`](../docs/LORAS.md) | adapter inventory, where the mass is, merge theory, provenance |
| [`../docs/RECIPES.md`](../docs/RECIPES.md) | how each final LoRA was produced and how to reproduce it |
| [`../docs/BLOCK-SEARCH.md`](../docs/BLOCK-SEARCH.md) | the block/layer study (modules × layers) and glossary |
| [`../docs/TRAINING-FREE-HOOKS.md`](../docs/TRAINING-FREE-HOOKS.md) | what can be borrowed from the `strands-decider` model without training |
