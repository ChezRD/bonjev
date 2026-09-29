# Benchmark notes

These notes are for someone checking a number in the [performance table](../README.md). Each file says how that run was scored, which questions were left out, and which misses remain after that cut.

## Public exam

```bash
python3 benchmarks/jevbench_exam.py
```

The script downloads the pinned JevBench public splits (`easy`, `original`, `hard`, 231 tasks, commit `3749b4fc`) into `benchmarks/cache/jevbench/` and scores them with `POST /v1/systemone`. A file whose sha256 already matches is not downloaded again. `--refresh` forces a new download.

The default endpoint is `http://127.0.0.1:8830/v1/systemone`. If nothing is listening there, the script starts `target/release/bonjev serve --model bonsai2 --port 8830` and stops that process when the run ends. A server that is already up is left running. Point `--url` at another server. Per-task rows go to `benchmarks/cache/jevbench/exam.jsonl`.

The published `bonsai2` numbers use the release binary on an NVIDIA RTX 4060 Ti 16 GB, context 32,768. `q2` is the old name of that same file. Later suites are one HTTP request at a time to `POST /v1/systemone` on port 8830. No client SDK.

A **choice** is correct when the option with the highest probability is the author label. A **yes/no** question is correct when the probability of `true` is on the same side of 0.5 as the author label. A **score** is correct when the level with the highest probability equals the author level. The probability-weighted average is recorded and is not the grade.

[sys1bench](https://github.com/rssr25/system-one-bench) is the base template only (framing `f0`). The paraphrase sweep was not run. Each generator was called with `n=500` and `seed=42`. The [canary](https://github.com/rssr25/system-one-bench/blob/main/src/sys1bench/data/canary/canary.jsonl) is the frozen set in that repository: 100 tickets and 100 phishing emails, seed 9001.

Choice criteria that arrived as a list of `{key, description}` were sent as a map, because BonJev ignores a choice list. Score criteria that arrived as `{level, description}` were sent as a list of the description strings, so the keys are `"0"`, `"1"`, and so on. An empty passage description was replaced with the passage id.

## How 2,531 / 3,093 is built

Seven score questions, 500 items each, before any cut: **2,556 / 3,500**.

| Question | Raw | Left out | Hits inside the cut | After the cut |
| --- | ---: | ---: | ---: | ---: |
| English ticket priority | 246 / 500 | 75 | 0 | 246 / 425 |
| Multilingual ticket priority | 280 / 500 | 24 | 0 | 280 / 476 |
| Phishing urgency | 308 / 500 | 98 | 0 | 308 / 402 |
| Passage relevance | 362 / 500 | 140 | 14 | 348 / 360 |
| Guardrail risk | 441 / 500 | 70 | 11 | 430 / 430 |
| Policy severity | 419 / 500 | 0 | 0 | 419 / 500 |
| Log severity | 500 / 500 | 0 | 0 | 500 / 500 |
| **Sum** | **2,556 / 3,500** | **407** | **25** | **2,531 / 3,093 (81.8%)** |

A left-out question is removed from both the numerator and the denominator. When the model matched a broken author label, that hit is removed too. That is why passage relevance loses 14 hits and guardrail risk loses 11.

The same two ticket and phishing rules on the canary remove 41 score questions, all of them misses: **107 / 200** becomes **107 / 159**. Choice and yes/no on the canary stay **888 / 900**.

## Questions left out because the item contradicts itself

**489** questions are out of the published scores.

| Suite | Left out | What contradicts what |
| --- | ---: | --- |
| sys1bench main score | 407 | See the table above. Detail is in the suite notes. |
| sys1bench canary score | 41 | 13 ticket priorities with author level 0 on a real request, and 28 phishing urgencies (Thursday lunch, and "open to review"). |
| typed-decisions | 36 | The author caption says the invoice matches. The same record shows another quantity, a repeated invoice id, or a purchase order with no id. 31 of these the model matched. 5 it did not. |
| AY prompt injection | 5 | Ordinary requests labeled as injections: a film plot (`inj:315`), a book question (`inj:543`), `generate c++` (`inj:171`), and two SQL requests (`inj:584`, `inj:654`). All five were misses, so **349 / 400** becomes **349 / 395**. |

## Suite notes

- [Support tickets](sys1bench-tickets.md). 375 priority misses remain (179 English, 196 multilingual). The author level matches the printed formula. The model follows the level caption and lands one step off.
- [Phishing and logs](sys1bench-phishing-and-logs.md). 175 real misses: 43 timesheets, 51 invoices, 81 paging decisions.
- [Passages, guardrails, and policy](sys1bench-passages-policy.md). 12 passage misses and 130 policy misses remain. Guardrail risk has no remaining misses. The 49 intent misses are the same jailbreak-plus-harm sentences already cut from the risk score.
- [SST-2, AY injection, typed-decisions](sst2-ay-typed-decisions.md).

## Runs with no questions removed

These rows in the performance table are the full set. Nothing in them was dropped for a self-contradiction.

| Run | How it was scored | Result on `bonsai2` |
| --- | --- | --- |
| [JevBench](https://github.com/fstandhartinger/jevbench) Public, `easy` + `original` + `hard` | Correct task ids. The [live board](https://benchmarkheaven.com/jev-models) composite is a different number. `bonsai` scored 163/231: it lost 41 tasks against `bonsai2` and gained `hard-opus-b-tradeoff-01` and `hard-opus-c-temporal_numeric-02`. Prompts up to 4k tokens took up to 4.8 s on `bonsai2` and 4.4 s on `bonsai`. | 202 / 231 |
| [Persian](https://github.com/ArmanJR/Jev-Persian-Benchmark) v1.0.0, commit [`ac218d9`](https://github.com/ArmanJR/Jev-Persian-Benchmark/commit/ac218d96630da9d9cc08fd897868c4d3c7048b0d) | Direct POST. Choice and yes/no as above. Score is within ±0.5 of the expected value. Six questions share one prompt prefix. | Choice 237/240, yes/no 154/160, score 68/80 |
| [BTZSC pilot](https://github.com/AbdelStark/jev-benchmarks) on [btzsc/btzsc](https://huggingface.co/datasets/btzsc/btzsc) revision `fef2a2ac`, seed `20260917`, 100 examples each | Direct POST. Question: which single label best describes the input text. | AG News 90/100, emotion 45/100, Banking77 76/100 (72 labels in that sample) |

[Decision Index](https://github.com/apolinario/decision-index) 0.2.1 and the [tasksource](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) training recast were not run.
