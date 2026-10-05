# Flight check: models × LoRA on the Jev Flight Lab

The Jev Flight Lab demo (a Three.js drone, 4 checkpoints) was run against our `bonjev` server on our
models and the `work/best/` LoRAs. It is a direct test of whether a model can actually drive, not just
pick labels.

**Contents**

- [Setup](#setup)
- [Results](#results)
- [Conclusions](#conclusions)

## Setup

- Server: `bonjev serve` on port 8830 with the LoRA under test.
- Deterministic mode (no prefix cache, no checkpoint path):

  ```bash
  BONJEV_NO_REUSE=1 BONJEV_CKPT=0 \
    ./target/release/bonjev serve --model <id> --lora <file> --port 8830
  ```

- Demo backend on port 8010; the probe is `scripts/flight_probe.py`.
- The flight is **not reproducible** (near-tie decisions cascade and change the trajectory), so each
  combo is run **3 times**.
- Metrics: success (reached the end), p50/p95 decision latency, elapsed, distance, **stuck** (holds),
  **checkpoint** (progress out of 4), efficiency (distance / elapsed).

## Results

13 combos × 3 runs. All LoRAs are `work/best/` recipes.

| model | LoRA | success | p50 ms | p95 ms | elapsed s | distance m | stuck | checkpoint | eff m/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | base | **1/3** | 328 | 334 | 46.9 | 83.1 | 3 | 4 | 1.77 |
| 1.7B | `m17_tiny25_single` | **0/3** | — | — | — | — | — | — | — |
| 4B | base | 2/3 | 765 | 777 | 47.0 | 62.0 | 3 | 4 | 1.37 |
| 4B | `B10` | **3/3** | 1184 | 1201 | 50.0 | 61.5 | 0 | 4 | 1.23 |
| 4B | `all3_align` | 3/3 | 1178 | 1195 | 49.5 | 58.5 | 0 | 4 | 1.18 |
| 8B | base | 3/3 | 1211 | 1226 | 52.1 | 58.5 | 1 | 4 | 1.12 |
| 8B | `m8_inv_b04` | 3/3 | 1767 | 1787 | 62.4 | 60.0 | 0 | 4 | 0.96 |
| 8B | `R3` | 3/3 | 1648 | 1784 | 61.9 | 58.5 | 0 | 4 | 0.95 |
| 27B | base | 3/3 | 4387 | 4411 | 114.6 | 57.7 | 0 | 4 | 0.50 |
| 27B | `vega_clef_plumb` | 3/3 | 5976 | 6009 | 155.8 | 58.5 | 1 | 4 | 0.38 |
| 27B | `m27_clef_plumb_at_half` | 3/3 | 5970 | 5996 | 161.8 | 58.5 | 2 | 4 | 0.36 |
| 27B | `vega_clef_plumb_at_half` | 3/3 | 5993 | 6672 | 167.2 | 58.5 | 2 | 4 | 0.35 |
| 27B | `clef_plumb_vegaffn` | 3/3 | 6685 | 6847 | 172.9 | 57.7 | 1 | 4 | 0.34 |

Medians are over successful runs (failure = "Six consecutive holds").

## Conclusions

- **1.7B cannot fly:** `tiny25` 0/3, base 1/3 (it gets stuck on holds).
- **4B / 8B / 27B are stable at 3/3**, all reaching 4/4 checkpoints.
- **Latency grows with size:** 4B ~1.2 s, 8B ~1.7 s, 27B ~6 s (p50); p95 is close to p50.
- **Efficiency (m/s) drops with size:** 4B ~1.2, 8B ~0.95, 27B ~0.35–0.5 — the flight time is set by
  decision latency, not quality.
- The LoRAs **do not break** control (4B/8B/27B succeed 3/3, stuck 0–2).

Raw logs are kept outside the repository.
