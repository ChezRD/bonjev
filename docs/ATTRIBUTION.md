# Attribution and licenses

Every third-party model used to build the BonJev LoRA adapters, with its license. BonJev's own code
is **MIT** (see [../LICENSE](../LICENSE) and [../NOTICE](../NOTICE)). The **adapters** are training-free
recombinations of the deltas below, which are all **Apache-2.0**, so the adapters are distributed under
**Apache-2.0** with the attribution below.

Licenses were read from the Hugging Face model cards on 2026-10-05.

**Contents**

- [Target models](#target-models)
- [Sources used in the published adapters](#sources-used-in-the-published-adapters)
- [Sources used in experiments](#sources-used-in-experiments)
- [Considered but not used](#considered-but-not-used)
- [Base models](#base-models)
- [Notices](#notices)

## Target models

The adapters are built for the ternary Bonsai packs (applied at runtime with `BONJEV_LORA`).

| repo | what | license |
|---|---|---|
| `prism-ml/Ternary-Bonsai-2-27B-gguf` | Bonsai 2 27B (PQ2_0 / PTQ1_0) | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-27B-gguf` | Ternary-Bonsai-27B | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-8B-gguf` | Ternary-Bonsai-8B | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-4B-gguf` | Ternary-Bonsai-4B | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-1.7B-gguf` | Ternary-Bonsai-1.7B | Apache-2.0 |

## Sources used in the published adapters

| adapter | source repo | base model | license |
|---|---|---|---|
| 1.7B `m17_tiny25_single` | `lostargon/Tiny-Jev` (`Tiny-Jev-1.7B`) | Qwen3-1.7B | Apache-2.0 |
| 4B `B10` | `jaredpalmer/kev-4b` (branch `qwen3`) | Qwen3-4B-Base | Apache-2.0 |
| 4B `B10` | `CullenYap/CandiGate-Qwen3-4B` | Qwen3-4B | Apache-2.0 |
| 4B `B10` | `sennaLLMLearner/qwen3-4b-system-one-lora` | Qwen3-4B-Instruct-2507 | Apache-2.0 |
| 8B `m8_inv_b04` | `jaredpalmer/kev-8b` | Qwen3-8B-Base | Apache-2.0 |
| 8B `m8_inv_b04` | `CaoHaoWei/Jev-LCT-Qwen3-8B` | Qwen3-8B | Apache-2.0 |
| 27B `vega_clef_plumb` | `Cloudflare/clef` | Qwen3.8-27B | Apache-2.0 |
| 27B `vega_clef_plumb` | `totum-labs/Qwen3.5-27B-plumb` (plumb/2) | Qwen3.5-27B | Apache-2.0 |
| 27B `vega_clef_plumb` | `vllm-sr/Decision-2.0-Vega-27B` | Qwen3.8-27B | Apache-2.0 |

## Sources used in experiments

Used while developing the recipes (the block/layer and merge study), not in the published adapters.

| source repo | base model | license |
|---|---|---|
| `Cloudflare/clef-flash` | Qwen3.5-9B | Apache-2.0 |
| `ZefanCai/Open-Jev-9B` | Qwen3.5-9B | Apache-2.0 |
| `ZefanCai/Open-Jev-27B-v1.1` | Qwen3.8-27B | Apache-2.0 |
| `autotrust/JEV-27B` | Qwen3.8-27B | Apache-2.0 |
| `autotrust/JEV-9B` | Qwen3.5-9B | Apache-2.0 |
| `Okura66/Kahn1-Qwen3.5-4B-LoRA` | Qwen3.5-4B | Apache-2.0 |
| `xuhaodev/Qwen3-1.7B-Jev` | Qwen3-1.7B | Apache-2.0 |
| `SimpleJev/JevAny-Qwen3.8-27B-LoRA` | Qwen3.8-27B | Apache-2.0 |
| `SargeDev/Jev_Qwen3.8-27B-r2-LoRA` | Huihui-Qwen3.8-27B-abliterated | Apache-2.0 |
| `alibiserikbay/JevK5` | Qwen3.5-4B | Apache-2.0 |
| `alibiserikbay/JevK5-9B` | Qwen3.5-9B | Apache-2.0 |
| `denis-pplx/autojev-27b` | Qwen3.8-27B | Apache-2.0 |
| `aikexue170/jev-novel-2-27b-bf16` | Qwen3.8-27B | Apache-2.0 |
| `Camellia86/Canopy-Jev-27B` | Qwen3.8-27B | Apache-2.0 |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | Qwen3.8-27B | Apache-2.0 |

## Considered but not used

Listed for completeness; none of these are in a published adapter.

| source repo | base model | license | notice |
|---|---|---|---|
| `interfaze-ai/lev` | Qwen3.5-4B/9B | Apache-2.0 | — |
| `jsaurabh/qwen3.5-9b-jev-data-mix-v2` | Qwen3.5-9B | Apache-2.0 | — |
| `HopitAI/hopper` | Qwen3.5-4B | **other: research-and-demo** | research and demo use only; see the model card |
| `dhtocks/malkuth-4b` | Qwen3.5-4B-Base | **CC-BY-NC-4.0** | non-commercial; not used in any adapter |

## Base models

The source adapters were trained against the Qwen models below, all **Apache-2.0**:

`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-4B-Instruct-2507`,
`Qwen/Qwen3-8B`, `Qwen/Qwen3-8B-Base`, `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`,
`Qwen/Qwen3.8-27B`, `huihui-ai/Huihui-Qwen3.8-27B-abliterated`.

The Bonsai packs are derived from `Qwen/Qwen3.8-27B` (Apache-2.0).

## Notices

- **HopitAI/hopper** is released for **research and demo use only** (license `other`); it is not part
  of any published adapter.
- **dhtocks/malkuth-4b** is **CC-BY-NC-4.0** (non-commercial); it is not part of any published adapter.
- **SimpleJev/JevAny** is Apache-2.0 and its own NOTICE credits **Kev by Jared Palmer**
  (`jaredpalmer/kev`, Apache-2.0); we recombine its delta with attribution.
- **plumb** and **canopy**: their orgs were not pinned in our notes; the public repos named above
  (`totum-labs/Qwen3.5-27B-plumb`, `Camellia86/Canopy-Jev-27B`) are Apache-2.0.
- **Cloudflare/clef** and **Cloudflare/clef-flash** are Apache-2.0.
- All other sources and all base models are Apache-2.0.

## See also

- [LORAS.md](LORAS.md) — adapter inventory and provenance.
- [RECIPES.md](RECIPES.md) — how each final adapter is built.
- [../NOTICE](../NOTICE) — the repository notice.
