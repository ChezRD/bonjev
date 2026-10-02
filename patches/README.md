# Patches

Две коллекции, разложенные по роли назначения:

- **ada-surgery** — 33 патча из [professorpalmer/bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery/tree/main/patches) (MIT).
  База: `PrismML-Eng/llama.cpp` @ `adfffbe` (он же пин `vendor/prism-llama.cpp` и релиз `prism-b10743-adfffbe`).
- **experimental** — наши локальные изменения (Clef joint schema head). База при снятии: `9a9394a89`;
  на `adfffbe` требуют повторной проверки `git apply --check`.

Порядок накатки: `base → prefill → decode → kv → batch-invariant → mtp → server → sampling → diagnostics → experimental`.
Внутри каталога номера сохраняют исходный порядок серии.

**Важно:** серия линейная — накатывать надо целиком и через `git am -3` (проверено: все 33
применяются на `adfffbe`; `git apply` без 3-way даёт 19/33, а выборочные наборы падают на
зависимостях). `scripts/build-prism-llama.sh` по умолчанию применяет все 33, экспериментальные —
только по явному запросу.

## Сводка по ролям

| Каталог | Патчей | Суть | Для нас |
| --- | ---: | --- | --- |
| `base/` | 3 | Фундамент PTQ1_0-пути (planar layout), PDL-корректность, тесты | нужен, если используем PTQ1_0 |
| `prefill/` | 4 | Ускорение prompt processing: GDN-gather, PTQ1_0 MMQ-диспатч, FWHT→q8 фьюжн, пул | **высокий приоритет** |
| `decode/` | 9 | PTQ1_0 mat-vec/multi-column, bf16 small-row, smem, review-фиксы | средний (растёт при PTQ1_0) |
| `kv/` | 3 | In-place q4_0/q8_0 K/V в FA, tiered KV (`--kv-vram-cells`), GQA-decode на MMA | средний |
| `batch-invariant/` | 3 | `GGML_CUDA_BATCH_INVARIANT`: воспроизводимость near-tie решений | средний |
| `mtp/` | 8 | MTP/speculative (draft, catch-up, окна) | **не нужен** (single-pass) |
| `server/` | 1 | `--reasoning-effort-*`, max-tokens floor | n/a (нет генерации) |
| `sampling/` | 1 | Отбрасывание out-of-vocab из backend sampling | n/a |
| `diagnostics/` | 1 | `GGML_CUDA_OP_TIMING=1` — время по нодам графа | полезно для профилирования |
| `experimental/` | 3 | Clef joint schema head (llama-head + графт + C API) | отдельная фича, см. README внутри |

## Каталог

| # | Патч | Роль | Для нас | Заметка |
| --- | --- | --- | --- | --- |
| 0001 | planar-transposed activation layout для PTQ1_0 | base | high* | фундамент PTQ1_0-пути; *если PTQ1_0 |
| 0002 | dedicated PTQ1_0 mat-vec kernel (full-lane) | decode | med | зависит от 0001 |
| 0003 | тесты: Bonsai 2 projection shapes для mul_mat | base | — | тесты |
| 0004 | `GGML_CUDA_BATCH_INVARIANT` (small matmuls) | batch-invariant | med | в prebuilt env-переменной нет |
| 0005 | bf16 mat-vec для матриц <64 колонок | decode | med | малые батчи |
| 0006 | fold recurrent-state gather в GATED_DELTA | prefill | **high** | работает на всех упаковках |
| 0007 | PTQ1_0 decode на Ampere/Ada: SoA q8 | decode | med | |
| 0008 | PTQ1_0 hybrid dispatch: y_soa + MMQ | prefill | **high** | именно он даёт ~2× prefill у PTQ1_0 |
| 0009 | FA MMA читает q4_0/q8_0 K/V in-place | kv | med | убирает F16 scratch-копию, важен на глубине |
| 0010 | PTQ1_0 multi-column mat-vec (raw digits) | decode | med | verify/малые батчи |
| 0011 | qwen35: MTP-граф публикует только output rows | mtp | low | MTP |
| 0012 | draft-mtp: catch-up ряды одним декодом | mtp | low | |
| 0013 | Hadamard сам квантует свой выход (q8_1) для PTQ1_0 mat-vec | prefill | med | экономит отдельный quantize-launch |
| 0014 | sampling: отброс out-of-vocab токенов | sampling | n/a | нет генерации |
| 0015 | draft-mtp: сброс stale deferred рядов | mtp | low | |
| 0016 | FWHT-q8 held pool: LIFO-освобождение | prefill | low | память/пулы |
| 0017 | server: `--spec-draft-depth-max` | mtp | low | |
| 0018 | off-PTQ1_0 параметры ядра; Ampere использует … | decode | med | тюнинг |
| 0019 | review-фиксы native q4/q8 FA + MTP | mtp | low | |
| 0020 | shared smem budget для PTQ1_0 mat-vec | decode | med | occupancy |
| 0021 | flush MTP catch-up при переполнении | mtp | low | |
| 0022 | вернуть scalar PTQ1_0 vec-dot на MUSA | decode | n/a | MUSA (не NVIDIA) |
| 0023 | MTP catch-up переживает failed draft decode | mtp | low | |
| 0024 | PTQ1_0 layout helper после `ggml_cuda_…` | decode | med | build-фикс |
| 0025 | остаток review #221: pass q8 layout | decode | med | CUDA/HIP/MUSA |
| 0026 | PDL-wait в PTQ1_0 planar | base | high* | корректность на Hopper+; на Ada no-op |
| 0027 | вернуть warp-reduce epilogue под BATCH_INVARIANT | batch-invariant | med | |
| 0028 | tiered KV cache (`--kv-vram-cells`) | kv | med | длинный контекст; VMM-рантайм |
| 0029 | quantized-KV GQA decode на in-place MMA FA | kv | med | decode на глубине |
| 0030 | BATCH_INVARIANT: occupancy-independent FA KV split | batch-invariant | med | |
| 0031 | `--spec-draft-window`, `--spec-draft-n-max-tail` | mtp | low | |
| 0032 | server: `--reasoning-effort-allow/-fallback`, floor | server | n/a | нет генерации |
| 0033 | `GGML_CUDA_OP_TIMING=1` | diagnostics | полезно | профилирование |

## Что из этого делается через shim bonjev (без сборки форка)

Все 33 патча живут **ниже уровня `llama.h`** (CUDA-ядра, граф, VMM, сервер) — через C-шим их
реализовать нельзя. На шим-уровне доступно другое:

1. **q8_0 K/V** — не патч, а `llama_context_params.type_k/type_v`. По `QUALITY.md`: q4_0 KV
   переворачивает top-1 токен в 1 из 48 позиций на глубине 8–16k, q8_0 — в 1 из 160 (эталон f16).
   Добавлено `BONJEV_KV_TYPE=q8_0` в `c/src/shim.c`.
2. **`GGML_CUDA_BATCH_INVARIANT`** (0004/0027/0030) — env-переменная ggml; в prebuilt
   `prism-b10743-adfffbe` её нет (проверено `strings`) — только сборка.
3. **Tiered KV, MTP, reasoning-флаги** — в prebuilt отсутствуют; путь один: собрать
   `vendor/prism-llama.cpp` с нужным набором патчей (см. `scripts/build-prism-llama.sh`).
