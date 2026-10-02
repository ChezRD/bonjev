# Experimental patches: Clef joint schema head в llama.cpp

Базовый коммит: `PrismML-Eng/llama.cpp` @ `9a9394a89` — тот же, на котором стоит
`vendor/prism-llama.cpp`. Патчи сняты с локального рабочего дерева и проверены
`git apply --check` на чистом чек-ауте.

**Статус: experimental.** Каждый блок — кандидат на удаление или переработку.
Цель выноса — оценить, что из этого реально нужно, а что мусор.

## Патчи

| # | Файлы | Что делает |
| --- | --- | --- |
| 0001 | `src/llama-head.h`, `src/llama-head.cpp`, `src/CMakeLists.txt` | ggml-реализация головы Clef: структура `llama_head` (2× EvidenceRouting, 4× TransformerDecoderLayer, option_summary, residual scorer, prior/joint scales), загрузка весов из GGUF (`joint_head.gguf`), per-request входы (`a_q`, `a_o`, `oq`, `mask`, `lex`, `type`, `last`) в отдельном ggml-контексте и backend-буфере |
| 0002 | `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-graph.h`, `src/llama-model.cpp` | Графт головы в граф модели: `llama_head` как optional слой поверх финальных hidden states; сбор hidden states по чанкам (`head_acc`), запуск головы в графе последнего чанка; инвалидация `allow_reuse` при смене head; состояние в `llama_context` |
| 0003 | `include/llama.h` | Публичный C API: `llama_head_load`, `llama_head_set_record`, `llama_head_finalize`, `struct llama_head_record_c` |

## Применение

```bash
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
git checkout 9a9394a89
git am /home/chez/projects/jev/bonjev/patches/experimental/000*.patch
```

## Что оценить (кандидаты «оставить / выбросить»)

- **0001**: нужна ли голова именно в форке (против отдельного рантайма); нужен ли
  chunked-аккумулятор `head_acc` для промптов длиннее `n_batch`; хардкод геометрии
  из `joint_head_config.json`; минимальна ли правка `src/CMakeLists.txt`.
- **0002**: правильная ли точка графта (финальные hidden states последнего чанка);
  нужна ли переиспользуемость графа (`allow_reuse`) с головой вообще; не проще ли
  вынести head-стадию за пределы `llama_context`.
- **0003**: минимальность C API — `set_record` + `finalize` можно свести к одному
  вызову; `struct llama_head_record_c` дублирует внутренний `llama_head_record`.

Связанное: `joint_head.gguf` (512 МБ) лежит в кэше `Cloudflare/clef`;
конвертер `safetensors → GGUF` для головы в этот патч-сет не входит.
