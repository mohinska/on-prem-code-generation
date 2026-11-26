# sft-gemma4: те саме SFT на Gemma 4 31B (порівняння з Qwen3.8-27B)

## Підсумок

Ми дотренували `google/gemma-4-31B-it` (QLoRA), seed 42 і seed 43, на тих самих даних, з тими самими налаштуваннями і протоколом оцінки, що й [sft-v2](sft-v2.md) на Qwen3.8-27B. Мета — перевірити, чи виграш sft-v2 специфічний для Qwen. Критерії ті самі ([sft-v1-preregistration.md](sft-v1-preregistration.md)); база — Gemma без адаптера.

- **Вердикт: успіх.** Обидва seed виконують основний критерій: COBOLEval sampled pass@1 0.357 і 0.379, база 0.250. Held-out задачі теж вищі за базу.
- **Python:** HumanEval-Python sampled pass@1 падає лише на 0.010 (ліміт 0.02). У Qwen sft-v2 guard Python не пройшов (−0.041).
- **CobolCodeBench completion:** guard не пройшов (pass@1 0.087 → 0.009). Адаптер у 735 з 920 відповідей пише лише продовження програми від `WORKING-STORAGE SECTION.`, без заголовка, і така відповідь не компілюється окремо (розділ 3).
- **Gemma база значно сильніша за Qwen базу** (COBOLEval 0.250 проти 0.021), тому приріст від SFT менший (+0.107 проти +0.253), але після SFT Gemma вища за Qwen sft-v2.
- **Рішення:** `sft-gemma4-adapter` лишається `candidate`, бо не виконано guard CobolCodeBench.

| Benchmark (sampled pass@1) | Qwen base | Qwen sft-v2 seed 42 / 43 | Gemma base | **Gemma sft seed 42** | **Gemma sft seed 43** |
|---|---|---|---|---|---|
| COBOLEval | 0.021 | 0.274 / 0.282 | 0.250 | **0.357** | **0.379** |
| Held-out задачі CodeNet | 0.003 | 0.075 / 0.093 | 0.055 | **0.109** | **0.100** |
| CobolCodeBench completion | 0.009 | 0.026 / – | 0.087 | **0.009** | – |
| HumanEval-Python | 0.969 | 0.928 / – | 0.980 | **0.970** | – |

«–» = не вимірювали: seed 43 за преєстрацією оцінюється лише на COBOLEval sampled і held-out задачах.

## 1. Що змінилось порівняно з sft-v2

Лише модель. Дані `sft-dataset:decontaminated-v2` (v6), 565 рядків для навчання / 76 для eval, QLoRA 4-bit, r 16, alpha 32, lr 2e-4, 3 епохи (108 кроків), декодування й held-out задачі ті самі ([project-log](project-log.md#gemma-4-31b-comparison-setup)).

| | Qwen3.8-27B | Gemma 4 31B |
|---|---|---|
| Модель | dense 27B, hybrid Gated DeltaNet + attention | dense 31B (30.7B текстова частина), 50 sliding-window + 10 full-attention шарів |
| LoRA | attention + DeltaNet + MLP, 116,727,808 параметрів | `q/k/v/o_proj` + MLP, 122,429,440 параметрів (у full-attention шарах немає `v_proj`) |
| Eval loss, крок 108 (seed 42 / 43) | 0.1614 / 0.1605 | 0.1857 / 0.1874 (інший токенізатор, напряму не порівнюється) |
| Baseline | baseline v2 + заморожений baseline | `gemma4-baseline`: усі групи, лише non-thinking |

Відповіді в даних згенерував Qwen3.8 (rejection sampling), тож для Gemma вони off-policy. Запуски, генерації й час: [project-log](project-log.md#sft-gemma4-sft-on-gemma-4-31b-and-its-generations).

## 2. Результати

Δ = Gemma sft − Gemma base, парний bootstrap, 95% CI по задачах.

| Benchmark | Metric | Gemma base | sft seed 42 | sft seed 43 | Δ seed 42 [95% CI] | Δ seed 43 [95% CI] |
|---|---|---|---|---|---|---|
| COBOLEval greedy | pass@1 | 0.249 | **0.379** | – | +0.130 [+0.053, +0.208] | – |
| | compile rate | 0.409 | 0.776 | – | | |
| COBOLEval sampled | pass@1 | 0.250 | **0.357** | **0.379** | +0.107 [+0.048, +0.163] | +0.129 [+0.070, +0.186] |
| | pass@10 | 0.478 | **0.638** | **0.644** | +0.161 [+0.090, +0.227] | +0.167 [+0.100, +0.230] |
| | compile rate | 0.388 | 0.751 | 0.757 | | |
| | truncated | 0.001 | 0.002 | 0.002 | | |
| CobolCodeBench completion greedy | pass@1 | 0.087 | 0.007 | – | −0.080 [−0.159, −0.022] | – |
| | compile rate | 0.486 | 0.138 | – | | |
| CobolCodeBench completion sampled | pass@1 | 0.087 | 0.009 | – | −0.078 [−0.157, −0.021] | – |
| | pass@10 | 0.098 | 0.022 | – | −0.076 [−0.152, −0.011] | – |
| | compile rate | 0.497 | 0.108 | – | | |
| | truncated | 0.000 | 0.012 | – | | |
| CobolCodeBench instruct greedy | pass@1 | 0.000 | 0.000 | – | +0.000 [+0.000, +0.000] | – |
| | compile rate | 0.377 | 0.543 | – | | |
| CobolCodeBench instruct sampled | pass@1 | 0.017 | 0.008 | – | −0.010 [−0.032, +0.003] | – |
| | pass@10 | 0.080 | 0.066 | – | −0.014 [−0.057, +0.027] | – |
| HumanEval-Python greedy | pass@1 | 0.978 | 0.974 | – | −0.004 [−0.018, +0.006] | – |
| HumanEval-Python sampled | pass@1 | 0.980 | 0.970 | – | −0.010 [−0.025, +0.001] | – |
| | pass@10 | 0.991 | 0.991 | – | +0.000 [−0.003, +0.004] | – |

**Held-out задачі CodeNet.** Ті самі 421 задача, greedy 1 і sampled 5 відповідей на задачу.

| Model | greedy pass@1 | sampled pass@1 | sampled pass@5 | Δ sampled pass@1 [95% CI] | Δ sampled pass@5 [95% CI] | compile rate (greedy / sampled) |
|---|---|---|---|---|---|---|
| Gemma base | 0.048 | 0.055 | 0.095 | | | 0.202 / 0.210 |
| Gemma sft seed 42 | 0.116 | 0.109 | 0.216 | +0.054 [+0.032, +0.075] | +0.121 [+0.086, +0.159] | 0.404 / 0.379 |
| Gemma sft seed 43 | 0.107 | 0.100 | 0.195 | +0.045 [+0.026, +0.065] | +0.100 [+0.062, +0.138] | 0.375 / 0.386 |

**Висновки.**

- **COBOLEval:** обидва seed вищі за базу з CI вище 0, різниця між seed 0.022. Compile rate майже подвоївся (0.388 → 0.75).
- **Що ще не компілюється** (COBOLEval sampled, seed 42, 721 відповідь): помилки використання COBOL, ті самі, що в Qwen sft-v2: арифметика в `PERFORM VARYING … FROM I + 1` (110) і `… TO` (54), індекси (108), невідомі функції (46). У бази переважали помилки формату (`unexpected *` 444).
- **Held-out задачі:** приріст +0.054 / +0.045, менший, ніж у Qwen (+0.072 / +0.089), але від вищої бази; абсолютно Gemma sft вища.
- **HumanEval-Python:** майже без втрат (−0.010, CI містить 0). Чому Gemma втрачає менше за Qwen, не перевіряли.
- **CobolCodeBench instruct:** compile rate виріс з 0.377 до 0.543, pass@1 майже 0 і в бази, і в адаптера. Цей бенчмарк не входить у критерії.

## 3. CobolCodeBench completion: відповіді без заголовка програми

727 з 810 помилок компіляції (seed 42, sampled) — `PROGRAM-ID header missing`.

| CobolCodeBench completion sampled (920 відповідей) | Уся програма | Лише від `WORKING-STORAGE` | Без програми |
|---|---|---|---|
| Qwen base | 459 | 361 | 100 |
| Qwen sft-v2 | 753 | 1 | 166 |
| Gemma base | 911 | 9 | 0 |
| Gemma sft seed 42 | 174 | 735 | 11 |

- SFT на Qwen зсунуло відповіді до повної програми, а на Gemma — у протилежний бік. 33 з 46 задач Gemma sft у жодній з 20 спроб не пише повну програму. COBOLEval це не зачіпає: 2913 з 2920 відповідей — повні програми.
- У промпті CobolCodeBench дві протилежні вказівки: коментар бенчмарку «Complete only the WORKING-STORAGE SECTION and PROCEDURE DIVISION» і наше речення «Write a complete GnuCOBOL program».
- У навчальних даних такого випадку немає. Усі 641 відповіді — повні програми; усі 327 completion-промптів закінчуються фразою «answer with the whole program»; початок програми в промпті — 3 рядки (медіана), до `DATA DIVISION`; жодного `FILE SECTION`, `WORKING-STORAGE` чи «Complete only» у промптах; file I/O лише в 24 відповідях.
- Ймовірне пояснення (не перевірене): Gemma пов'язала «повну програму» з фразою «answer with the whole program», а без неї виконує коментар бенчмарку. Qwen узагальнив ширше.
- Навіть повні програми Gemma sft тут рідко правильні: зі 174 компілюються 99, проходять тести 8. Причина та сама, що в Qwen: майже немає file I/O у даних.

**Діагностика: що, якщо додати заголовок.** Це не результат моделі, бо відповіді відредаговано за правилом. Тим відповідям, де немає `PROGRAM-ID`, підставили початок програми з промпту (`repair()` з `eval/repair_format.py`, лише це правило) і виконали їх знову ([project-log](project-log.md#sft-gemma4-cobolcodebench-header-repair-diagnostic-answers-edited)).

| CobolCodeBench completion sampled | Gemma base | Gemma sft | Gemma sft + заголовок |
|---|---|---|---|
| pass@1 | 0.087 | 0.009 | 0.047 |
| pass@10 | 0.098 | 0.022 | 0.087 |
| compile rate | 0.497 | 0.108 | 0.455 |

Відсутній заголовок пояснює більшу частину падіння. З ним адаптер уже не значуще гірший за базу (Δ pass@1 −0.040 [−0.102, +0.005]), але й не кращий. Решта — робота з файлами.

## 4. Критерії й вердикт

| Критерій | Qwen sft-v2 | **Gemma sft** |
|---|---|---|
| Основний: COBOLEval sampled pass@1 вищий за базу (нижня межа CI > 0), compile rate вищий | так, обидва seed | **так**, обидва seed |
| Додатковий: COBOLEval pass@10 вищий (нижня межа CI > 0), greedy pass@1 не нижчий | так | **так** |
| Guard Python: HumanEval-Python sampled pass@1 падає не більше ніж на 0.02 | ні (−0.041) | **так** (−0.010) |
| Guard CobolCodeBench: completion pass@1 не гірший (верхня межа CI ≥ 0), truncated +10 пунктів максимум | так | ні (−0.078 [−0.157, −0.021]) |
| Held-out задачі: pass@1 вищий за базу (нижня межа CI > 0) | так, обидва seed | **так**, обидва seed |
| **Вердикт** | успіх без guard Python | **успіх** (guard CobolCodeBench не виконано) |

За преєстрацією «успіх» = основний критерій + guard Python + held-out задачі; guard CobolCodeBench у правило вердикту не входить.

## 5. Наступні кроки

1. **Дані completion.** Додати completion-рядки з початком програми до `WORKING-STORAGE`, з file I/O, і частину без фрази «answer with the whole program». Перевірити на обох моделях.
2. **Python на Qwen.** Gemma майже не втрачає Python на тих самих даних; варто порівняти з Qwen після replay Python або sweep lr × LoRA rank.
3. **DPO** на парах «пройшла / не пройшла» для помилок використання COBOL (розділ 2), для моделі, яку оберемо далі.

## 6. Обмеження

- Лише два seed. Guards і CobolCodeBench виміряні тільки для seed 42.
- Лише non-thinking; thinking-режим Gemma не оцінювали.
- Навчальні відповіді згенерував Qwen, а не Gemma (off-policy для Gemma).
- Held-out задачі перевіряються кількома прикладами тестів з умови CodeNet, лише формат stdin, 5 відповідей на задачу.
- Окремо не перевіряли, що vLLM застосовує LoRA на `k_proj` у full-attention шарах Gemma так само, як у навчанні; `serve.check` лише підтверджує, що адаптери змінюють відповіді.
