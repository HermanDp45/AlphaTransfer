# V3: отдельные H3/H5 модели и точное воспроизведение

**Все шесть отдельных моделей обучены заново:** H3/H5 × cutoffs 1 января 2024, 2025 и 2026. Это ежегодное скользящее окно, не ежемесячное переобучение. H5 полностью воспроизводит историческую `basis_train_120m`; H3 тоже совпадает с её историческими отдельно обученными H3-прогнозами. Максимальная погрешность raw/calibrated probability и utility: **2.27e-13**, ни одного отличия candidate/portfolio flags. Все исходные V3/V4 файлы сохранены побитово.

## Точный рецепт

Пять валют обучаются вместе. 120 месяцев train заканчиваются до 12 месяцев calibration. Признаки: исходные 14 return/rank/volatility плюс MOEX CNY close-minus-fixing basis, без расширения историей CNY. Median imputation с сохранением пустых колонок, StandardScaler и one-hot коридора; HGB: 120 итераций, depth 2, learning rate 0.05, leaf 40, L2 2, seed 20260903, **early_stopping=False**. Здесь нет нового квантильного preprocessing, missing-indicator признаков, Halyk, OXR или локальной KZT адаптации.

Вызван исходный `temporal_split` и отдельно проверена дата созревания каждой метки как h-е следующее наблюдение внутри валюты. Train labels строго до начала calibration; calibration labels строго до cutoff. Annual test сохраняет старый horizon-specific purge. Поэтому H3 содержит на две даты в год больше H5; сравнения горизонтов на полном различающемся наборе дат не выдаются за парные.

NOW target — текущий reference R не хуже каждого следующего R на своём горизонте; R = RUB за единицу валюты получателя. Для каждого h заново рассчитаны mean-forward bps, симметричный utility и regret. H3 модель обучена на H3 labels; H5 labels и scores не переименованы в H3.

`probability` и `original_v3_probability` — исходная **общая для пяти валют Platt calibration** на предыдущих 12 месяцах. Сохраняется original fallback: monotone positive slope и улучшение validation Brier, иначе identity. Дополнительно экспортирована `raw_probability`. Root может применять свои одинаковые политики к raw, но строгий baseline 0.5 использует именно original pooled-calibrated probability.

## Original policy и строгий 0.5

`candidate_signal` / `original_candidate_signal` — старые per-corridor thresholds, выбранные на прошлой calibration. `signal` / `original_signal` — исходный общий portfolio stream. Эти поля нужны для исторического parity.

`strict05_candidate_signal` — probability ≥ 0.5 и cooldown 3 эффективных CBR-наблюдения; разница session ordinals должна быть строго больше 3. `strict05_signal` дополнительно применяет исходный общий portfolio cooldown и нейтральный tie-break. Для test начальное состояние воспроизводится по полной предыдущей 12-месячной истории с тем же порогом 0.5. Здесь нет скрытой локальной перекалибровки или замены порога частотным.

| train_horizon | year | policy | dates | contacts | hit_rate | lift_standardized | forward_delta_bps | mean_all_observed_week_coverage | brier |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2024 | candidate_signal | 245 | 54 | 0.425926 | 1.272584 | 2.303364 | 0.921569 | 0.205421 |
| 3 | 2024 | strict05_candidate_signal | 245 | 41 | 0.439024 | 1.311719 | 18.936700 | 0.764706 | 0.205421 |
| 3 | 2025 | candidate_signal | 244 | 52 | 0.384615 | 1.466346 | 40.244094 | 0.921569 | 0.178073 |
| 3 | 2025 | strict05_candidate_signal | 244 | 21 | 0.476190 | 1.815476 | 75.004494 | 0.392157 | 0.178073 |
| 3 | 2026 | candidate_signal | 158 | 35 | 0.457143 | 1.078038 | 24.147137 | 0.939394 | 0.229475 |
| 3 | 2026 | strict05_candidate_signal | 158 | 19 | 0.684211 | 1.613511 | 77.321777 | 0.515152 | 0.229475 |
| 5 | 2024 | candidate_signal | 243 | 47 | 0.361702 | 1.352209 | 20.726037 | 0.843137 | 0.192563 |
| 5 | 2024 | strict05_candidate_signal | 243 | 34 | 0.500000 | 1.869231 | 83.331965 | 0.666667 | 0.192563 |
| 5 | 2025 | candidate_signal | 242 | 44 | 0.431818 | 2.177083 | 56.462709 | 0.803922 | 0.156469 |
| 5 | 2025 | strict05_candidate_signal | 242 | 12 | 0.416667 | 2.100694 | 86.958441 | 0.215686 | 0.156469 |
| 5 | 2026 | candidate_signal | 156 | 34 | 0.441176 | 1.207430 | 31.604299 | 0.939394 | 0.234980 |
| 5 | 2026 | strict05_candidate_signal | 156 | 5 | 0.600000 | 1.642105 | 133.588278 | 0.121212 | 0.234980 |

Высокий lift строгого H5 сопровождается низкой частотой: в KZT 2026 всего 5 контактов и 12.12% наблюдаемых недель с 1–2 контактами. Отдельный H3 даёт 19 контактов и 51.52% недель; это тоже ниже цели 85%. Такой trade-off нужно показывать одновременно с качеством.

## Почему у коллеги H3 = 65% и lift 1.921938637

Цифра воспроизведена точно на **727 KZT датах 2023–2025**: берутся исторические H5 probabilities, применяется 0.5 и cooldown 3, сохраняются **60 H5-сигналов**; их исходы пересчитываются на H3. Получается **39/60 = 65%**, lift **1.921938637** и forward delta **81.910520 bps**. Последние две метрики стандартизированы по годам с весами числа сигналов, как исходный V3.

| method | contacts | hits | hit_rate | lift_standardized | lift_unstandardized | forward_delta_bps | mean_all_observed_week_coverage |
|---|---|---|---|---|---|---|---|
| threshold_only | 96 | 62 | 0.645833 | 1.890625 | 1.916412 | 84.901076 | 0.400392 |
| cooldown_replay_test_annual | 60 | 39 | 0.650000 | 1.921939 | 1.928776 | 81.910520 | 0.387451 |
| cooldown_replay_test_continuous | 60 | 39 | 0.650000 | 1.921939 | 1.928776 | 81.910520 | 0.387451 |

Это корректная **sensitivity фиксированной H5-политики к горизонту оценки**, если явно указать происхождение. Это не результат отдельно обученной H3 модели. Неизменная частота здесь механически следует из неизменной маски H5. Простое threshold-only без cooldown даёт 96 дат и другую метрику. Нестандартизированный pooled lift тоже другой: 1.928776 для 60 сигналов, поэтому способы агрегации нельзя смешивать.

`matched_horizon_rescore.csv` отдельно показывает настоящий H3→H3, H5→H5 и оба cross-horizon rescore на одном пересечении date/corridor. Пример KZT 2026, строго 156 общих дат:

| train_horizon | evaluate_horizon | contacts | hits | hit_rate | lift_standardized | forward_delta_bps | mean_all_observed_week_coverage |
|---|---|---|---|---|---|---|---|
| 3 | 3 | 19 | 13 | 0.684211 | 1.617225 | 78.149055 | 0.515152 |
| 3 | 5 | 19 | 12 | 0.631579 | 1.728532 | 99.446705 | 0.515152 |
| 5 | 3 | 5 | 4 | 0.800000 | 1.890909 | 109.572609 | 0.121212 |
| 5 | 5 | 5 | 3 | 0.600000 | 1.642105 | 133.588278 | 0.121212 |

Эта таблица не предлагает выбирать горизонт по 2026. Результаты 2026 уже изучались в предыдущем исследовании; здесь проводится воспроизведение, а не новый независимый holdout.

## Контракт данных и warmup

`raw_predictions.csv.gz` содержит split `validation`, `history`, `test`; ключ — `(config_id=v3, train_horizon, fold_test_year, split, date, corridor)`. В history сохраняется полный прошлый год, включая незрелый хвост; target/forward/symmetric/regret в нём masked. Вероятности этого хвоста доступны для causal policy-state replay. Labels и utility test — только offline evaluation fields.

Отдельный `warmup.csv.gz` содержит **63 последних PANEL даты строго до начала calibration**, включая horizon-purged training tail: 315 строк для каждой из шести моделей, всего 1890. Используются те же frozen checkpoint и original pooled calibrator, все outcomes masked. Флаг `in_sample_training_warmup` различает реально использованные в обучении строки и последние 15/25 строк H3/H5, исключённые purge. Это технический warmup состояния для root rolling-policy, а не OOT validation и не новое доказательство качества. Исходная V3 policy и её parity не зависят от этого дополнительного warmup.

## Проверки и воспроизведение

Verifier **PASS, 7 блоков**: SHA исходников, фактические сроки maturity, независимое вычисление labels/utility, checkpoint parity, ручная Platt formula, повторные fits после отравления будущих features/незрелых train labels, независимый 0.5 scheduler для corridor и portfolio, отсутствие зависимости scheduler от будущих outcomes, warmup с полным purged tail. Дополнительно historical parity H3/H5 по всем трём годам — PASS.

Шесть основных HGB fits заняли суммарно около **1.95 секунды** на одном потоке. Команды и схема — `README.md`. Общий рынок на одну дату не превращается в пять независимых рыночных наблюдений: для статистических сравнений нужны парные date/month blocks. Reference-utility не является сбережениями по исполненной банковской котировке.
