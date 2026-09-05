# KZT Halyk: устойчивость к задержке и две даты фиксации модели

**Главная поправка к прежней интерпретации V4:** train-lag1 → test-lag2 даёт Brier **0.181404** на development 2023–2025. Значение **0.186250** было получено при отдельном переобучении на lag2. Это разные эксперименты. Настоящая задержка входных данных исходной lag1-модели ухудшает Brier с 0.176324 до 0.181404, но он остаётся ниже long V3 0.182705. Из старого retrain-lag2 результата нельзя было заключать, что тот же deployed lag1 checkpoint обязательно теряет преимущество.

**Улучшения без потери качества не найдено.** Предобъявленный minimax выбор из четырёх новых защитных методов — `minimax_shrink_v3`. Он снижает worst-view Brier с 0.181404 до 0.180753 (на 0.000652), но повышает normal Brier на 0.001903. Ни один новый метод не выполнил одновременно normal≤original lag1 и delay≤long V3. Исходный Halyk lag1 проходит этот более мягкий delay-vs-V3 критерий сам; это не означает отсутствие чувствительности к задержке.

У простого контроля **long V3 + KZT-only calibration** Brier 0.180435 в обоих режимах. По worst-view он даже немного лучше новой minimax blend 0.180753. Поэтому выбор внутри robust-family не означает превосходства над всеми простыми controls. Приоритет зависит от того, важнее ли normal качество или гарантируемая в пределах проверенных сценариев нечувствительность к Halyk.

Все результаты — **ретроспективное исследование уже просмотренной истории**, не prospective holdout, не causal uplift клиента и не доказательство заработка на исполнимой банковской котировке. Старые данные и алгоритмы V3/V4 неизменны. Независимый verifier проверяет прежний V4 seal из `continuation/previous_v4_manifest.json`: только две обновляемые root-страницы README/REPORT сверяются по byte-identical backups, все остальные старые файлы — по текущим оригиналам. V3 проверяется полностью по прежнему manifest.

## Что предобъявлено

`PROTOCOL.md` записан до вычисления continuation результатов. Один target h5 KZT, один long120m HGB с прежними параметрами. Восемь правил:

| Правило | Fit / calibration | При normal и delayed deployment |
|---|---|---|
| V3 global calibration | Pooled HGB, предыдущая validation всех 5 коридоров | Halyk не используется |
| V3 local calibration | Тот же HGB, предыдущая KZT validation | Halyk не используется |
| Halyk lag1 | Исходный V4 residual_shrink fitL1/calL1 | Тот же checkpoint получает L1 либо L2 |
| Retrain lag2 reference | Отдельный fitL2/calL2 | Получает L2 либо L3; не равен mismatch предыдущей строки |
| Lag augmentation | По детерминированному date hash training дата выбирает L1 либо L2 | Одна строка на market-date/corridor, нет фиктивного удвоения markets |
| Feature dropout | На 25% training dates все Halyk features заменены missing | Обычные доступные L1 либо L2 features на inference |
| Lag ensemble | Среднее двух отдельно обученных L1/L2 моделей | Views(L1,L2) либо(L2,L3) |
| Minimax shrink to V3 | Alpha∈{0,.25,.5,.75,1} по worst предыдущей validation | Смешиваются calibrated HalykL1 и global-calibrated V3 |

Pooled HGB: depth2,120 iterations, learning rate.05,leaf40,L2=2,early_stopping=False. KZT continuation:40 Newton logloss stumps,depth1,leaf60,learning rate.025, frozen **float64** pooled probabilities. Для augmentation/dropout stage-2 weight выбирается по worst validation L1/L2. Robust Platt calibrator использует две возможные views одних validation дат; это loss augmentation, **не новые независимые labels**. Пороги candidates выбираются только на прошлой нормальной validation, затем сохраняются.

Primary robust-family rule выбирается по минимальному max(normal,delay) Brier development 2023–2025; selection сохранена **до нового вычисления 2026**. Никакие настройки не подбирались по новым 2026 test outcomes. Уже известный ранее 2026 всё равно не становится нетронутым holdout.

## Сравнение на всех 727 development KZT датах

| rule | normal_brier | delayed_brier | worst_brier | delay_penalty |
|---|---|---|---|---|
| age_fallback_minimax | 0.176764 | 0.181889 | 0.181889 | 0.005125 |
| age_fallback_v3 | 0.176848 | 0.181512 | 0.181512 | 0.004664 |
| feature_dropout | 0.181399 | 0.187654 | 0.187654 | 0.006255 |
| halyk_l1 | 0.176324 | 0.181404 | 0.181404 | 0.005080 |
| halyk_l2_retrained | 0.186250 | 0.188400 | 0.188400 | 0.002150 |
| lag_augmentation | 0.181122 | 0.185926 | 0.185926 | 0.004803 |
| lag_ensemble | 0.180418 | 0.184706 | 0.184706 | 0.004288 |
| minimax_shrink_v3 | 0.178228 | 0.180753 | 0.180753 | 0.002525 |
| v3_long_globalcal | 0.182705 | 0.182705 | 0.182705 | 0.000000 |
| v3_long_localcal | 0.180435 | 0.180435 | 0.180435 | 0.000000 |

Возрастные fallback arms — отдельно предобъявленный после основного результата follow-up, подробно ниже. Они не входят в первоначальный robust-family selection.

## Качество прогноза и качество выбранных дат

| rule | delay | brier | candidate_count | candidate_quality | candidate_lift_standardized | forward_delta_bps |
|---|---|---|---|---|---|---|
| feature_dropout | 0 | 0.181399 | 136 | 0.448529 | 1.621896 | 53.690763 |
| feature_dropout | 1 | 0.187654 | 136 | 0.404412 | 1.483643 | 32.492720 |
| halyk_l1 | 0 | 0.176324 | 134 | 0.462687 | 1.679948 | 50.229046 |
| halyk_l1 | 1 | 0.181404 | 131 | 0.442748 | 1.612364 | 49.279143 |
| halyk_l2_retrained | 0 | 0.186250 | 141 | 0.418440 | 1.490372 | 42.012259 |
| halyk_l2_retrained | 1 | 0.188400 | 143 | 0.419580 | 1.490541 | 39.075704 |
| lag_augmentation | 0 | 0.181122 | 146 | 0.424658 | 1.507582 | 39.665466 |
| lag_augmentation | 1 | 0.185926 | 148 | 0.371622 | 1.317028 | 30.030585 |
| lag_ensemble | 0 | 0.180418 | 140 | 0.464286 | 1.658648 | 47.598346 |
| lag_ensemble | 1 | 0.184706 | 138 | 0.456522 | 1.629863 | 45.174875 |
| minimax_shrink_v3 | 0 | 0.178228 | 152 | 0.388158 | 1.375738 | 30.091079 |
| minimax_shrink_v3 | 1 | 0.180753 | 153 | 0.392157 | 1.387160 | 27.349719 |
| v3_long_globalcal | 0 | 0.182705 | 149 | 0.422819 | 1.477837 | 29.971421 |
| v3_long_globalcal | 1 | 0.182705 | 149 | 0.422819 | 1.477837 | 29.971421 |
| v3_long_localcal | 0 | 0.180435 | 149 | 0.422819 | 1.477837 | 29.971421 |
| v3_long_localcal | 1 | 0.180435 | 149 | 0.422819 | 1.477837 | 29.971421 |
| age_fallback_minimax | 0 | 0.176764 | 136 | 0.448529 | 1.624883 | 45.400838 |
| age_fallback_minimax | 1 | 0.181889 | 141 | 0.425532 | 1.532633 | 42.719205 |
| age_fallback_v3 | 0 | 0.176848 | 136 | 0.448529 | 1.624883 | 45.400838 |
| age_fallback_v3 | 1 | 0.181512 | 143 | 0.391608 | 1.404077 | 37.714181 |

Brier оценивается на **всех** датах. Candidate quality — только среди отправленных моделью сигналов. Это разные метрики; рост hit rate при меньшем числе сигналов не обязательно улучшает прогноз.

`candidate_lift_standardized` полностью совместим с V3: число hits делится на сумму annual-cell baseline hit probabilities по выбранным датам. Каждая year×corridor cell получает именно её candidate-count вес. `forward_delta_bps` — средняя разность forward return кандидата и baseline mean той же cell. В CSV сохранены `candidate_lift_unstandardized` и `forward_bps_absolute`; они не подменяют стандартизированные сравнения. На single-cutoff KZT различие lift исчезает, но при pooling2023–2025 оно существенно. Отдельный численный тест воспроизводит V3 standardized lift и forward delta.

Candidate policy сохраняет прежний cooldown: не чаще одной возможности через более 3effective CBR sessions. Порог выбирается на prior validation по прежней cadence/loss функции. Это один KZT candidate stream; общий bank CRM/user cap здесь не моделируется.

## January freeze и March freeze

| cutoff | train_start | train_end | train_max_label_available | calibration_start | calibration_end | calibration_max_label_available | history_end | blend_alpha |
|---|---|---|---|---|---|---|---|---|
| 2023-01-01 | 2012-01-11 | 2021-12-24 | 2021-12-31 | 2022-01-11 | 2022-12-24 | 2022-12-31 | 2022-12-31 | 0.250000 |
| 2024-01-01 | 2013-01-10 | 2022-12-24 | 2022-12-31 | 2023-01-10 | 2023-12-23 | 2023-12-30 | 2023-12-30 | 0.000000 |
| 2025-01-01 | 2014-01-01 | 2023-12-23 | 2023-12-30 | 2024-01-10 | 2024-12-24 | 2024-12-29 | 2024-12-29 | 0.750000 |
| 2026-01-01 | 2015-01-01 | 2024-12-24 | 2024-12-29 | 2025-01-10 | 2025-12-24 | 2025-12-31 | 2025-12-31 | 0.250000 |
| 2026-03-01 | 2015-03-03 | 2025-02-21 | 2025-02-28 | 2025-03-01 | 2026-02-20 | 2026-02-28 | 2026-02-28 | 0.250000 |

- **January2026 freeze:** training заканчивается 2024-12-24, последние train labels созревают до 2025. Calibration берёт 2025 outcomes, завершившиеся к 2025-12-31. Никаких 2026 labels в fit.
- **March1,2026 freeze:** rolling calibration `[2025-03-01,2026-03-01)`, отдельно training10years до 2025-03-01. Calibration может использовать January/February2026 observations только с известными к cutoff labels: последний calibration observation2026-02-20, его h5 заканчивается 2026-02-28. Это другая freeze scheme, не January-модель с незаметно обновлённым порогом.
- Evaluator **не вызывает `core.run_fold`**, потому что его `validation_history` привязана к январю. History cooldown явно равна `[cutoff−12months,cutoff)`, включая незрелый label tail только для past scores/state, без использования его outcomes.
- One-day delay начинается на cutoff. Normal и delayed world получают одну и ту же исходную cooldown state, рассчитанную на нормальной предшествующей истории. Это mismatch stress после фиксации модели, а не переобучение в другом мире.
- Самостоятельный V3 baseline реально переобучен для каждой freeze scheme. January и March сравниваются на общей сетке **2026-03-05—2026-08-25,120 KZT дат**. March1 policy также имеет 122 даты начиная 2026-03-03; две до March5 учитываются в его carry state. January policy содержит 156 зрелых 2026 дат начиная 2026-01-13.
- Frozen CBR source заканчивается **2026-09-01**, последний зрелый h5 target — **2026-08-25**. Это не полные шесть месяцев до конца сентября; отсутствующий хвост не заполняется.

### Одинаковые 120 дат после March5

| cutoff | rule | delay | brier | candidate_count | candidate_quality | candidate_lift_standardized | forward_delta_bps |
|---|---|---|---|---|---|---|---|
| 2026-01-01 | halyk_l1 | 0 | 0.228392 | 27 | 0.444444 | 1.134752 | 49.983815 |
| 2026-01-01 | halyk_l1 | 1 | 0.243261 | 28 | 0.392857 | 1.003040 | 23.604022 |
| 2026-01-01 | minimax_shrink_v3 | 0 | 0.234606 | 28 | 0.535714 | 1.367781 | 42.770377 |
| 2026-01-01 | minimax_shrink_v3 | 1 | 0.238117 | 29 | 0.448276 | 1.144534 | 12.158943 |
| 2026-01-01 | v3_long_globalcal | 0 | 0.238843 | 27 | 0.481481 | 1.229314 | 31.432106 |
| 2026-01-01 | v3_long_globalcal | 1 | 0.238843 | 27 | 0.481481 | 1.229314 | 31.432106 |
| 2026-01-01 | v3_long_localcal | 0 | 0.246721 | 27 | 0.481481 | 1.229314 | 31.432106 |
| 2026-01-01 | v3_long_localcal | 1 | 0.246721 | 27 | 0.481481 | 1.229314 | 31.432106 |
| 2026-03-01 | halyk_l1 | 0 | 0.225637 | 27 | 0.444444 | 1.134752 | 55.536215 |
| 2026-03-01 | halyk_l1 | 1 | 0.238453 | 29 | 0.482759 | 1.232575 | 15.063117 |
| 2026-03-01 | minimax_shrink_v3 | 0 | 0.234444 | 28 | 0.464286 | 1.185410 | 28.283301 |
| 2026-03-01 | minimax_shrink_v3 | 1 | 0.237466 | 28 | 0.464286 | 1.185410 | 24.570766 |
| 2026-03-01 | v3_long_globalcal | 0 | 0.238962 | 27 | 0.444444 | 1.134752 | 28.576172 |
| 2026-03-01 | v3_long_globalcal | 1 | 0.238962 | 27 | 0.444444 | 1.134752 | 28.576172 |
| 2026-03-01 | v3_long_localcal | 0 | 0.243931 | 27 | 0.444444 | 1.134752 | 28.576172 |
| 2026-03-01 | v3_long_localcal | 1 | 0.243931 | 27 | 0.444444 | 1.134752 | 28.576172 |

Предварительно выбранный minimax blend на January freeze:0.234606 normal / 0.238117 delay против 0.238843V3. На March freeze:0.234444/0.237466 против 0.238962same-cutoffV3. Небольшой положительный результат delay protection в 2026 описывается **как retrospective confirmation**, с неопределённостью ниже. Halyk lag1 на January freeze без delay заметно лучше blend, а при delay хуже V3. Это реальный tradeoff.

Полные 2026 таблицы всех arms сохранены в `metrics_by_cutoff.csv` и `combined_common_march5.csv`. Не выбираем удачный dropout или другой метод по тому, что он лучше выглядит на уже увиденном 2026.

## Age-aware fallback: полезная гипотеза, отрицательный итог

После основного результата добавлены только два правила (`AGE_FOLLOWUP_PROTOCOL.md`): при stale/missing Halyk использовать V3 либо уже выбранный minimax. Age — максимальный calendar age **доступной observation date** для personalRUB,legalRUB,personalUSD; отсутствие любого необходимого feed даёт unavailable. Сам gate получает только ages и probabilities, **не normal / delayed flag**. Порог age>2 или>3calendar days выбирается по worst purged prior-validation Brier. Никакой дополнительной Platt-калибровки после переключения нет: на свежих датах probability точно равна исходной HalykL1.

Development fallback→V3:0.176848 normal / 0.181512 delay; fallback→minimax:0.176764/0.181889. Оба варианта немного хуже исходного HalykL1 в обоих режимах. Это не доказательство бесполезности настоящего feed-health gate; age из исторического chart snapshot лишь proxy, а публикационные timestamps и delivery logs отсутствуют. **Расширять поиск до положительного результата не стали.**

| cutoff | rule | delay | age_cutoff | fallback_share | median_quote_age |
|---|---|---|---|---|---|
| 2023-01-01 | age_fallback_minimax | 0 | 3 | 0.024793 | 1.000000 |
| 2023-01-01 | age_fallback_minimax | 1 | 3 | 0.214876 | 2.000000 |
| 2023-01-01 | age_fallback_v3 | 0 | 3 | 0.024793 | 1.000000 |
| 2023-01-01 | age_fallback_v3 | 1 | 3 | 0.214876 | 2.000000 |
| 2024-01-01 | age_fallback_minimax | 0 | 2 | 0.024691 | 1.000000 |
| 2024-01-01 | age_fallback_minimax | 1 | 2 | 0.226337 | 2.000000 |
| 2024-01-01 | age_fallback_v3 | 0 | 2 | 0.024691 | 1.000000 |
| 2024-01-01 | age_fallback_v3 | 1 | 2 | 0.226337 | 2.000000 |
| 2025-01-01 | age_fallback_minimax | 0 | 3 | 0.016529 | 1.000000 |
| 2025-01-01 | age_fallback_minimax | 1 | 3 | 0.210744 | 2.000000 |
| 2025-01-01 | age_fallback_v3 | 0 | 3 | 0.016529 | 1.000000 |
| 2025-01-01 | age_fallback_v3 | 1 | 3 | 0.210744 | 2.000000 |
| 2026-01-01 | age_fallback_minimax | 0 | 3 | 0.000000 | 1.000000 |
| 2026-01-01 | age_fallback_minimax | 1 | 3 | 0.166667 | 2.000000 |
| 2026-01-01 | age_fallback_v3 | 0 | 3 | 0.000000 | 1.000000 |
| 2026-01-01 | age_fallback_v3 | 1 | 3 | 0.166667 | 2.000000 |
| 2026-03-01 | age_fallback_minimax | 0 | 2 | 0.000000 | 1.000000 |
| 2026-03-01 | age_fallback_minimax | 1 | 2 | 0.196721 | 2.000000 |
| 2026-03-01 | age_fallback_v3 | 0 | 2 | 0.000000 | 1.000000 |
| 2026-03-01 | age_fallback_v3 | 1 | 2 | 0.196721 | 2.000000 |

Эта идея появилась после просмотра новых 2026 результатов и прямо помечена как post-inspection follow-up. Выбор age threshold использует только предыдущую validation;2026 test labels не участвуют, но историческая исследовательская зависимость сохраняется.

## Неопределённость на реальных рыночных датах

Paired circular block bootstrap,1500 replicates, block10/20/40 effective CBR dates. В каждой паре одни и те же даты и labels; отрицательная delta означает меньший Brier у A. Fixed policies held constant, **без refit и без поправки на весь накопленный поиск**; интервалы диагностические, не confirmatory significance.

| scope | rule_a | delay_a | rule_b | delay_b | dates | delta_brier | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| development | halyk_l1 | 0 | v3_long_globalcal | 0 | 727 | -0.006380 | -0.015772 | 0.002537 |
| development | halyk_l1 | 1 | v3_long_globalcal | 0 | 727 | -0.001300 | -0.010494 | 0.008028 |
| development | minimax_shrink_v3 | 0 | halyk_l1 | 0 | 727 | 0.001903 | -0.005558 | 0.010040 |
| development | minimax_shrink_v3 | 1 | halyk_l1 | 1 | 727 | -0.000652 | -0.008412 | 0.007977 |
| development | minimax_shrink_v3 | 1 | v3_long_globalcal | 0 | 727 | -0.001952 | -0.005405 | 0.000858 |
| development | v3_long_localcal | 0 | minimax_shrink_v3 | 1 | 727 | -0.000318 | -0.002883 | 0.001988 |
| common_march5_2026-01-01 | halyk_l1 | 0 | v3_long_globalcal | 0 | 120 | -0.010451 | -0.027960 | 0.005699 |
| common_march5_2026-01-01 | halyk_l1 | 1 | v3_long_globalcal | 0 | 120 | 0.004418 | -0.015791 | 0.022721 |
| common_march5_2026-01-01 | minimax_shrink_v3 | 0 | halyk_l1 | 0 | 120 | 0.006214 | -0.005832 | 0.019228 |
| common_march5_2026-01-01 | minimax_shrink_v3 | 1 | halyk_l1 | 1 | 120 | -0.005144 | -0.018924 | 0.009518 |
| common_march5_2026-01-01 | minimax_shrink_v3 | 1 | v3_long_globalcal | 0 | 120 | -0.000725 | -0.006371 | 0.003766 |
| common_march5_2026-01-01 | v3_long_localcal | 0 | minimax_shrink_v3 | 1 | 120 | 0.008604 | -0.002409 | 0.019746 |
| common_march5_2026-03-01 | halyk_l1 | 0 | v3_long_globalcal | 0 | 120 | -0.013324 | -0.029265 | 0.000370 |
| common_march5_2026-03-01 | halyk_l1 | 1 | v3_long_globalcal | 0 | 120 | -0.000508 | -0.018658 | 0.015844 |
| common_march5_2026-03-01 | minimax_shrink_v3 | 0 | halyk_l1 | 0 | 120 | 0.008807 | -0.001312 | 0.020651 |
| common_march5_2026-03-01 | minimax_shrink_v3 | 1 | halyk_l1 | 1 | 120 | -0.000987 | -0.013756 | 0.012305 |
| common_march5_2026-03-01 | minimax_shrink_v3 | 1 | v3_long_globalcal | 0 | 120 | -0.001496 | -0.006294 | 0.002365 |
| common_march5_2026-03-01 | v3_long_localcal | 0 | minimax_shrink_v3 | 1 | 120 | 0.006465 | -0.001474 | 0.014752 |

Все приведённые 95% интервалы при block=20 включают ноль. В частности, development преимущество minimax при delay против исходного HalykL1 составляет −0.000652, но интервал равен [−0.008412; 0.007977]. Для delayed minimax против V3 на общих датах 2026 года интервалы также включают ноль: January freeze [−0.006371; 0.003766], March freeze [−0.006294; 0.002365]. Знак среднего здесь не является статистическим подтверждением превосходства.

727 development и 120 common 2026 дат — реальные исходные рыночные observations, зависимые во времени. Training lag augmentation не увеличивает их число. Нельзя считать две lag views или несколько models независимыми новыми рынками.

## Независимые алгоритмические проверки

Статус: **PASS**, 7 блоков. `verify.py` проверяет:

1. Все прежние sealed V3/V4 данные, алгоритмы и source fingerprints неизменны; две root-страницы V4 проверены в byte-identical backups, прежний V4 manifest сохранён отдельно от обновляемого root seal.
2. Сохранённые checkpoints воспроизводят probabilities через **независимую** сборку HGB + Newton corrections + Platt coefficients; candidate stream и cutoff-specific history state воспроизводятся отдельным циклом.
3. Annual2023–2026 controls совпадают с frozenV3/V4 на 883KZT датах: probability error≤1.12e−16,0candidate mismatches. Это включает originalHalykL1 и отдельно retrainedL2, поэтому их сравнение не меняет даты/семантику модели.
4. Все train/calibration labels завершаются до следующего cutoff; end dates проверяются независимо через session positions, а не доверием к полю label_available_date.
5. После cutoff все market features×100, все будущие и ещё не созревшие labels инвертируются; повторный fit сохраняет модели, calibrated predictions, shrink weights, candidate thresholds и age-gate parameters. Проверены January и March отдельно.
6. Изменение будущих rawHalyk values не меняет прошлые признаки и observation-age metadata для lag1/lag2/lag3. Future test rows/outcomes не меняют prefix candidates.
7. Неравные годовые base rates в ручном fixture дают standardizedlift1.6 вместо naive2.5; реальный V3 lift/forward delta воспроизводится.

`results/model_receipts.json` содержит training/calibration timestamps, чекпойнты, SHA256 и feature fingerprints; `results/selection.json` отделяет development selection от 2026. `results/verification.json` содержит детали проверок. Snapshot as-of joining и poison tests не доказывают действительную историческую доступность опубликованных данных: archive/vintage publication time остаётся ограничением.

## Практическое решение

- Исправить прежнюю формулировку lag sensitivity: **retrainedL2 и trainL1/testL2 mismatch — разные величины**.
- Если нужна лучшая normal Brier в проверенной истории, исходный HalykL1 остаётся сильнее новых защит. Нельзя обещать отсутствие просадки при задержке.
- Если важнее полная независимость от Halyk feed, простой longV3 с KZT calibration — обязательный control; он не уступает новой minimax worst-score на development.
- Minimax blend — прозрачный компромисс: меньшая delay penalty ценой normal качества. Он пригоден для следующей prospective проверки с надёжными timestamps, но не получает production promotion из этого backtest.
- Age fallback сохранили как отрицательный эксперимент. Новых порогов после этого не подбирали.
- Пользовательский conversion, деньги банка и реальные execution prices не оценены. All-in live quotes, feed monitoring и prospective holdout нужны отдельно.
