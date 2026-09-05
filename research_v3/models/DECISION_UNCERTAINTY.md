# Независимая проверка uncertainty и policy frontier

Выполнено 2026-09-05. Код: `decision_uncertainty.py`; основной чужой evaluator `assess.py` не изменён. Данные — его сохранённые probabilities. Новые модели не обучались.

## Исправление семантики интервалов

`assess.py:bootstrap_paired` корректно использует общие временные blocks и paired model losses. Его `forward_policy_ci` явно фиксировал годовые corridor baselines до resampling. Это условный интервал с другой семантикой, чем полный random-day baseline uncertainty. Поэтому здесь одновременно ресэмплируются exposure и signal outcomes, заново оцениваются cell baselines, и только затем рассчитываются lift/forward/symmetric advantage.

В каждом из 10,000 draws целые календарные месяцы выбираются с возвращением внутри года. Все пять коридоров даты и **все сравниваемые policies** получают одни и те же bootstrap multiplicities. Для cell j рассчитываются `n_j`, `hit_j`, `forward_sum_j`, `symmetric_sum_j`; для policy p — свои `signal_n_pj`, hits и sums. Затем:

```text
lift_p = Σ signal_hits_pj / Σ(signal_n_pj × hit_j/n_j)
forward_advantage_p = Σ(signal_forward_pj − signal_n_pj × forward_sum_j/n_j)
                      / Σ signal_n_pj
```

Symmetric advantage аналогичен. Парный policy delta — разность этих метрик **в одном draw**, с собственными counts каждой policy. Уменьшать denominator либо механически уравнивать counts нельзя. Более редкая policy вправе выбрать другие даты, но её более высокая precision не доказывает преимущество при одинаковой частоте.

`symmetric_bps` старого `assess.py` — абсолютное среднее у сигналов; новый `symmetric_delta_bps` — matched-cell advantage. Их нельзя сравнивать как одну величину. Оба представления допустимы при правильной подписи.

## Численный tradeoff, development 2023–2025

Все показатели относятся к matured-label corridor candidate stream. Bps — official reference; не клиентская экономия. Пороги .50/.65 были рассмотрены после изучения истории: **это exploratory frontier, не selection-corrected confirmation**.

| Policy | Candidates | Lift [95% block CI] | Forward advantage, bps [95% CI] | Средняя частота/corridor/week | Min cell weeks с 1–2 |
|---|---:|---:|---:|---:|---:|
| Incumbent legacy | 663 | 1.435 [1.313;1.560] | +47.69 [33.48;62.63] | .887 | 66.7% |
| Incumbent P≥.50 | 288 | 1.782 [1.486;2.145] | +81.70 [57.10;106.97] | .385 | 10.4% |
| Incumbent P≥.65 | 123 | 1.913 [1.490;2.509] | +109.15 [78.54;142.69] | .164 | 0.0% |
| Train120m legacy | 723 | 1.371 [1.236;1.522] | +40.08 [26.57;54.64] | .967 | 73.5% |
| Train120m P≥.50 | 351 | 1.786 [1.562;2.038] | +95.35 [73.01;118.63] | .469 | 20.4% |
| Train120m P≥.65 | 139 | 2.122 [1.737;2.607] | +108.25 [82.04;133.84] | .186 | 0.0% |

Более строгий threshold улучшает качество момента ценой очень редких контактов. Нельзя писать «добились lift2.12 вместо1.44» без рядом стоящего уменьшения frequency .887→.186/week. Для opt-in opportunistic уведомления это может быть обоснованная policy; для обещания стабильных 1–2 сигналов/неделю — другой продукт.

Парные deltas против incumbent legacy:

| Policy | Δlift [95% CI] | Δforward bps [95% CI] |
|---|---:|---:|
| Incumbent P≥.50 | +.347 [.078;.678] | +34.01 [10.52;57.47] |
| Train120m legacy | −.064 [−.194;.067] | −7.61 [−18.07;2.51] |
| Train120m P≥.50 | +.351 [.171;.566] | +47.66 [29.18;67.30] |
| Train120m P≥.65 | +.687 [.338;1.127] | +60.57 [37.12;83.42] |

Эти intervals учитывают совместную зависимость outcomes и baseline, но не весь процесс выбора порогов/моделей. Положительная нижняя граница не возвращает исследованию untouched-test статус.

## Частичный 2026 и редкие события

Incumbent legacy: 168 candidates, lift1.375, +37.77bps. При P≥.50 — 46 candidates, lift2.075, +98.41bps; Train120m P≥.50 — 54 candidates, lift1.853, +92.54bps. Период уже просмотрен, поэтому остаётся диагностикой.

При incumbent P≥.65 в 2026 всего **два** corridor candidates, при Train120m P≥.65 — **семь**. Блочный bootstrap на такой support не может достоверно оценить частоту неизвестных ошибок. Часть draws не содержит signals: соответствующие ratios остаются NaN, не заменяются фиктивным denominator1. Указаны `*_valid_draws`, `zero_signal_draw_fraction`, `interval_conditioning`, `n_signal_dates`, `n_signal_month_blocks` и sparse-support flags. Численный interval таких строк условен на непустые draws; его нельзя защищать как номинальный unconditional confidence interval.

Порог предупреждения 20 signal dates / 6 active signal months — прозрачная диагностическая эвристика, не универсальная теорема и не новый обязательный gate кейса.

## Warm start и чувствительность

Fixed-threshold policies стартуют с пустым cooldown state в первой имеющейся дате января2023 и затем продолжают обработку всех сохранённых prediction dates. Legacy masks несут состояние, рассчитанное старым validation replay. В сохранённых файлах отсутствуют pre-2023 scores и часть year-end строк с незрелой меткой; полностью восстановить одинаковый operational state невозможно.

Проверка чувствительности удаляет первые 20 **оценочных** дат каждого outer year, сохраняя уже выбранные masks. Development:

| Policy | Исходный lift / forward | После trim20 |
|---|---:|---:|
| Incumbent legacy | 1.435 / +47.69bps | 1.434 / +48.17bps |
| Incumbent P≥.50 | 1.782 / +81.70bps | 1.758 / +81.72bps |
| Train120m P≥.50 | 1.786 / +95.35bps | 1.780 / +96.73bps |

Вывод о tradeoff сохраняется на development. Это sensitivity, не доказательство полного PIT replay. 2026 реагирует сильнее: incumbent P≥.50 падает с lift2.075 до1.845, Train120m P≥.50 с1.853 до1.635. Поэтому нельзя скрывать состав частичного периода.

## Артефакты и выполненные проверки

Запуск: `python research_v3/models/decision_uncertainty.py`. Выходы: `decision_policy_intervals.csv`, `decision_policy_paired_deltas.csv`, `decision_policy_manifest.json` с code/output hashes. В paired CSV есть сравнения с incumbent legacy, с собственной legacy policy и с incumbent при том же threshold.

Проверено равенство всех date×corridor outcome rows между тремя сравниваемыми prediction-файлами. Аналитический sanity check: selection всех дней даёт lift1 и standardized forward/symmetric advantage0; selection нуля дней оставляет ratios неопределёнными. Точечные incumbent lift1.4351907 и forward47.685646 воспроизводят V2.

Cadence здесь считается по календарному span **matured-label exposure**, не по всем фактическим operational weeks; численно она поэтому может слегка отличаться от иной weighted-frequency агрегации. Нельзя обозначать её full operational coverage. Основной количественный вывод — полезный front между качеством и частотой; порядок полной production promotion этот readout не меняет.
