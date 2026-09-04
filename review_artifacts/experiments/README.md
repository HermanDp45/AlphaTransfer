# Независимый clean experiment: 5 валютных коридоров

> **Legacy / superseded (2026-09-04).** Этот каталог сохраняется как первый
> независимый аудит evaluator и исторический evidence trail. Его model grid,
> threshold rule, calibration и post-selection статистика **не являются
> текущим каноническим экспериментом**. Для решений используйте
> [`../quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md`](../quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md),
> [`../quant_macro/quant_feature_ablation.py`](../quant_macro/quant_feature_ablation.py)
> и завершённый `results-final-20260904-v2/`. Числа ниже нельзя смешивать с
> новым h=5 protocol или выдавать за production-ready результат.

Статус результатов: **exploratory / post-selection**. Это воспроизводимый
аудит на tracked-снимках данных, а не подтверждение готовности модели к
production или обещание lift. Скрипт не делает сетевых запросов и не импортирует
исследуемый код проекта.

## Что проверено

- Источник основного panel: `data/cbr_daily.csv`, 8 205 строк (по 1 641
  официальной записи для AMD, KGS, KZT, TJS и UZS), 2020-01-01--2026-09-01.
  В snapshot есть только `date`, без фактического `published_at`; CBR XML date
  фактически отражает дату действия курса. Поэтому порядок записей используется
  как proxy `publication session`, но эксперимент не доказывает точный
  point-in-time decision timestamp.
- Цель для каждого горизонта `h_cbr_rows in {1, 3, 5, 10, 20}`: следующих
  CBR effective-date rows, используемых как publication proxy:
  `rate[T] <= min(rate[T+1:T+h])`. Равенство считается успехом.
- Primary denominator: все даты соответствующего OOT-фолда, у которых известна
  полная forward-метка. Это явно сохранено как `eligible_count`; число успехов
  denominator -- `baseline_hits`.
- OOT-фолды: календарные 2023, 2024, 2025 и неполный 2026 год. Для каждого
  фолда validation -- предыдущий календарный год, train -- два календарных года
  перед validation. Последние `h_cbr_rows` наблюдений каждого коридора удаляются из
  train, validation и закрытых test-years, чтобы forward-окно метки не
  пересекало границу. В неполном 2026 остаются все строки с уже полностью
  разрешившимся target.
- Порог модели выбирается **без validation labels**: среди 301 квантиля
  validation-score выбирается порог, дающий среднюю частоту ближе всего к
  1 сигналу/неделю/коридор после cooldown в 3 публикационные сессии. При
  равенстве выбирается более строгий порог. Перед test cooldown-state
  прогревается replay всей истории предыдущего года с замороженными для fold
  моделью и порогом; это backtest warm-up, не замена production delivery-ledger.
- Baselines: первая публикация каждой недели (`weekly_first`), три
  последовательных снижения (`down3`), низкий `pr120`. Challengers: L2-logistic
  с пятью ablation-наборами и shallow HistGradientBoosting.
- Экономические метрики: forward advantage относительно среднего следующих
  `h_cbr_rows` значений, regret относительно лучшего значения в окне (bps). Симметричная
  метрика сохранена лишь для сопоставления со старым исследованием: она включает
  прошлое и не является честной forward-метрикой.

Полный контракт, версии библиотек, seed и SHA-256 входов находятся в
`manifest.json`. Точные counts, пороги, интервалы Wilson и метрики для каждого
из 4 x 5 fold/corridor cells -- в `fold_corridor_metrics.csv`.

## Главный результат -- честно без «победителя»

Просмотрены все 45 `configuration × horizon`. Из 30 learned-вариантов 28
попали в post-hoc frequency-band `[0.8,1.2]/week`, а 14 дополнительно имели
положительные абсолютные point estimates и по обязательной для кейса
signal-date `±h`-выгоде, и по forward advantage. Это **экран по уже открытым OOT outcomes**, не заранее
заданный выбор, поэтому ни один вариант не называется winner.

Лидер по основной ranking-метрике -- `logit_full, h_cbr_rows=1`: cell-standardized lift
`1.110`, абсолютный signal-date forward `+14.57 bps`, но обязательная абсолютная
`±h`-выгода отрицательна (`−2.07 bps`; incremental vs matched baseline
`−2.41 bps`). Post-hoc экономически отфильтрованный кандидат, для которого
ниже приведены интервалы, -- `logit_full, h_cbr_rows=20`. Он показывает ещё одну важную
неустойчивость: вывод меняется при способе агрегации.

| Показатель для `logit_full, h_cbr_rows=20` | Cell-standardized primary | Exposure-pooled sensitivity |
|---|---:|---:|
| Baseline hit-rate | 0.1840 | 0.1985 (`817/4115`) |
| Signals / hits; signal hit-rate | 774 / 156; 0.2016 | то же |
| Lift | **1.0951** | **1.0152** |
| Hit-rate delta | +1.75 pp `[-1.03, +4.81]` | +0.30 pp `[-2.48, +3.48]` |
| Absolute signal-date `±h` advantage | +26.65 bps | то же |
| Incremental `±h` vs random-day baseline | +16.46 bps | +16.02 bps |
| Absolute signal-date forward advantage | +35.23 bps | то же |
| Incremental forward vs random-day baseline | **+13.84 bps** | **−3.80 bps** |
| Incremental regret improvement | +6.43 bps | −4.24 bps |

Cell-standardization сравнивает результат сигнала с baseline именно тех
`fold × corridor` cells, куда модель распределила сигналы, и потому является
primary. Exposure-pooled вариант отвечает на вопрос о грубом общем потоке.
Разворот знака относится к **incremental сравнению с random-day baseline**, а
не к абсолютной выгоде сигнальных дат: она остаётся `+35.23 bps`. Это не повод
выбрать удобную оценку, а **robustness failure / Simpson-mix warning**. Он
блокирует headline и требует заранее закрепить estimand и business weighting.

Остальные diagnostics кандидата:

| Показатель | Результат |
|---|---:|
| Средняя частота | 0.916 сигнала/неделю/коридор |
| Годовые standardized lifts >= 1 | 3/4; minimum 0.8855 |
| Коридорные standardized lifts >= 1 | 5/5; minimum 1.0417 |
| Fold × corridor cells с raw lift >= 1 | 12/20 |
| Brier / prior-year prevalence Brier | 0.1957 / 0.1667 |
| Brier skill vs prior-year prevalence | −17.4% |
| Brier skill vs same-fold oracle climatology | −26.6% |
| ECE, 10 equal-count bins | 0.1754 |

Годовая устойчивость по primary standardization и реально применённые пороги:

| Test year | Eligible / baseline hits | Signals / hits | Std. lift | Threshold |
|---|---:|---:|---:|---:|
| 2023 | 1 135 / 303 | 122 / 45 | 1.3627 | 0.00010 |
| 2024 | 1 140 / 232 | 241 / 50 | 1.0139 | 0.07187 |
| 2025 | 1 135 / 109 | 272 / 23 | 0.8855 | 0.11774 |
| 2026 (partial) | 705 / 173 | 139 / 38 | 1.1130 | 0.10572 |

Drift порога, провал 2025 и отрицательный Brier skill означают, что score пока
можно рассматривать только как ranking, не как калиброванную вероятность.

### Неопределённость и multiple testing

Month-cluster bootstrap (20 000 повторов; коридоры внутри месяца остаются
вместе) для post-hoc кандидата дал absolute signal-date 95% intervals:
`±h` `[+0.38, +53.70]` bps и forward `[−25.59, +97.36]` bps. Интервал первой
метрики едва выше нуля, но он не скорректирован за post-selection/multiplicity.
Для incremental primary получены: lift
`[0.945, 1.266]`, `±h` delta `[+6.57, +27.82]` bps, forward delta
`[+0.57, +28.28]` bps, regret improvement `[−7.30, +21.00]` bps. Для
exposure-pooled sensitivity: lift `[0.873, 1.178]`, forward
`[−25.22, +17.44]` bps, regret `[−29.11, +19.34]` bps. Месячный блок короток
для перекрывающихся `h_cbr_rows=20` labels, поэтому эти CI только диагностические;
следующий анализ должен проверить moving/stationary blocks длиной 2–3 месяца.

Circular-shift null (10 000 повторов, временной рисунок сигналов сохраняется
внутри годового фолда) дал односторонние p-value: lift `0.139`, forward delta
`0.163`, regret improvement `0.305`, `±h` delta `0.0037`. После простой
Bonferroni-поправки на 45 просмотренных вариантов: `1.0 / 1.0 / 1.0 / 0.166`.

Bootstrap/null выполнены **после post-hoc screening** и считают конфигурацию
фиксированной. Поэтому даже Bonferroni здесь -- только alarm, не полноценная
поправка за адаптивный отбор и не замена новому untouched holdout или nested
walk-forward. Ни ranking-лидер, ни screened-кандидат не прошли gates.

### Ablation на h_cbr_rows=20

| Feature set / model | Std. lift | Std. Δforward, bps | Pooled Δforward, bps | Frequency/week |
|---|---:|---:|---:|---:|
| logistic core10 | 0.931 | −7.19 | −19.36 | 0.797 |
| logistic core + levels | 1.107 | +22.46 | +3.85 | 0.738 |
| logistic core + volatility | 1.040 | +15.44 | +15.86 | 1.015 |
| logistic core + volatility + calendar | 0.957 | +8.18 | +6.66 | 1.118 |
| logistic full | 1.095 | +13.84 | −3.80 | 0.916 |
| shallow HGB core + volatility | 1.020 | +2.08 | +8.12 | 1.011 |

Levels-вариант имеет лучший lift/forward, но не дотягивает до нижней границы
cadence; volatility-вариант экономически согласованнее, но lift почти единица.
Это Pareto-набор гипотез, а не доказательство независимого вклада feature group.

У `logit_full, h_cbr_rows=20` наиболее устойчивы знаки `ret10` (−), `pr20` (+),
`dmin120` (+), `ret20` (−) и `ret3` (−): по 4/4 folds. Наибольший по модулю
средний коэффициент `ret60=−0.531` имеет SD `0.394` и совпадение знака лишь
3/4. Все standardized coefficients по фолдам находятся в
`feature_coefficients.csv`; причинно интерпретировать их нельзя из-за
коррелированных rolling-признаков.

## Дополнительные проверки текущего KZT V0 и источников

`corrected_saved_kzt_metrics.csv` пересчитывает сохранённые OOT events. Только
NOW использует точный Q&A label; CLOSING в этом файле — отдельная endpoint
sensitivity `rate[t+h]>rate[t]`, не contract primary `max-within-h>δ`. Clock —
строки MOEX panel, test-tail purged внутри каждого fold. Freshness проверяется в
T, но будущие CBR/NBK значения могут быть forward-filled, поэтому это
диагностика источника, а не сопоставимые native publication-session estimands.
Например, для сценария favorable-now, `h_moex_rows=5`, после purge остаётся всего 6
событий: CBR `4/6`, lift `2.50` и forward advantage `+18.35 bps`; NBK `4/6`,
lift `2.43` и `+4.64 bps`; MOEX `3/6`, lift `2.15` и `+41.95 bps`.
Малый N (Wilson 95% для 4/6: `[0.300, 0.903]`) и source sensitivity не
позволяют заявлять победу. Сохраненный policy дает
17 событий за 638 дней (`0.187/неделю`), 5/7 фолдов вообще без кандидатов,
82.6% недель без сигнала.

`source_profile.csv` показывает: при level-correlation CBR/OXR 0.984--0.996
корреляция однодневных log-return всего 0.108--0.292, а 95-й процентиль
абсолютного basis 324--426 bps. Exact Jaccard сигналов down3 -- 0.045--0.119.
`source_down3_crosscheck.csv` демонстрирует чувствительность к trigger/truth
source на одних и тех же датах. До дальнейшего моделирования приоритетнее
зафиксировать point-in-time источник, publication timestamp, календарь и
реально исполнимый quote/spread.

## Файлы

- `aggregate_metrics.csv` -- standardized primary и exposure-pooled sensitivity
  для всех 45 вариантов.
- `fold_corridor_metrics.csv` -- 900 строк: каждый вариант x год x коридор;
  counts, denominator, thresholds, Wilson CI, Brier и экономические метрики.
- `feature_coefficients.csv` -- коэффициенты logistic по фолдам, mean/SD и
  sign agreement.
- `calibration_bins.csv` -- reliability table post-hoc statistical focus.
- `statistical_tests.json` -- bootstrap, circular-null, Bonferroni, ECE/Brier,
  tolerance sensitivity и частота сохраненного KZT policy.
- `corrected_saved_kzt_metrics.csv` -- purged пересчёт сохранённых событий по
  CBR/NBK/MOEX: Q&A NOW и отдельная CLOSING endpoint sensitivity на MOEX-row clock.
- `source_profile.csv`, `source_down3_crosscheck.csv` -- риск смены источника.
- `manifest.json` -- data hashes и полный машинно-читаемый протокол.
- `clean_five_corridor_experiment.py` -- офлайн-перезапуск всего набора.

## Воспроизведение

Из корня репозитория:

```bash
python3.11 -m venv .venv-review
.venv-review/bin/python -m pip install -r review_artifacts/experiments/requirements.txt
.venv-review/bin/python review_artifacts/experiments/clean_five_corridor_experiment.py \
  --repo-root . \
  --output-dir review_artifacts/experiments \
  --bootstrap-reps 20000 \
  --null-reps 10000
```

После установки зависимостей сам эксперимент работает только на tracked-файлах
и не требует сети. Быстрый smoke-test можно сделать с `--bootstrap-reps 200
--null-reps 200` и отдельным output-dir; статистические p-value такого smoke-run
не использовать.

## Ограничения и следующий честный тест

1. 2026 -- неполный год. Горизонты -- последовательные записи CBR, трактуемые
   как proxy публикационных сессий, а не календарные или банковские дни; в
   snapshot нет настоящего `published_at`. Их нельзя молча сравнивать с
   OXR/MOEX или выдавать за точный intraday decision clock.
2. Forward labels перекрываются; Wilson CI в CSV диагностические. Month-block
   bootstrap частично учитывает зависимость, но месячный блок короток для
   `h_cbr_rows=20`, всего около 44 месяцев и один макрорежим -- слабая база для inference.
3. Нет executable bid/ask, комиссии, лимита, праздников назначения, времени
   cut-off и факта доступности курса клиенту. Forward bps -- proxy, не P&L.
4. Текущие пять рядов CBR движутся общими макрофакторами; 4 x 5 cells не равны
   20 независимым экспериментам.
5. Весь model/feature/horizon Pareto-front выбран по тем же OOT-годам. Сначала
   бизнес должен выбрать actionable horizon и primary estimand, затем команда
   фиксирует один challenger до новой временной выборки либо запускает внешний
   nested walk-forward. Primary endpoint заранее: forward/regret delta с
   достаточно длинным block-CI; lift, `±h` и signal rate -- обязательные
   secondary/gates; также нужны coverage, worst-corridor/year и executable
   baseline. Текущий скрипт моделирует только NOW, не CLOSING.
6. Cooldown перед каждым test-year прогревается контрфактическим replay
   предыдущего года с новой fold-моделью. Это устраняет механический reset, но
   не воспроизводит реальный непрерывный CRM ledger: даты отдельных сигналов
   могут измениться. Confirmatory evaluator должен переносить immutable
   delivered/suppressed state между folds, включая purged tail.
7. Самый вероятный прирост сейчас -- не более сложная модель, а point-in-time
   executable quotes и признаки календаря платежа/ликвидности: спред,
   cut-off/праздники обеих стран, лаг публикации, carry/ставки, нефть и RUB
   volatility/regime. После этого: calibrated logistic/GAM как baseline,
   LightGBM/CatBoost с монотонными/низкоразмерными взаимодействиями как
   challenger; deep sequence-модели только после кратного роста истории.
