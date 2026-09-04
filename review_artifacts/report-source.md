# AlphaTransfer ML audit — каноническая база фактов

## Дополнение 2026-09-04 — текущая каноническая позиция

Полный новый evidence layer находится в
[`quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md`](quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md)
и completed run
[`quant_macro/results-final-20260904-v2/`](quant_macro/results-final-20260904-v2/).
Он **заменяет** числовой model-selection вывод legacy-разделов ниже, но не
стирает их: они нужны как история обнаружения ошибок.

| Claim | Тип | Evidence | Статус |
|---|---|---|---|
| Headline ветки `NOW lift=10.57`, `CLOSING hit=100%` | evaluation | независимый recheck truth/boundaries | **опровергнут** |
| CBR five-corridor cross является predictive alpha | mechanism | CBR 6956-U/OD-1012; live 5/5; KZT 412/413 | **опровергнут**: contemporaneous structural compatibility; future оба cross-компонента неизвестны |
| Compact CNY spot-minus-fixing displacement (`basis` в ID) улучшает base | retrospective association | Brier −12.846%; CI delta [−.038115;−.017824]; Holm .0037; 3/3 years, 15/15 cells | **подтверждено только на development snapshot при MOEX T+1** |
| Эффект устойчив к source timing | robustness | lag 1→2: Brier .19062→.22551; candidate lift 1.435→1.065 | **опровергнут** |
| Broad macro/market stack лучше compact block | ablation | full −6.97%; all-contract −11.98% относительно CNY basis | **опровергнут** |
| Есть open-data-only ML winner | ablation | public KZ official −0.386%, 0/3 years; public-fast −6.03% | **опровергнут** |
| Candidate policy проходит quality | conditional policy audit | lift 1.435 [1.305;1.574], forward +47.69 bps [32.77;64.02] | **да, post-selected official-rate diagnostic** |
| Candidate policy проходит operations | full corridor audit | minimum weekly fulfillment .667–.833 vs gate .90 | **нет, 5/5 cadence fail** |
| Bundle воспроизводим | artifact audit | `_SUCCESS`, 5 code/34 input/35 output hashes, exact whitelist | **да, byte-level snapshot replay** |
| Исторический PIT и data rights воспроизводимы | source audit | нет immutable timestamps/vintages; MOEX/KASE agreement required | **нет, fail-closed** |
| Можно запускать client pilot | causal readiness | нет Alpha quote/economics, frozen shadow, explanations, CRM simulation | **NO-GO** |

Строгая формулировка: **GO на deterministic cross/residual engineering и
prospective data collection; NO-GO на production signal и client contact**.
`3 635` rows нельзя выдавать за независимый N: парный development анализ имеет
727 decision dates, а target correlation между коридорами в среднем 0.705.

### Новые первичные источники

- [CBR 6956-U](https://www.cbr.ru/Queries/XsltBlock/File/105012/-1/2531) и
  [Annex OD-1012](https://www.cbr.ru/Content/Document/File/186003/Annex_OD-1012_en.pdf)
  — механизм cross-rate и пять USD/LCY pairs;
- [NBK exchange-rate metadata](https://nationalbank.kz/en/news/metadannye/7252)
  — KASE basis и official-rate semantics;
- [MOEX Market Data Policy](https://www.moex.com/files/4a1jy8j83qc25vv9p286tzsmc1) и
  [KASE non-display terms](https://kase.kz/en/information/non-display) —
  договорные ограничения;
- [NBK data terms](https://nationalbank.kz/en/page/data-usage-terms) и
  [Kazakhstan BNS terms](https://stat.gov.kz/ru/description/) — условия
  повторного использования официальной статистики;
- [FRED Terms](https://fred.stlouisfed.org/legal/) — причина исключения FRED
  из ML bundle.

## Legacy evidence 2026-09-03

Дата среза: 2026-09-03. Рабочая ветка: `edelkin_test` / `origin/edelkin_test`
(`745fdc3`; в её истории также `09cc060` и `b250871`). В репозитории основная
ветка называется `main`, а не `master`.
Merge-base: `0c3f0d2f722d7c84a5b29d74014254679388f3e6`.

Этот файл — evidence layer для итогового отчёта. Здесь факты отделены от
интерпретаций и гипотез. Независимые пересчёты сохранённых событий не считаются
новым untouched-test.

## 1. Исследовательские вопросы

1. Соответствует ли реализация `edelkin_test` постановке и свежей Q&A?
2. Можно ли защищать опубликованные OOT lift/hit/bps/frequency?
3. Что остаётся перспективным после исправления truth-функций?
4. Какие данные, признаки и модели имеют наибольшую ожидаемую ценность?
5. Какой offline/online metric contract не позволит выиграть «ошибкой evaluator»?

## 2. Проверенная внутренняя доказательная база

### Ветка `edelkin_test`

- `global_task.md`, `Вопросы и ответы.md`;
- `src/alphatransfer/{ingestion,features,models,evaluation,policy,cli,reporting}.py`;
- `configs/kzt_v0.toml`, `tests/test_kzt_v0.py`;
- `data/kzt_v0/{observations,features,backtest,signal,report_snapshot}.json/csv`;
- `docs/*.md`, `reports/2026-09-01_backtest_summary.md`, `star/*`;
- общий diff относительно merge-base и содержимое review app.

### Контекст только в `origin/main`

| Commit | Материал | Значение для решения |
|---|---|---|
| `e829559` | полная свежая Q&A и сводка | уточняет бизнес-цель, truth, все `h`, cadence и deliverables |
| `269d836` | `Researches/chat_research_kz/*` | JTBD/сегменты; потенциальный запрет на использование в сдаче из-за приватного raw/PII |
| `d8c9e8e` | пяти-коридорный OXR baseline | useful negative control: ни одна из 100 комбинаций не прошла все gates |
| `87de36f` | `HACKATHON_ML_PLAN_V2.md` | актуальный decision log: сначала evaluator, потом model zoo |

### Выполненные проверки

- Python 3.11: `9/9` unit tests проходят.
- 15 793 observations: нет duplicate keys, non-positive/non-finite значений и
  нарушений нормализации; OHLC invariants не нарушены.
- `features.csv` воспроизводится из `observations.csv`.
- Полный backtest воспроизводит события/решения; отличия JSON — только floating
  epsilon до `2.22e-16`.
- Независимый fixed-event recheck: `review_artifacts/independent_recheck.py`.
- Независимый пяти-коридорный clean experiment: см.
  `review_artifacts/experiments/` после генерации.

## 3. Числовой снимок ветки

### Опубликованный V0

- 1 295 MOEX-сессий, 7 OOT folds, 58 raw/suppressed events;
- 17 policy-eligible событий, все только в двух последних folds;
- `favorable_now, h_moex_rows=5`: 7 signals, hit `0.5714`, baseline `0.0540`, lift
  `10.5739`, timing `83.79 bps`;
- `window_closing, h_moex_rows=5`: 10 signals, hit `1.0`, baseline `0.7671`, lift
  `1.3036`, timing `28.14 bps`.

Эти headline-числа **invalid**: NOW использует другую truth, CLOSING truth
тавтологична, а selection/evaluation имеют point-in-time и aggregation defects.

### Fixed-event recheck по Q&A truth

Заморожены те же даты policy events; модель не переобучалась.
`h_moex_rows` считает следующие позиции в сохранённой MOEX-aligned панели, а не
distinct CBR publications или календарные дни; CBR/NBK значения могут
carry-forward. Поэтому `443` ниже — eligible rows одного decision clock, не
443 независимых дня.

| Scenario | Truth | h_moex_rows | Hits/N | Baseline | Lift | Fisher one-sided p | `±h` mean bps | Forward mean bps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NOW | `S_t ≤ min(S_t+1…S_t+h)` | 5 | 4/7 | 0.253 | 2.260 | 0.072 | +81.96 | −0.36 |
| CLOSING | `max future rise >35 bps` | 5 | 9/10 | 0.634 | 1.419 | 0.069 | +19.57 | +144.49 |

NOW truth следует Q&A. Точная форма CLOSING в Q&A не зафиксирована, поэтому
`max-within-h > δ` — командное допущение; `δ=35` взят post hoc из trigger-config
и имеет только диагностический статус.

При строгом исключении последних `h_moex_rows` строк каждого test fold остаётся 6 и 9
сигналов: NOW `4/6`, lift `2.747`; CLOSING `8/9`, lift `1.390`. Это не усиливает
claim: выборка становится ещё меньше.

NOW `h_moex_rows=5` source sensitivity на тех же семи датах:

| Source | Hits/N | Baseline | Lift |
|---|---:|---:|---:|
| CBR | 4/7 | 0.253 | 2.260 |
| NBK | 4/7 | 0.287 | 1.993 |
| MOEX | 3/7 | 0.242 | 1.774 |

Следствие: опубликованное «NBK/MOEX lift=0» — артефакт несогласованных
симметричных labels, но corrected source robustness остаётся exploratory.

Эта таблица использует общую сетку 443 MOEX decision rows, а не native source
clocks: freshness в T не фильтруется, future CBR/NBK может быть forward-filled.
Она показывает label sensitivity, но не три независимые robustness-репликации.
Отдельный purged source-fresh срез хранится в experiment artifacts и также
остаётся диагностическим из-за MOEX-row clock.

### Реальная cadence

OOT-span: `2024-11-11…2026-08-10`, 638 дней / 92 календарные недели.

| Scenario | Signals | Signals/week | Silent-week share | Max gap с границами |
|---|---:|---:|---:|---:|
| all | 17 | 0.187 | 82.6% | 473 days |
| NOW | 7 | 0.077 | 92.4% | 473 days |
| CLOSING | 10 | 0.110 | 90.2% | 479 days |

15 из 22 месяцев пусты; первые 5 из 7 folds не дали ни одного сообщения.

### Оптимистичная нижняя граница статистической мощности

Односторонний exact-binomial test, `α=0.05`, power `80%`, baseline известен,
события IID, один frozen test, без multiplicity:

- NOW `h_moex_rows=5`, baseline `0.253`, alternative lift `1.3`: примерно
  **224** signals;
- CLOSING `h_moex_rows=5, δ=35`, baseline `0.634`, alternative lift `1.3`:
  **36** signals.

Это нижняя граница. Temporal dependence, tuning, пять horizons и пять corridors
увеличивают требование. При текущей cadence NOW один KZT-коридор собирал бы 224
события десятилетиями.

### Clean five-corridor experiment на official CBR

Независимый от branch implementation скрипт проверил 9 model/feature sets на
пяти `h_cbr_rows`, то есть 45 вариантов. Из 30 learned-вариантов 28 попали в post-hoc
cadence-band `[0.8,1.2]/week`, 14 также имели положительные абсолютные
signal-date point estimates `±h` и forward bps. Общие OOT test-years: 2023,
2024, 2025 и неполный 2026;
rolling train — два года, validation — предыдущий год, хвосты train/validation
и закрытых test-years purged на `h_cbr_rows`. Threshold использует validation scores
только для частоты, без validation labels. Cooldown перед test прогрет
контрфактическим replay предыдущего года с новой fold-policy, а не непрерывным
immutable delivery ledger. Experiment моделирует только NOW.

`data/cbr_daily.csv` не содержит настоящего `published_at`: поле `date` из CBR
XML соответствует дате действия курса. Последовательность записей использована
как proxy publication clock, поэтому эксперимент не закрывает точный PIT
timestamp/calendar sensitivity.

Единственного winner нет. Ranking-лидер `logit_full, h_cbr_rows=1` имеет
cell-standardized lift `1.110` и абсолютный forward `+14.57 bps`, но
обязательную абсолютную `±h`-выгоду `−2.07 bps` (incremental `−2.41`).
Post-hoc screened-кандидат для подробной статистики —
`logit_full, h_cbr_rows=20`: baseline `817/4115`, signals `774`, hits `156`, frequency
`0.916/week/corridor`. Абсолютная signal-date `±h`-выгода `+26.65 bps`,
forward `+35.23 bps`. Primary cell-standardized lift `1.095`, incremental
hit-rate `+1.75 pp [−1.03, +4.81]`, forward `+13.84 bps`, regret improvement
`+6.43 bps`; exposure-pooled lift `1.015`, hit-rate `+0.30 pp
[−2.48, +3.48]`, incremental forward `−3.80 bps`, regret `−4.24 bps`. Смена знака
относится к сравнению с baseline, а не абсолютной выгоде: это robustness
failure, а не повод выбрать удобный estimand. Standardized minimum yearly lift `0.8855` (3/4 ≥1),
minimum corridor lift `1.0417` (5/5 ≥1); raw-positive `12/20` cells.

Month-block interval для fixed screened-кандидата: absolute signal-date `±h`
`[+0.38, +53.70]`, forward `[−25.59, +97.36] bps`; primary lift
`[0.945, 1.266]`, incremental forward `[+0.57, +28.28] bps`; exposure lift
`[0.873, 1.178]`, forward `[−25.22, +17.44] bps`. Месячный блок короток
относительно перекрывающихся `h_cbr_rows=20` labels. Circular-shift p-value primary lift
`.139`, forward `.163`, regret `.305`; после Bonferroni-45 все равны `1.0`.
Brier `0.1957` хуже prior-year climatology `0.1667`, ECE `0.175`. Статус —
**exploratory/post-selection/no-winner**; horizon/model для нового frozen
holdout надо выбрать по продукту, а не по максимуму этой матрицы.

## 4. External primary sources

| ID | Источник | Проверенный факт | Использование |
|---|---|---|---|
| E1 | [Банк России: время публикации](https://www.cbr.ru/Reception/TopicalMessage/Page/2661) | точное время не регламентировано; обычно публикация до 18:00 МСК | нужен реальный `published_at`, а не полночь `effective_date` |
| E2 | [Банк России: база курсов](https://www.cbr.ru/currency_base/) | курс вступает в силу на следующий календарный день и действует до следующего приказа | разделить publication/effective clocks; official rate не execution quote |
| E3 | [Актуальный перечень/метод валют ЦБ, Annex OD-1012](https://www.cbr.ru/Content/Document/File/186003/Annex_OD-1012_en.pdf) | AMD/KGS/KZT/TJS/UZS рассчитываются через соответствующие USD-пары; прямые CNY/USD — отдельный контур | декомпозиция общего RUB/USD фактора и corridor residual; источники не независимы |
| E4 | [НБК: metadata exchange rates](https://nationalbank.kz/en/news/metadannye/7252) | USD/KZT основан на KASE; другие валюты — cross-rates; official публикуется после setting; rate не обязателен для сделки | NBK/CBR не являются независимыми execution labels; важен timestamp |
| E5 | [MOEX: программа market maker KZTRUB_TOM](https://fs.moex.com/files/18735/31337) | допустимый bilateral spread 1%, quote obligation 40% основной сессии, minimum quote 40m RUB, volume obligation отсутствует | OHLC без volume/spread/quote coverage нельзя считать одинаково надёжным каждый день |
| E6 | [НБК: FX operations/interventions](https://nationalbank.kz/en/news/foreign-exchange-market-interventions/rubrics/2572) | доступны помесячные интервенции, monetary-neutrality sales и операции NF | regime/context features с point-in-time release lag |
| E7 | [НБК: release calendar](https://nationalbank.kz/en/statisticalinformation/grafik-vypuska-statisticheskoy-informacii?filter=month&section_id=) | опубликован календарь статистики | корректный `available_at` для макро-признаков |
| E8 | [EIA: daily Brent](https://www.eia.gov/dnav/pet/hist/RBRTED.htm) | доступен открытый ежедневный Brent spot history | внешний KZT regime feature, ценность должна быть доказана ablation |
| E9 | [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | time split поддерживает `gap`, исключающий хвост train перед test | внешний sanity reference для boundary purge |
| E10 | [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | calibrated `p≈0.8` должен реализовываться примерно в 80% случаев; Brier смешивает reliability/resolution/uncertainty | branch score нельзя называть probability без отдельной calibration проверки |
| E11 | [Chernozhukov et al.: conformal inference with dependent data](https://arxiv.org/abs/1802.06300) | для dependent time series используются block structures, IID exchangeability неприменима напрямую | CI/null tests должны сохранять временную зависимость |

## 5. Claim ledger

| Claim | Тип | Evidence | Статус |
|---|---|---|---|
| Current CLOSING 100% hit tautological | факт | `policy.py:26-35`, `features.py:189-194`, 43/43 raw events | подтверждён |
| Current NOW 10.57× measures wrong truth | факт | `evaluation.py:58`, `features.py:181-186`, fixed-event recheck | подтверждён |
| Validation labels use first test rows | факт | `evaluation.py:103-119`; 5 rows × 7 folds | подтверждён |
| Historical `signal --as-of` uses future-selected model/lift | факт | `cli.py:41-58`, `evaluation.py:195-204`; replay 2025-01-10 | подтверждён |
| `available_at` is ignored in feature eligibility | факт | `features.py:17-19`; synthetic 2030 revision changes 2025 feature | подтверждён |
| Monday freshness is a constructed artifact | факт | 0/261 Monday rows fresh; `features.py:71-91` | подтверждён |
| Current result is deployable GO | claim ветки/UI | validity gates fail; N=7/10, zero-heavy folds | опровергнут |
| Corrected NOW/CLOSING may contain signal | inference | lifts >1 on same frozen dates, but p≈0.07 and invalid selection | promising, inconclusive |
| Five-corridor NOW grid has a confirmed winner | inference | ranking leader fails `±h`; screened h20 has std lift 1.095 vs pooled 1.015, adjusted p=1 | опровергнут; Pareto hypotheses only |
| Adding a bigger model will solve quality | inference | best simple challenger misses 1.3/calibration/stability gates; data/source instability | evidence against prioritisation |
| CBR, NBK and MOEX are three independent truths | claim ветки | E3–E5; common USD/cross construction and different semantics | overstated |
| MOEX liquidity fields may improve reliability | inference | E5 + 46.7% pre-2022 zero-range rows | high-priority test, not proven lift |
| Brent/NBK flows can improve prediction | hypothesis | E6–E8 | test only after PIT contract; no quality claim |
| Telegram chat evidence is admissible | unknown | `origin/main` says private export/from_id; case prohibits non-public/PII | do not submit until explicit approval |

## 6. Открытые пробелы

1. Нет historical Alpha execution quote, spread, fee, quote expiry и recipient
   amount — CBR оценивает market timing, а не фактическую ценность перевода.
2. Нет полностью point-in-time release timestamp/vintage для всех sources;
   five-corridor snapshot также хранит CBR effective `date`, не `published_at`.
3. Есть exploratory five-corridor NOW evaluator, но нет CLOSING grid и нет
   preregistered untouched holdout после выбора business horizon/model.
4. Нет нового untouched test после исправления дефектов.
5. Для следующего holdout ещё не выбран actionable `h_pub/h_cal`; CLOSING `δ*`,
   экономический FP-cost и online MDE требуют банковских quote/unit economics.
6. Нет подтверждения, что Telegram export публичен и разрешён для конкурсной
   доказательной базы.

## 7. Воспроизведение

```bash
# Из корня AlphaTransfer, Python 3.11+ в PATH
python3.11 -m venv .venv-review
.venv-review/bin/python -m pip install -e .
.venv-review/bin/python -m pip install \
  -r review_artifacts/experiments/requirements.txt

PYTHONPATH=src .venv-review/bin/python -m unittest discover -s tests -v

.venv-review/bin/python review_artifacts/independent_recheck.py \
  --bootstrap-replicates 5000 \
  --check-known-v0 \
  --output-dir review_artifacts/generated

.venv-review/bin/python \
  review_artifacts/experiments/clean_five_corridor_experiment.py \
  --repo-root . \
  --output-dir review_artifacts/experiments \
  --bootstrap-reps 20000 \
  --null-reps 10000
```

`independent_recheck.py` использует только stdlib, фиксированный seed и пишет
machine-readable JSON/CSV плюс краткую Markdown-сводку. Block bootstrap и
circular shifts являются диагностикой сохранённых событий, а не correction за
post-selection.
