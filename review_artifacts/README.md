# AlphaTransfer ML review — навигация

> **Структура обновлена 2026-09-04.** Исполняемый training/evaluation-контур,
> frozen-данные и канонический bundle перенесены в [`../final_solution/`](../final_solution/).
> Эта папка теперь хранит отчёты и исторический audit trail, а не рабочий код.

## Канонический пакет 2026-09-04

Итоговый verdict: **NO-GO на production/client pilot; GO на deterministic
cross-rate engine и frozen prospective shadow после закрытия P0 data
contracts**. Старые headline `lift=10.57` / `100% hit` невалидны. Новый
`hgb + CNY spot-minus-fixing displacement` (`basis` в feature ID) — только
contract-required retrospective hypothesis:
Brier лучше base на 12.846%, но один дополнительный день source lag уничтожает
эффект, все пять коридоров проваливают cadence, а PIT/rights/explanation gates
не закрыты.

Читать в таком порядке:

1. [`EXECUTIVE_DECISION_2026-09-04.md`](EXECUTIVE_DECISION_2026-09-04.md) —
   управленческий вывод, ошибки ветки, целевая архитектура и приоритеты.
2. [`quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md`](quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md) —
   точные 10k-bootstrap результаты, ablation, timing и cadence frontier.
3. [`quant_macro/SOURCE_CATALOG_2026-09-04.md`](quant_macro/SOURCE_CATALOG_2026-09-04.md) —
   31 source family, 76 artifacts, clocks, coverage и data rights.
4. [`quant_macro/MODEL_EXPERIMENT_ROADMAP.md`](quant_macro/MODEL_EXPERIMENT_ROADMAP.md) —
   deterministic/residual architecture, модели и новые experiments.
5. [`METRIC_CONTRACT_V2.md`](METRIC_CONTRACT_V2.md) и
   [`PILOT_METRIC_CONTRACT.md`](PILOT_METRIC_CONTRACT.md) — offline и causal
   online decision contracts.
6. [`quant_macro/DISTRIBUTION_AND_LICENSE_README.md`](quant_macro/DISTRIBUTION_AND_LICENSE_README.md) —
   что нельзя передавать наружу до legal/data-owner sign-off.
7. [`DEFENSE_STORYBOARD_7MIN.md`](DEFENSE_STORYBOARD_7MIN.md) — готовый
   7-минутный сюжет защиты и ответы на неудобные вопросы жюри.

Отдельный access-check MOEX WAP0:
[`quant_macro/moex_wap0_access_probe_2026-09-04.json`](quant_macro/moex_wap0_access_probe_2026-09-04.json) —
700 строк metadata/history, но ни одного доступного `WAPRICE` в бесплатном ISS.

Канонический machine-readable run:
[`../final_solution/model_bundle/`](../final_solution/model_bundle/).
Он имеет `_SUCCESS=complete`, `run_tier=final`, 10 000 bootstrap iterations,
совпавшие 5 code / 34 input / 35 output hashes и пустые prospective/production
tracks. `results-final-20260904-v1/` прерван и запрещён к использованию.
Fail-closed проверка bundle:
[`../final_solution/training/verify_bundle.py`](../final_solution/training/verify_bundle.py).

Быстрая проверка нового контура:

```bash
PYTHONDONTWRITEBYTECODE=1 python3.11 -m unittest discover \
  -s final_solution/tests -v

.venv/bin/python final_solution/training/verify_bundle.py

.venv/bin/python final_solution/main.py \
  --rebuild final --bootstrap-reps 10000 --verify-full-bundle
```

## Legacy snapshot 2026-09-03 — сохранён для audit trail

Дата среза: 2026-09-03. Проверены `origin/edelkin_test@745fdc3` и дополнительный
контекст `origin/main@87de36f`.

## Короткий вердикт

Текущие headline `lift=10.57` для NOW и `100% hit` для CLOSING невалидны из-за
ошибок truth/evaluation и не должны попасть в защиту. После исправления target
сохранённые события выглядят интересно, но их слишком мало (`N=7/10`).
Независимый five-corridor NOW experiment не нашёл подтверждённого winner.
Post-hoc кандидат `logit_full, h_cbr_rows=20` даёт cell-standardized lift `1.095`, но
exposure-pooled lift лишь `1.015`, а forward delta меняет знак (`+13.84` против
`−3.80 bps`). Это incremental delta к random-day baseline; абсолютный
signal-date forward равен `+35.23 bps`. Кандидат не прошёл lift, uncertainty, multiple-testing, calibration и
stability gates. Правильный статус: **INVALID EVALUATION → REBUILD → NEW
UNTOUCHED TEST**.

## Что читать

1. [`ML_REVIEW_2026-09-03.md`](ML_REVIEW_2026-09-03.md) — основной отчёт:
   аудит реализации, пересчёты, данные/признаки/модели и план на 72 часа.
2. [`METRIC_CONTRACT_V2.md`](METRIC_CONTRACT_V2.md) — формальные truth,
   clocks, denominators, offline/online метрики и hard gates.
3. [`experiments/README.md`](experiments/README.md) — независимый clean
   five-corridor experiment, итоговая таблица и ограничения.
4. [`report-source.md`](report-source.md) — канонический evidence/claim ledger
   с внутренними и внешними источниками.

## Машинно-воспроизводимые артефакты

- [`../final_solution/research/legacy_evaluator_audit.py`](../final_solution/research/legacy_evaluator_audit.py) — stdlib-аудит сохранённых
  KZT-событий без переобучения модели.
- [`generated/audit_summary.json`](generated/audit_summary.json) — полный
  machine-readable результат fixed-event audit.
- [`generated/corrected_event_metrics.csv`](generated/corrected_event_metrics.csv)
  и [`generated/delta_sensitivity_metrics.csv`](generated/delta_sensitivity_metrics.csv)
  — метрики по horizons и sensitivity к `δ`.
- [`../final_solution/training/core_experiment.py`](../final_solution/training/core_experiment.py)
  — offline-модели, признаки и временные сплиты на пяти CBR-коридорах.
- [`experiments/manifest.json`](experiments/manifest.json) — seed, протокол,
  версии и SHA-256 входов.
- [`experiments/aggregate_metrics.csv`](experiments/aggregate_metrics.csv),
  [`experiments/fold_corridor_metrics.csv`](experiments/fold_corridor_metrics.csv),
  [`experiments/statistical_tests.json`](experiments/statistical_tests.json) —
  полный model grid, granular OOT и статистические проверки.

`generated-smoke/` — контрольный повтор full fixed-event run; его файлы должны
быть byte-identical соответствующим файлам в `generated/`.

## Статус доказательств

| Результат | Допустимая формулировка |
|---|---|
| Ошибки текущего evaluator | Подтверждённый blocker, воспроизводится кодом |
| Corrected KZT fixed-event lift | Диагностика/гипотеза; не новый OOT |
| Five-corridor NOW Pareto-front | Exploratory post-selection; winner нет, бизнес-horizon ещё не заморожен |
| Online volume/revenue uplift | Не измерен; нужен randomized ITT pilot |

## Быстрая проверка

Из корня репозитория, Python 3.11+ в `PATH`:

```bash
python3.11 -m venv .venv-review
.venv-review/bin/python -m pip install -e .
.venv-review/bin/python -m pip install \
  -r review_artifacts/experiments/requirements.txt

PYTHONPATH=src .venv-review/bin/python -m unittest discover -s tests -v

.venv-review/bin/python final_solution/research/legacy_evaluator_audit.py \
  --bootstrap-replicates 5000 \
  --check-known-v0 \
  --output-dir review_artifacts/generated

.venv-review/bin/python -m py_compile \
  final_solution/research/legacy_evaluator_audit.py \
  final_solution/training/core_experiment.py
```

Полный five-corridor rerun и pinned dependencies описаны в
[`experiments/README.md`](experiments/README.md). Все экспериментальные расчёты
используют только tracked public snapshots; внешний контекст в отчёте ссылается
на открытые первичные источники. Рабочая реализация после ревью собрана в
`final_solution/`.
