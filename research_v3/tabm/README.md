# TabM challenger: ограниченный воспроизводимый benchmark

Этот эксперимент проверяет одну компактную конфигурацию [TabM (ICLR 2025), официальный код](https://github.com/yandex-research/tabm) на тех же frozen features и temporal folds, что текущий HGB. Его цель — измерить, помогает ли neural ensemble, а не переносить лидерство на общих табличных benchmarks на валютную задачу.

Из корня `AlphaTransfer`, в Python 3.11 окружении с `tabm torch pandas numpy scikit-learn scipy`:

```bash
python research_v3/tabm/run_tabm_benchmark.py
```

Конфигурация задана до оценки outer test: TabM с 2 блоками по 128, k=16, dropout=0.1, AdamW lr=0.002, weight_decay=0.0003, batch=256. Обучается loss каждого из k predictions, на inference усредняются вероятности. Это базовый TabM без дополнительных feature embeddings, не полный HPO архитектуры. Features: 14 core+vol либо те же 14 + один CNY spot-minus-fixing. Corridor — native categorical feature. Median imputation и quantile-to-normal fit только на train.

Внешние годы 2023/2024/2025 — development OOT, 2026 — ранее просмотренная диагностика. Train: два года; validation: предыдущий год; purge h=5. Early stopping TabM использует последние 63 уникальные даты **внутри train**, отделённые purge5 от inner train. После выбора числа epochs модель переобучается на всех train2y; outer validation остаётся для Platt/corridor thresholds и blend weight. Выбор identity/Platt и threshold точно наследуют incumbent evaluator, включая описанное в методологическом отчёте повторное использование validation.

HGB, TabM и convex blend оцениваются теми же `core.run_fold`, `metric_rows` и `aggregate_metrics`. Blend weight на TabM выбирается из 0/.25/.5/.75/1 по raw Brier предыдущего validation года, затем blend проходит тот же Platt/threshold. Все outer test labels используются только при readout. В сравнении с incumbent повторно обучается HGB в том же новом окружении; версия sklearn и numeric parity должны проверяться по результату.

Артефакты в `output/`:

- `frozen_protocol.json`: параметры, версии и input/code hashes;
- `predictions.csv`, `fold_corridor_metrics.csv`, `aggregate_metrics.csv`;
- `scorecard.csv`: компактные mandatory metrics;
- `paired_brier_intervals.csv`: year-stratified month-block interval, 10,000 repeats, exploratory;
- `training_details.json`: выбранные epochs и prior-validation blend weights;
- `artifacts/*/`: fitted weights, preprocessing и learning curves;
- `_SUCCESS.json`: длительность и hashes итоговых CSV.

Большой поиск гиперпараметров, новые данные и production-готовность этим experiment не заявляются. Те же ограничения source availability proxies и effective CBR horizon, что у incumbent, сохраняются. Brier intervals не корректируют весь исторический процесс отбора; для promotion нужен новый frozen prospective period. Результаты и фактические deltas записаны в `RESULTS.md`. Дополнительную оценку стабильности и prediction-level parity воспроизводит `python research_v3/tabm/finalize_results.py`; она не переобучает модели.
