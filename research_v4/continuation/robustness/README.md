# KZT Halyk: устойчивость к задержке данных

[REPORT.md](REPORT.md) содержит результаты и ограничения; [PROTOCOL.md](PROTOCOL.md) — первоначальный фиксированный набор методов; [AGE_FOLLOWUP_PROTOCOL.md](AGE_FOLLOWUP_PROTOCOL.md) — два дополнительных возрастных гейта, предложенных после просмотра основных результатов.

Ключевое уточнение: **обучение на L1 / применение к L2 даёт Brier 0.181404**, тогда как прежние **0.186250 при переобучении на L2** относятся к другому эксперименту. Новые защитные методы не улучшили устойчивость без потери качества на своевременных данных. Первоначальный minimax-выбор использует только development 2023–2025; результаты 2026 года остаются ретроспективной диагностикой.

Воспроизведение из корня AlphaTransfer, один вычислительный поток:

```bash
python research_v4/continuation/robustness/experiment.py
python research_v4/continuation/robustness/age_followup.py
python research_v4/continuation/robustness/verify.py
python research_v4/continuation/robustness/build_report.py
```

Использованный Python: `/private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python`; зависимости: numpy, pandas, scikit-learn, scipy, threadpoolctl. Полный fit занимает около 13 секунд, независимые проверки — около 15 секунд. Старые данные и алгоритмы V3/V4 только читаются; результаты записываются исключительно в эту папку. Прежний V4 manifest сохранён в `continuation/previous_v4_manifest.json`. Только две обновляемые root-страницы V4 проверяются по прежним копиям в `continuation/prior_reports/`; все остальные старые файлы сверяются на месте.

## Артефакты

- `results/headline.csv` — Brier каждого метода на своевременных и задержанных данных, максимум и величина ухудшения.
- `results/combined_development.csv`, `results/combined_common_march5.csv` — итоговые сводки, включая два ограниченных follow-up. Стандартизированные lift и forward delta совместимы с V3.
- `results/development_summary.csv` — восемь первоначальных правил; `results/robust_selection_table.csv` и `results/selection.json` — критерий и выбор метода до нового вычисления результатов 2026 года.
- `results/metrics_by_cutoff.csv` — все доступные зрелые h5 outcomes каждой даты фиксации; `results/common_march5_metrics.csv` — одинаковые 120 дат 2026-03-05—2026-08-25 для January/March freeze.
- `results/checkpoints/`, `results/model_receipts.json` — реальные модели, состояние калибровки и порогов, даты обучения, SHA256 и fingerprints признаков.
- `results/age_validation_trials.csv`, `results/age_receipts.json`, `results/age_exposure.csv` — выбор возрастных порогов 2/3 дня по прошлой validation и фактическая доля переключений.
- `results/paired_brier_intervals.csv` — парные блоки по 10/20/40 рыночных дат, условная диагностика без повторного обучения и поправки на исследовательский поиск.
- `results/old_parity.json`, `results/verification.json` — точное воспроизведение замороженных контролей, независимые численные и временные проверки.
- `artifact_manifest.json` — SHA256 всех исходников и результатов этой ветки; записывается последним `build_report.py`.

## Форматы и ограничения

`delay=0` означает обычную доступность источника; `delay=1` — дополнительный календарный день задержки после cutoff, без переобучения. Для lag1-модели это L1→L2, для отдельно обученной lag2-модели — L2→L3.

`cutoff=2026-03-01` использует явную 12-месячную историю калибровки, отдельно от 10-летнего обучения до 2025-03-01. Пороги и состояние cooldown сохранены в checkpoint. Конец h5 определяется по фактическим сессиям CBR. Исторические snapshots и предполагаемые даты доступности не заменяют настоящие publication timestamps.

История 2026 года уже изучалась в V4: отсутствие её outcomes в fit/selection не делает её нетронутым holdout. Банковский P&L, исполнение переводов и причинный эффект на клиента здесь не измеряются.
