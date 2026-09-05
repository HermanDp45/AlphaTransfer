# AlphaTransfer V3 — исследование и работающий decision preview

Итог: [REPORT.md](REPORT.md). Полный численный ledger: [COMPARISON.csv](COMPARISON.csv), краткая матрица: [FINAL_COMPARISON.md](FINAL_COMPARISON.md). Исследование выполнено 4–5 сентября 2026; все исторические результаты exploratory.

## Быстрый запуск без внешних библиотек и сети

Из каталога `AlphaTransfer`:

```bash
python3 final_solution/main.py --research-v3 \
  --model basis_train_120m --policy selective --threshold 0.50 \
  --as-of 2025-12-16
```

Результат: `research_v3/preview_output/decision.json`. `selected=null` означает честное отсутствие сигнала при данном пороге. Другие варианты: `--model baseline_reproduction`, `--policy legacy`, `--horizon 1/3/5/10/20`, `--corridors KZT,TJS`, `--client-context research_v3/examples/behavior.synthetic.json`. API `research_v3.preview.decision(date, ...)` использует только доступный префикс прогнозов и проверяет SHA-256 receipt. Поддерживаются сохранённые OOT даты; текущий live-курс не выдумывается.

Старый режим `python3 final_solution/main.py` сохраняет точные baseline-метрики, но использует исправленные factual-copy и optional behavior gate. Реальных пушей оба режима не отправляют.

## Пересборка из frozen snapshots

Модели: Python 3.11+ и зависимости старого `final_solution/requirements-ml.txt`. Проверенная среда основного прогона: Python 3.14.7, numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0. Вызовы ниже выполняются из `AlphaTransfer`:

```bash
python research_v3/models/experiment.py --stage annual
python research_v3/models/experiment.py --stage dynamic
python research_v3/models/assess.py
python research_v3/models/decision_uncertainty.py
python research_v3/build_comparison.py
python -m unittest discover -s final_solution/tests -v
python -m unittest discover -s research_v3/tests -v
```

Для остальных горизонтов: `python research_v3/models/experiment.py --horizon 10 --only baseline_reproduction,basis_train_120m,annual_recent_calibration_3m,long_with_historical_cny` (аналогично 1/3/20). Per-run receipts проверяют code/input fingerprint, спецификацию и hash прогнозов перед reuse.

Отдельные воспроизводимые ветки:

- [external_data/REPORT.md](external_data/REPORT.md): FRED/CBOE/ICE counterfactual, Treasury replica, GPR, длинные комбинации; импортируемый `benchmark.py`. Сырые restricted inputs не являются автоматически разрешённым публичным distribution bundle.
- [tabm/README.md](tabm/README.md): TabM, torch 2.14.0, tabm 0.0.3, отдельная проверенная Python 3.11 среда; checkpoints сохранены.
- [behavior/README.md](behavior/README.md): генератор, 12 сценариев, 3 seeds, проверки и таблицы. Все эффекты явно simulation-only.
- [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md): первичные научные источники и аудит постановки.
- `models/fetch_history.py`: дополнительная открытая история ЦБ 2010–2019; сеть нужна только для обновления snapshot.

`python research_v3/verify.py` проверяет центральный manifest, численную идентичность baseline и полноту обязательных артефактов. Для графиков: `MPLCONFIGDIR=/tmp/alphatransfer-mpl python research_v3/build_figures.py` в окружении с matplotlib.

## Что считать результатом

Главное улучшение — разделение forecast quality, policy utility и пользовательского контакта. Нет оснований объявлять новую архитектуру победителем сразу по всем метрикам или превращать 2026 в свежий holdout. Для prospective проверки фиксируются две политики в [selection.json](selection.json); это исследовательский контракт, не активный мониторинг и не отправка клиентам.
