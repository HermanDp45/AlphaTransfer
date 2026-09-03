# Pre-production ML-контур AlphaTransfer

Эта папка отвечает на практический вопрос хакатона: **что построить и что доказать до любого пилота на клиентах**.

Главный документ — [HACKATHON_ML_PLAN.md](HACKATHON_ML_PLAN.md). В нём зафиксированы контракт таргета, backlog, критерии приёмки модели и MVP, а также выводы первого baseline-прогона.

## Что уже реализовано

- аудит пяти целевых рядов;
- четыре прозрачных индикатора: momentum, level, reversal и expanding seasonality;
- матрица метрик по коридорам и горизонтам `1/3/5/10/20`;
- out-of-time разрез по годам;
- random-day baseline на том же коридоре и периоде;
- cooldown, частота, кучность и разброс интервалов;
- симметричная выгода `±h` и forward-only robustness;
- month-block bootstrap-интервал выгоды;
- таблица сигналов в контракте кейса;
- unit-тест prefix-invariance для защиты от lookahead.

Это диагностический baseline, а не финальная ML-модель.

## Воспроизведение

Из директории `AlphaTransfer`:

```bash
python3 scripts/hackathon_baseline.py
python3 -m unittest -v tests/test_hackathon_baseline.py
```

Срез на произвольную дату:

```bash
python3 scripts/hackathon_baseline.py \
  --as-of 2025-12-31 \
  --output-dir Researches/ml_hackathon/results_2025_12_31
```

В скрипте нет сторонних зависимостей.

## Результаты

- `results/data_audit.json` — полнота, дубли, календарные разрывы, неизменившиеся дни;
- `results/baseline_metrics.csv` — полная матрица, включая out-of-time годы;
- `results/baseline_verdicts.csv` — overall-срез и первичный вердикт;
- `results/signals.csv` — требуемая таблица `дата × коридор × индикатор × направление × сила × скорость × сценарий`;
- `results/manifest.json` — параметры запуска и определения индикаторов.

`results_*` со срезами нужны только для разовой проверки и не должны заменять основной `results/`.
