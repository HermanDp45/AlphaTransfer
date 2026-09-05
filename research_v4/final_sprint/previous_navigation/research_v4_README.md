> Новое продолжение: [OXR2010–2026, длинные модели и банковский PoC](oxr2010_bank/REPORT.md). Предыдущие результаты сохранены.

> **Продолжение V4:** OXR 2018–2026, десять лет обучения NOW поверх Chronos, бюджет 300/900 шагов и стресс-тест задержки источников. [Новый отчёт](continuation/REPORT.md), [новая таблица сравнений](continuation/COMPARISON.csv). Ниже — исходный основной этап V4.

# AlphaTransfer V4

[Основной отчёт](REPORT.md) объясняет все пять направлений и изменения метрик. [COMPARISON.csv](COMPARISON.csv) содержит общий FX ledger; [segments/results/headline.csv](segments/results/headline.csv) — клиентские сценарные метрики с другим знаменателем.

## Направления

- [Foundation models](foundation/REPORT.md): Chronos-2 Small/Synth, zero-shot, full-weight fine-tune, KZT stage2, random-walk controls, сохранённые веса.
- [Казахстан](kazakhstan/REPORT.md): локальная калибровка, KZT-only, residual adaptation, shrinkage и новые источники.
- [Котировки и ликвидность](liquidity/README.md): реальные KASE/Halyk данные, календарь доступности, исходные ответы, 28 ablations/combinations.
- [Крипто](crypto/REPORT.md): реальные USDT/RUB и USDT/KZT, независимый прирост, доступность источника и проверки OHLCV.
- [Сегменты](segments/REPORT.md): реальные WB агрегаты, прошлый выбор политики, фиксированные веса, frequency/quality tradeoff. [Продуктовые выводы](../product_artifacts/V4_SEGMENTATION_EVIDENCE.md).

## Локальное воспроизведение

Для market/KZT веток: Python 3.11 и [requirements-market.txt](requirements-market.txt). Для foundation — отдельное окружение и [его requirements](foundation/requirements.txt). Для графиков дополнительно matplotlib 3.11.1. Все команды ниже запускаются из корня `AlphaTransfer`.

```sh
python research_v4/liquidity/experiment.py
python research_v4/liquidity/run_combo.py
python research_v4/kazakhstan/experiment.py
python research_v4/kazakhstan/run_context.py
python research_v4/analyze_market.py
python research_v4/build_report.py
python research_v4/build_figures.py
python research_v4/verify_market.py
```

Загрузка источников отделена от обучения; команды есть в README каждой ветки. Нельзя повторно скачивать свежие revised snapshots и считать их теми же frozen inputs: проверяйте сохранённые hashes. WB raw microdata не включены в поставку; включены только агрегаты. Для самостоятельного повторения агрегации используйте официальный access flow, описанный в segments/README.

## Исторический preview

```sh
python research_v4/market_preview.py --model kzt_residual_shrink_120m__halyk_lag1 --as-of 2025-12-11 --corridor KZT
python -m research_v4.segments.preview --context research_v4/segments/examples/context.json --scores research_v4/segments/examples/scores.json --as-of 2026-08-04
```

Market preview проверяет receipt и не читает будущие labels/payoff. Сегментные thresholds обучались на V3: они не переносятся автоматически на новые вероятности. Оба режима локальные и исследовательские.

## Проверки и границы

`market_verification.json` проверяет 24 KZT-модели на всех 883 OOT строках каждая, источники, checkpoints, будущие perturbations и неизменность sealed V3. У foundation и crypto собственные verify scripts; у сегментов семь функциональных/temporal тестов. `liquidity/source_audit.json` сопоставляет 1 538 дневных USD/KZT значений с отдельным XLS-экспортом KASE: все совпали. `liquidity/kase_session_resolution.csv` сохраняет решения по 224 дублированным date×instrument ячейкам.

2023–2025 — уже исследованная development OOT история, 2026 — уже просмотренная диагностика. Brier, качество сигналов и сценарная клиентская ценность не взаимозаменяемы. Выбор новых кандидатов не означает их production promotion; frozen V3 и final_solution сохранены.
