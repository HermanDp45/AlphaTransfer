# AlphaTransfer final solution

Один entry point превращает проверенный ML-кандидат в полный product decision: проверяет входы, собирает три ключевые метрики, формирует исторический сигнал, применяет TTL/CRM/timezone/quote-гейты и пишет понятный отчёт.

## Быстрый запуск

Из корня репозитория:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 final_solution/main.py
```

Результат появится в `final_solution/output/`:

- `EXECUTIVE_SUMMARY.md` — короткий итог;
- `model_scorecard.json` — качество и гейты;
- `signal_decision.json` — сигнал, UX-copy и suppression ledger;
- `key_metrics.csv`, `source_receipt.csv` — плоские артефакты;
- `run_receipt.json`, `_SUCCESS.json` — воспроизводимость запуска.

Быстрый режим использует зафиксированный final-run и исторический demo-as-of `2025-12-16`. Он не требует сети и сторонних библиотек.

## Полная пересборка ML

Нужен Python 3.11+ с зависимостями из `final_solution/requirements-ml.txt`. Один запуск:

```bash
python3.11 final_solution/main.py --rebuild final --verify-full-bundle
```

`final` требует не менее 10 000 block-bootstrap повторов. Для проверки wiring без inferential-выводов:

```bash
python3.11 final_solution/main.py --rebuild smoke --bootstrap-reps 100
```

В `main` отсутствовал обязательный `data/cbr_daily.csv`, поэтому переносимая frozen-копия включена в `final_solution/data/` и защищена SHA-256 lock-файлом. Пересборка проходит в `final_solution/work/` и не меняет старые данные репозитория.

Произвольная историческая дата поддерживается с полным prediction-файлом:

```bash
python3 final_solution/main.py \
  --as-of 2025-12-10 \
  --predictions review_artifacts/quant_macro/results-final-20260904-v2/development_h5_predictions.csv
```

## Важно

Pipeline ничего не отправляет наружу. Текущий результат честно имеет статус `SHADOW_ONLY`: lift и proxy-выгода проходят, недельное покрытие и production-гейты — нет. Смысл подхода, определения, тайминг и путь клиента описаны в [APPROACH.md](APPROACH.md).
