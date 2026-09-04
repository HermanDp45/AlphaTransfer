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

Весь исполняемый контур находится внутри `final_solution`: обучение и оценка — в `training/`, сбор данных — в `data_pipeline/`, frozen-входы — в `data/`, эталонный результат — в `model_bundle/`. Пересборка пишет только в `final_solution/work/`; код и данные во время запуска не копируются.

Произвольная историческая дата поддерживается с полным prediction-файлом:

```bash
python3 final_solution/main.py \
  --as-of 2025-12-10 \
  --predictions final_solution/model_bundle/development_h5_predictions.csv
```

Полный prediction-файл велик и может отсутствовать в облегчённой копии репозитория. В этом случае сначала выполните `--rebuild final` или используйте включённый demo-as-of.

## Структура

- `main.py`, `alphatransfer_final/` — единый продуктовый entry point;
- `training/train_and_evaluate.py` — полный walk-forward training/evaluation;
- `training/core_experiment.py` — модели, признаки и временные сплиты;
- `training/verify_bundle.py` — строгая проверка provenance и артефактов;
- `data_pipeline/fetch_open_data.py` — воспроизводимый сбор открытых данных;
- `data/` — frozen raw/normalized snapshots и manifest;
- `model_bundle/` — зафиксированный эталонный ML-run.
- `research/` — дополнительные cadence/mechanism и legacy-аудиты, не входящие
  в основной production-like путь.

## Важно

Pipeline ничего не отправляет наружу. Текущий результат честно имеет статус `SHADOW_ONLY`: lift и proxy-выгода проходят, недельное покрытие и production-гейты — нет. Смысл подхода, определения, тайминг и путь клиента описаны в [APPROACH.md](APPROACH.md).
