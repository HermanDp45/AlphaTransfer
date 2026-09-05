# P2P RUB/KZT rate extraction

Воспроизводимый pipeline извлекает исторические котировки RUB↔KZT из трёх Telegram JSON и разделяет:

- `card_transfer` — переводы по картам/банкам без наличных и крипты;
- `cash` — наличная нога или физический обмен;
- `crypto` — участие криптовалюты;
- `unspecified` — способ расчёта не установлен (только audit-слой).

Все курсы нормализованы как `KZT за 1 RUB`. Направления `rub_to_kzt` и `kzt_to_rub` не смешиваются. Значение дня — медиана после ограничения до одного значения на участника в каждой дневной ячейке.

Неподтверждённый день имеет пустой `observed_rate_kzt_per_rub`. Последний известный курс находится в `effective_rate_kzt_per_rub`; его происхождение видно по `fill_method`, `days_since_observed`, `is_stale_7d` и `is_stale_30d`.

## Запуск

Из корня `ai product hack`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 AlphaTransfer/Researches/p2p_rub_kzt_rates/extract_p2p_rates.py \
  --input Chats/result0.json Chats/result1.json Chats/result2.json \
  --official-rates AlphaTransfer/data/open_exchange_rates/rub_cis_daily.csv \
  --output AlphaTransfer/Researches/p2p_rub_kzt_rates/output
```

Проверки:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  AlphaTransfer/Researches/p2p_rub_kzt_rates/test_extract_p2p_rates.py

python3 AlphaTransfer/Researches/p2p_rub_kzt_rates/verify_outputs.py \
  --output AlphaTransfer/Researches/p2p_rub_kzt_rates/output \
  --manual-validation AlphaTransfer/Researches/p2p_rub_kzt_rates/MANUAL_VALIDATION.csv
```

## Артефакты

- `confirmed_rate_observations.csv` — 27 вручную проверенных автоматически принятых котировок;
- `rate_observations.csv` — полная audit-воронка accepted/review/rejected без авторов и сырого текста;
- `daily_rates.csv` — полный календарь в long-формате;
- `daily_rates_wide.csv` — основные effective-ряды в широком формате;
- `daily_card_transfer_rates.csv` — фокусный карточный сегмент;
- `daily_cash_rates.csv` и `daily_crypto_rates.csv` — отдельные наличный и crypto-сегменты;
- `review_sample.csv` — детерминированная стратифицированная выборка обезличенных фрагментов;
- `quality_report.json` — паспорт источников, воронка, покрытие и проверки;
- `run_manifest.json` — параметры и SHA-256 входов/выходов.
- `MANUAL_VALIDATION.csv` — результат ручной проверки всех 27 принятых строк по обезличенным ссылочным ключам.
- `ANALYTICAL_REPORT.md` — выводы, покрытие, ограничения и рекомендации по использованию.

## Ограничения

Это рынок объявленных/обсуждаемых котировок, не ledger совершённых сделок. `is_observed=true` означает наличие текстового свидетельства, а не settlement. Forward fill удобен для моделирования, но не является новой котировкой и должен использоваться вместе с возрастом наблюдения.
