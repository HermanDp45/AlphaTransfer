# RUB→KZT V0

V0 — офлайн-исследование момента перевода, а не торговая рекомендация. Все
значения прямых рядов приведены к `RUB за 1 KZT`; меньше — выгоднее для
отправителя RUB.

## Запуск

Python 3.11+, внешних библиотек нет.

```bash
python -m pip install -e .
alphatransfer all --start 2020-09-03 --end 2026-09-03 --as-of 2026-09-03
```

Отдельные стадии:

```bash
alphatransfer fetch --start 2020-09-03 --end 2026-09-03
alphatransfer backtest
alphatransfer signal --as-of 2026-09-03
alphatransfer report
```

Конфигурация и все пути находятся в `configs/kzt_v0.toml`. `fetch` сохраняет
исходные ответы по SHA-256 URL и при повторном запуске работает из кэша.

## Контракт данных

- CBR: `Value / Nominal`; KZT — target, USD/CNY — плечи кросс-курсов.
- NBK: прямой RUB/KZT равен `Nominal / KZT value`; USD/CNY хранятся как
  `KZT per FX`.
- MOEX: каждый OHLC `KZTRUB_TOM / FACEVALUE`; значение FACEVALUE отдельно
  читается из ISS и должно совпасть с конфигурацией (100).
- В observation CSV остаются `effective_date`, `available_at`, источник,
  symbol, field, исходные nominal/value, normalized value/unit.
- Исторический `available_at` реконструирован консервативно: начало
  `effective_date` для официальных курсов и момент после дневного закрытия для
  MOEX. Это не timestamp фактической HTTP-загрузки.
- Feature-строка существует только для фактической свечи MOEX. Last-known
  CBR/NBK содержит `age_days`; stale CBR не создаёт return, streak или rebound.

## Модель и policy

Target `favorable_now` — уникальный локальный минимум CBR в `±h` MOEX-сессий.
Logistic regression — baseline; shallow gradient boosting из decision stumps —
challenger. Модель, гиперпараметр и порог выбираются на предыдущем validation,
после чего замораживаются для untouched test. Ошибка FP стоит 3× FN.

`window_closing` требует сильного недавнего кандидата и уже наблюдаемого
отскока. Календарь сам кандидата не создаёт. После выбора сценария применяются
cooldown в MOEX-сессиях и cap за 7 календарных дней. Отправка дополнительно
закрыта, если OOT lift сценария ниже 1.3.

Итоговые артефакты:

- `data/kzt_v0/backtest.json` — фолды, модели, события и метрики;
- `data/kzt_v0/calendar.csv` — все календарные дни с holidays/freshness;
- `data/kzt_v0/signal.json` — публичный контракт сигнала;
- `reports/kzt_v0_app/dist/index.html` — self-contained отчёт;
- `reports/kzt_v0.html` — короткий путь к тому же содержимому.

Overall `go` требует одновременно lift ≥1.3 хотя бы одного сценария,
подтверждение lift ≥1.3 минимум двумя из CBR/NBK/MOEX и отсутствие базовых
ошибок grain/completeness. Это более строгий исследовательский gate, чем
delivery gate отдельного сигнала.

Праздничный справочник содержит закреплённые законом даты и перенос праздника
с выходного. Нерегулярные правительственные переносы не угадываются: фактическое
наличие/отсутствие торгового дня берётся из свечей MOEX и имеет приоритет.
