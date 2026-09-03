# AlphaTransfer — RUB→KZT signal V0

Воспроизводимый сигнальный слой для коридора RUB→KZT. Target — нормализованный
курс ЦБ РФ; НБК и MOEX дают независимые признаки и robustness-проверки.
Генеративных моделей в контуре нет.

## Быстрый запуск

Нужен Python 3.11+; у реализации нет внешних зависимостей.

```bash
python -m pip install -e .
alphatransfer all --start 2020-09-03 --end 2026-09-03 --as-of 2026-09-03
```

Без установки: `PYTHONPATH=src python -m alphatransfer …`. Подробный контракт,
команды и ограничения описаны в [RUB→KZT V0](docs/kzt_v0.md). Проверка:
`PYTHONPATH=src:tests python -m unittest test_kzt_v0 -v`.

## Команды

- `fetch` — официальный CBR + NBK + MOEX, raw-кэш, retry, schema fail-fast;
- `backtest` — признаки и expanding walk-forward 36/6/3 месяца;
- `signal --as-of YYYY-MM-DD` — публичный JSON на любую дату;
- `report` — воспроизводимый HTML с графиком, OOT-метриками и `go/no-go`;
- `all` — все четыре стадии одной командой.

## Главные ограничения

Новый сигнал возможен только в дату свежей свечи MOEX. Last-known CBR/NBK не
считается новым наблюдением. Курс ЦБ — публичный ориентир, не исполнимая цена
перевода. V0 не подтверждает продуктовую конверсию, deep link или stale-price UX.

## Материалы кейса

- [Результат полного бэктеста](reports/2026-09-01_backtest_summary.md) —
  реальные out-of-time метрики и честный вывод по публичному дневному ряду.
- [Методология](docs/01_methodology.md), [комплаенс-тексты](docs/02_compliance_copy.md),
  [архитектура](docs/03_system_architecture.md) и [сценарий защиты](docs/04_evaluation_and_demo.md).
- [UX-механика при устаревшем пуше](star/README.md) и автономный
  [интерактивный прототип](star/prototype.html).
- [Ролевой продуктовый пакет по схеме n8n](agent_info/00_pipeline_map.md):
  Product Discovery, ICP, customer research, JTBD/VPC, hypotheses, RICE,
  Lean Canvas и market analogues.
