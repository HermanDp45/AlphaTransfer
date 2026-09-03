# Исследование Telegram-чата обменов Казахстана

В этой директории лежат воспроизводимые и обезличенные артефакты исследования `result.json` для кейса AlphaTransfer.

## Главный артефакт

- `report.md` — продуктовый отчёт: вывод, JTBD, сегменты, проверка timing-гипотезы, риски и ограничения для сигнального слоя.
- `../ml_hackathon/HACKATHON_ML_PLAN.md` — приоритетный backlog ML/алгоритма/MVP и проверки, которые реально выполнить на хакатоне.
- `../ml_hackathon/results/` — воспроизводимый аудит данных, baseline-матрица и таблица сигналов.

## Воспроизведение

Скрипты используют только стандартную библиотеку Python и не записывают исходные тексты, имена, handles, телефоны, карты, email или Telegram ID.

Из корня общей директории `ai product hack`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 AlphaTransfer/Researches/chat_research_kz/analyze_chat.py \
  --input result.json \
  --rates AlphaTransfer/data/open_exchange_rates/rub_cis_daily.csv \
  --output AlphaTransfer/Researches/chat_research_kz/data

PYTHONDONTWRITEBYTECODE=1 python3 AlphaTransfer/Researches/chat_research_kz/derive_slices.py
PYTHONDONTWRITEBYTECODE=1 python3 AlphaTransfer/Researches/chat_research_kz/build_charts.py
PYTHONDONTWRITEBYTECODE=1 python3 AlphaTransfer/Researches/chat_research_kz/verify_outputs.py
```

Первый шаг потоково читает Telegram JSON объёмом около 320 МБ и обычно занимает несколько минут.

## Данные и графики

- `data/summary.json` — паспорт корпуса, участие, темы, широкий автоматический screen и связь с рынком. Автоматический P2P-screen внутри не используется как prevalence.
- `data/p2p_validation.json` — отдельно валидированное high-precision ядро P2P-заявок и precision ручной проверки.
- `data/monthly.csv` — агрегаты для безопасного командного отчёта.
- `data/daily.csv` — внутренний аналитический ряд для воспроизводимости. Не публиковать без недельной агрегации и suppression редких ячеек `<5` (лучше `<10`): отсутствие прямых идентификаторов не устраняет membership-inference риск в приватном чате.
- `data/categories.csv`, `data/banks_and_rails.csv`, `data/directions.csv` — тематические агрегаты.
- `data/author_bins.csv` — распределение активности без идентификаторов.
- `data/robustness_slices.json` — проверка market-response на нескольких периодах.
- `assets/*.svg` — графики из агрегатов.

Категории построены эвристиками и пересекаются. Их значения — сигналы для проверки и верхние оценки упоминаний, а не доли клиентов или операций. Качественные выводы дополнительно проверены на стратифицированной ручной выборке эпизодов; подробности приведены в отчёте.
