# Рабочий профиль TabM KZT H3

`final_solution/main.py` по умолчанию запускает этот профиль. `--legacy` сохраняет старый H5 CLI с `config.legacy.toml`; исходный `config.toml` оставлен неизменным для проверки старого input lock. `--research-v3` остаётся отдельным историческим просмотром. Активная модель и политика H3 описаны в `tabm_h3/bundle.json`.

## Один полный запуск

Из AlphaTransfer:

```bash
python final_solution/main.py --output-dir final_solution/tabm_h3/output
```

Bundle задаёт дату запуска 2026-09-05 и режим `historical_smoke`; последний source feature date — 2026-09-03. Это **датированный демонстрационный score финальной модели в её calibration-периоде**, не OOT и не live execution. CLI принимает календарную дату YYYY-MM-DD. Intraday timestamps отклоняются; время feature batch фиксировано как 10:05 Москвы.

Прямой запуск профиля:

```bash
python -m final_solution.tabm_h3 --bundle final_solution/tabm_h3/bundle.json --as-of 2026-09-05 --mode historical_smoke --output-dir /tmp/alphatransfer-h3
```

`operational` требует feature rows после model cutoff и актуальную feature date. Обновлённый input prefix должен продолжать `operational_state.json` без пропуска или повторной обработки сессий. Инкрементальный запуск использует `--state-in` и `--state-out`. Повторная обработка уже завершённого prefix возвращает `no_new_source_sessions` и не создаёт второй сигнал.

## От источников до решения

`features.py` читает только семь явно указанных нормализованных CSV: CBR, MOEX CNY close/fixing, OXR, Halyk BANK SELL, две Treasury series. Ни одного research import или network call. Все 33 признака совпали побитово с frozen training panel на 4116 исторических KZT датах; расширение до 2–3 сентября не изменило прошлый prefix.

- CBR: текущий effective reference RUB/KZT и 14 returns/ranks/volatility.
- MOEX: close minus fixing только при совпадении торговой даты; доступность D+1 и предел возраста 7 дней от observation.
- OXR: max(published_at, конец дня UTC)+24h; as-of join с tolerance 7 дней; basis, изменения, z-score, availability и age.
- Halyk: отдельные BANK SELL RUB/USD series, D+1; tolerance 7 дней от доступности. BANK SELL RUB не является исполнимым RUB→KZT курсом клиента.
- Treasury: два breakeven proxy, D+7, предел возраста 14 дней от observation; значения до 2020 оставлены missing, как в обучении.

Оригинальный pandas float parser сохраняется при чтении raw source CSV: изменение парсинга может сдвигать точки ровно на границах эмпирических quantiles. H3 target и future utility **не вычисляются** в runtime. Они нужны только offline evaluator.

`model.py` загружает официальный TabM, numerical PeriodicEmbeddings, сохранённый sklearn preprocessing и state_dict после проверки SHA. Код категории KZT — 2 среди пяти исходных категорий; встроенное кодирование TabM one-hot. Numeric input: median + quantile-to-normal, все missing indicators, затем float32. Calibration выполняется в float64. Thread count 1 и 2 дают одинаковые native raw scores на проверенном финальном checkpoint. Отличие от CSV порядка 3e-8 — сериализация исходного float32 в короткую decimal-строку; после обратного приведения CSV к float32 raw scores совпадают точно. Rank и candidate decisions воспроизведены точно.

## Causal rank80 и состояние

`policy.replay(rows, config, state)` — pure stdlib API. Входные поля строго ограничены `date,corridor,session_ordinal,probability`. Любые target/future utility поля отклоняются. Текущий rank вычисляется по **предыдущим** максимум 63 probabilities; текущая вероятность добавляется в историю после решения. Минимум 20 прошлых scores, exact-tie midrank, cutoff сравнивается с past-fitted quantile. Максимум два market candidates за календарную неделю; между выбранными сессиями разница ordinal строго больше 2.

Состояние содержит последние 63 scores, последний обработанный ordinal/date, прошлый candidate ordinal и недельный счётчик. `binding_sha256` связывает его с model hashes, features, calibration, policy, cutoff и H3 contract. Нельзя переносить состояние на другой checkpoint или порог. Пропущенные, повторные и переставленные сессии отклоняются. Входное состояние не изменяется; возвращается новая копия.

`replay(..., emit_candidates=False)` прогревает только scores. При packaging используются 63 panel даты перед calibration, включая purged training tail, затем воспроизводится calibration/history с той же frozen моделью и calibration. Это техническая история состояния, а не дополнительные OOT наблюдения или реально отправленные сообщения.

## Два сценария с независимыми метками

Основной target:

`NOW_H3 = I(R[t] <= min(R[t+1], R[t+2], R[t+3]))`, R = RUB за 1 KZT.

Дополнительная CLOSING-голова имеет собственные checkpoint, probability и target:

`CLOSING_H3 = I(R[t+3] > R[t])`.

Она может добавить лишь вторую аннотацию на уже существующий NOW candidate, при собственном пороге и наблюдаемом `ret1>0`. Новых контактов и замены NOW probability нет. В финальном bundle CLOSING **disabled**: годовая annotation-policy не прошла lift 1.3 в 2025. CLI сохраняет диагностическую вероятность/condition, но `annotations=[]`, `annotation_active=false` и не добавляет клиентский вердикт «окно закрывается». Старый H5 CLOSING здесь не используется.

## Продуктовый результат

`signal_decision.json` явно разделяет run_as_of, feature_date, feature_known_at, model_cutoff, H3 contract, probability, rank, verdict и factual context. В теле сообщения — только дата и текущий опубликованный reference-курс. Нет гарантии будущей цены. `predictions.csv` — обработанные даты и отдельные диагностические CLOSING-поля; `next_state.json` — продолжение; `run_receipt.json` — hashes входов и результатов.

`authorized_contact=false`, внешние сообщения не отправляются. Исполнимая котировка Alpha, сумма получателю и клиентские CRM-ограничения не представлены этими historical source snapshots; market candidate не подменяет разрешение на отправку.

## Проверки

```bash
python -m unittest discover -s final_solution/tabm_h3/tests -p 'test_runtime.py' -v
```

15 meaningful tests: future-source poison, prefix invariance, source publication delay, quote-side guard, intraday rejection, missing feature warmup, batch/incremental equality, score-only warmup, state binding and immutability, skipped sessions, strict past rank, weekly cap, disabled/independent CLOSING semantics. Дополнительно выполнен полный запуск через main из bundled raw CSV; legacy/research routing проверяется отдельно. `runtime_verification.json` хранит итоговый статус интеграции.
