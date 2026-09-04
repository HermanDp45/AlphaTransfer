# AlphaTransfer: executive decision memo

Дата среза: **2026-09-04**. Объект ревью: ветка `edelkin_test`, актуальный
контекст `main`, постановка и две версии Q&A, продуктовый код, сохранённые
результаты и новый независимый quant/macro-контур.

Зафиксированные heads: `edelkin_test@745fdc3`, `main@87de36f`, merge-base
`0c3f0d2`. В репозитории основная ветка называется `main`, не `master`.
Сводная Q&A присутствует в tree `main`, хотя отсутствует в working tree
`edelkin_test`; она была прочитана непосредственно из `main`. Diff ветки —
263 файла и около 71.9k добавленных строк, значимая часть — generated review/UI
assets; размер diff сам по себе не является ML evidence.

## Решение

**Production-сигнал — NO-GO.** Опубликованные в ветке `lift=10.57` для NOW и
`100% hit` для CLOSING нельзя защищать: target/evaluator и последующий отбор
дают методически невалидный headline. Новый контрактный CNY/RUB challenger
интересен и проходит retrospective h=5 multiplicity screen, но остаётся
diagnostic: он проваливает cadence, point-in-time, data-rights и explanation
gates, а весь protocol формировался после просмотра исторических outcomes.

**GO на следующий этап — да:** собрать deterministic cross-rate engine,
заморозить один компактный challenger и после закрытия P0 data contracts начать
prospective immutable shadow без клиентских контактов. Самое сильное открытие
исследования — не «ещё одна ML-модель», а почти точная арифметическая
совместимость cross-формулы в ручном all-five snapshot и KZT-истории на
выбранном alignment после смены методики в конце 2024 года.

| Решение | Статус | Почему |
|---|---|---|
| Защищать headline ветки | **NO-GO** | неверные truth/evaluation, малый N, future/post-selection defects |
| Использовать CBR cross engine | **GO engineering; shadow после P0** | live snapshot 5/5; KZT 412/413 совпадений на выбранном alignment; actual denominator/source не установлен |
| Назвать `hgb + CNY spot-minus-fixing` победителем | **NO-GO** | timing-sensitive, post-selection, legal/PIT/cadence gates не пройдены |
| Добавлять весь macro stack | **NO-GO** | широкий stack ухудшает Brier относительно компактного CNY-блока |
| Запускать клиентский push | **NO-GO** | нет executable Alpha quote, CRM arbitration и causal ITT evidence |
| Запустить frozen prospective shadow | **GO после P0-контрактов** | единственный честный способ получить новый confirmatory период |

## Четыре вывода, которые меняют решение

### 1. Пять targets в значительной мере являются одной задачей

С 27.12.2024 Bank of Russia может считать AMD/KGS/KZT/TJS/UZS через общий
RUB-якорь и локальный `USD/LCY`. В нашей проверке формула совместима с live
snapshot 2026-09-04 в 5/5 случаях и с KZT-историей в 412/413 случаях. Это
**не predictive alpha**: модель не должна повторно учить известную формулу.
Исторический результат использует выбранный strict-prior effective-date
alignment без publication timestamps и не идентифицирует фактический источник
denominator ЦБ; live 5/5 — ручная arithmetic illustration.
Формула аппроксимирует contemporaneous official reference. Его NOW-h зависит
от будущих движений RUB-якоря и local `USD/LCY`. Alpha quote residual — уже
отдельный обязательный компонент прогноза исполнимой клиентской котировки и
business utility; все компоненты моделируются с учётом того, что реально
опубликовано к замороженному decision time.

### 2. Единственный сильный быстрый блок — CNY spot-minus-fixing, но он хрупок

На development OOT 2023–2025 для `h=5` **CBR proxy rows** (не календарных дней)
`hgb_plus_cnyrub_basis` улучшает Brier против
`hgb_base` на **12.85%**, улучшая 3/3 лет и 15/15 `fold × corridor` cells.
Contract-required corridor policy даёт 663 candidate events, point lift
**1.435**, forward delta **+47.69 bps** и symmetric `±h` delta **+26.80 bps**.
Внутреннее имя `basis` означает `log(CNYRUB_TOM close) − log(CNYFIXME)` той же
trade date: это spot-minus-fixing displacement, **не** cash-and-carry basis;
EOD close формируется после окончания сессии и недоступен к 15:30. Сдвиг
доступности MOEX с одного до двух дней меняет Brier
skill примерно с **+13.17% до −2.73%**, а candidate lift — с **1.435 до
1.065**. Пока реальный cutoff/published timestamp не доказан, это ассоциация,
а не внедряемый forecast.

### 3. «Больше данных» оказалось хуже

Полный research stack хуже компактного CNY-basis по Brier примерно на **6.97%**;
весь contract-market stack — примерно на **11.98%**. Public Kazakhstan
official stack не улучшил logistic baseline. Нефть, ставки, stress, CPI,
reserves и current account полезнее оставить regime/stratification features и
возвращать по одному блоку через nested PIT ablation, а не скармливать модели
одновременно.

### 4. Найден честный quality–cadence Pareto conflict

При мягком quantile policy cadence проходит, но lift остаётся около
`1.10–1.15`; при порогах, где lift достигает `1.38–1.42`, минимальная доля
недель с 1–2 candidate signals падает примерно до `0.84–0.76`. Ни одна
проверенная точка не дала одновременно `lift ≥ 1.3` и weekly fulfillment
`≥ 0.90`. Тихие недели нельзя заполнять искусственно: это расход доверия и
CRM-слота. Правильная policy — abstain, а недостающий объём добирать большим
числом eligible клиентов/коридоров или длиннее пилотом.

## Что было хорошо и плохо в `edelkin_test`

### Сильные стороны

- есть полноценный ingestion/features/models/evaluation/policy/CLI/reporting
  skeleton, as-of интерфейс, конфиги и unit tests;
- реализованы базовые индикаторы, cooldown, сценарии NOW/CLOSING, explainability
  payload и прототип клиентского пути;
- авторы увидели проблему разных clocks и попытались сравнивать CBR/NBK/MOEX;
- UX/engineering groundwork пригоден для переиспользования после замены
  evaluator и policy contract.

### Критические ошибки

1. NOW headline измерял не Q&A truth; CLOSING truth был фактически
   тавтологичным относительно trigger.
2. Selection и evaluation повторно использовали уже просмотренные outcomes;
   published `OOT` нельзя считать untouched.
3. Forward labels пересекали границы folds, clocks разных источников
   выравнивались нестрого, а `available_at` не был доказан реальным timestamp.
4. 17 policy-eligible событий, 5/7 folds без сигналов и месяцы тишины не
   поддерживают заявленный production claim.
5. Пять строк одной market date и перекрывающиеся horizons трактовались слишком
   близко к независимым наблюдениям.
6. Официальный курс подменял клиентскую исполнимую котировку; bps не являются
   savings/P&L без spread, fee, route и quote expiry.
7. Частота corridor candidates смешивалась с реальным лимитом контактов на
   клиента после конкуренции с другими CRM-кампаниями.

Подробный исходный аудит сохранён в
[`ML_REVIEW_2026-09-03.md`](ML_REVIEW_2026-09-03.md); старые эксперименты явно
помечены legacy и не смешиваются с новым h=5 protocol.

## Зрелый offline-контракт

Одна цифра lift недостаточна. Решение принимается по пяти слоям:

1. **Truth:** NOW = курс в T не хуже каждого из следующих `h`; CLOSING имеет
   отдельный заранее замороженный endpoint. Все `h={1,3,5,10,20}` публикуются,
   один primary фиксируется до confirmatory labels.
2. **Probability:** Brier/log loss против prior-year climatology,
   calibration intercept/slope/ECE; calibration fit только на прошлом.
3. **Decision:** cell-standardized lift, hit-rate delta, обязательные `±h` bps,
   forward bps и regret с date/month/episode block intervals.
4. **Operations:** число candidates, delivered contacts и каждый suppression
   reason раздельно; cadence на полном nominal exposure, silent weeks,
   p90/max gap, concentration, CRM competition.
5. **Readiness:** immutable `observed_at/available_at/vintage/hash`, legal
   rights, deterministic replay, per-signal explanation и новый prospective
   shadow. Любой отсутствующий P0 gate даёт fail-closed.

Полные определения: [`METRIC_CONTRACT_V2.md`](METRIC_CONTRACT_V2.md).
Randomized ITT pilot: [`PILOT_METRIC_CONTRACT.md`](PILOT_METRIC_CONTRACT.md).

## Рекомендуемая архитектура

```text
available local USD/LCY + licensed RUB-anchor feed + Alpha executable quote
                         │
                         ▼
deterministic cross-rate/reference engine
                         │
                         ▼
joint RUB-anchor/local-cross nowcast + Alpha residual/uncertainty
                         │
                         ▼
expected utility + abstention + corridor cooldown
                         │
                         ▼
global CRM arbitration + delivery/suppression ledger
```

### Модели в порядке ценности

1. deterministic contemporaneous cross + no-change/prior-year/rule baselines;
2. regularized logistic/GAM/EBM для прозрачного nonlinear basis;
3. shallow HGB как frozen challenger; depth 2 пока не даёт доказанного прироста;
4. hierarchical GLM с общим RUB-factor и corridor residual;
5. discrete-time hazard для согласованных `h=1/3/5/10/20`;
6. quantile/regret model для `send / wait / abstain`;
7. Kalman/state-space joint nowcast RUB-якоря и local `USD/LCY`, когда
   denominator ещё не опубликован;
8. MIDAS только для mixed-frequency macro с реальными release vintages;
9. block/sequential conformal uncertainty после накопления shadow.

Deep nets, Transformers, RL и LLM-news predictor сейчас имеют отрицательную
ожидаемую ценность: эффективная выборка мала, regime shifts велики, а основной
дефицит — clock/execution/policy evidence.

## Данные: что собрать следующим

Уже собраны 31 source family и 76 raw/normalized/provenance artifacts за
2020–2026: MOEX FX/indices/funding, Fed H.10, Treasury, EIA Brent, World Bank
commodities, ECB CISS, CBR/NBK rates/reserves/balance/cycle и Kazakhstan CPI.
Каталог: [`quant_macro/SOURCE_CATALOG_2026-09-04.md`](quant_macro/SOURCE_CATALOG_2026-09-04.md).

Приоритет дальнейшего сбора:

- **P0:** собственные executable Alpha quotes, нормализованные как
  `RUB_debited/LCY_received` для fixed notional: bid/ask, fee, recipient amount,
  direction, provider/route, expiry и client decision/open timestamps;
- **P0:** immutable снимки публикаций национальных банков с HTTP timestamp,
  timezone, raw response/hash; TJS считать T+1 до доказательства;
- **P1 licensed:** сначала MOEX `CNYRUB_WAP0`/mid с фактическим
  `observed_at/published_at ≤ decision_ts` и realtime fixing; окончание окна
  WAP в 15:30 ещё не доказывает availability. Затем order book, futures curve
  и RUB liquidity; KASE
  USDKZT/RUBKZT/CNYKZT, swaps/TONIA; National Fund planned
  sales/interventions с announcement time;
- **P2 regimes:** Urals–Brent spread, metals/uranium/food, tax/fiscal calendar,
  sanctions/methodology-change events, CFETS/PBoC USD/CNY fixing surprise,
  global USD/stress/rates.

Открытый ISS не заменяет договорный intraday feed: контрольный запрос `WAPS`
нашёл 700 исторических строк, но ни одного ненулевого `WAPRICE`. Поэтому WAP0
пока является спецификацией следующего cutoff-эксперимента, а не уже доступным
production-признаком. Протокол:
[`quant_macro/moex_wap0_access_probe_2026-09-04.json`](quant_macro/moex_wap0_access_probe_2026-09-04.json).

Email-регистрация сейчас не нужна. MOEX/KASE non-display/derived use и
Bloomberg Data License требуют согласованного договора/прав, а не создания
обычного аккаунта. До legal sign-off нельзя распространять соответствующие
raw/derived артефакты: см.
[`quant_macro/DISTRIBUTION_AND_LICENSE_README.md`](quant_macro/DISTRIBUTION_AND_LICENSE_README.md).
Chat-derived research из `main` не использовался как ML data: raw messages,
handles и любые персональные данные нельзя включать в сдачу без подтверждённых
consent/provenance и разрешённого контура; допустимы только проверенные
обезличенные агрегаты.

## План, максимизирующий шанс победы

### До защиты / 48 часов

1. В презентации начать со structural discovery и показать manual 1-day
   all-five arithmetic compatibility check 5/5 с неизвестным actual source, а
   не с «волшебного lift».
2. Показать invalid старый headline → исправленный evaluator → честный compact
   challenger → cadence frontier. Это сильнее попытки спрятать отрицательный
   результат.
3. Заморозить demo на public-only данных; contract-required CNY result показать
   как отдельный исследовательский трек с legal disclaimer.
4. На слайде architecture показать, где заканчивается official reference и
   начинается Alpha executable utility.
5. Зафиксировать один primary horizon и одно utility rule до нового периода.

### Следующие 2–4 недели

1. Собрать production-like immutable source clock и Alpha quote snapshots.
2. Реализовать deterministic cross engine, residual target, explanation и
   полный suppression ledger.
3. Cross-fit calibration/threshold во внутреннем temporal split; bootstrap
   стратифицировать по outer fold и проверить block-length sensitivity всех
   policy endpoints.
4. Frozen shadow без контактов; weekly data-quality review без просмотра
   outcome-based model ranking.

### После достижения frozen shadow information target

8–12 недель достаточно лишь для operational ramp и проверки collector/SLA,
но не для confirmatory market-regime вывода. Финальный shadow readout делается
один раз после заранее рассчитанного числа predeclared non-overlapping
decision-date/episode blocks с block-aware inference; при текущей cadence это,
вероятно, существенно дольше. Если gates
пройдены — persistent client-level 50/50 ITT pilot с primary `net contribution
margin / randomized client` на 90 днях; 7/30/60 дней — secondary trajectory,
а inference учитывает client и market-episode dependence. Ретроспективный lift
не заменяет этот тест.

## Формулировка для защиты

> Мы не утверждаем, что научились безошибочно предсказывать курс. Мы нашли,
> что ручной all-five snapshot и KZT-история на выбранном alignment почти точно
> совместимы с contemporaneous cross-формулой, не выдавая это за доказательство
> источника всех пяти рядов; отделили официальный reference forecast от
> отдельного прогноза Alpha executable residual,
> измерили единственный полезный быстрый proxy, обнаружили его timing-риск и
> доказали конфликт качества с коммуникационной частотой. Поэтому предлагаем
> проверяемый cross/residual engine, abstention и замороженный shadow-to-pilot
> протокол, который не расходует доверие клиента на ложный сигнал.

Это амбициознее model zoo: целевая policy должна знать, **когда молчать**,
объяснять каждый контакт и заранее задавать условия, при которых банк остановит
запуск. Текущий prototype этих требований ещё не выполняет.
