# AlphaTransfer: каталог открытых и лицензируемых данных

Дата среза: **2026-09-04**. Период загрузки основного research bundle:
2020-01-01—2026-09-03. Полный машинный provenance, URL запросов, SHA-256,
freshness checks, версии библиотек и 76 локальных артефактов находятся в
[`../../final_solution/data/data_manifest.json`](../../final_solution/data/data_manifest.json).
Загрузчик —
[`../../final_solution/data_pipeline/fetch_open_data.py`](../../final_solution/data_pipeline/fetch_open_data.py).

## Решение в одном абзаце

Для production нужен не максимально широкий набор, а два слоя. Первый —
операционный: доступный к decision time локальный `USD/LCY`, совместный nowcast
будущих RUB-якоря/local cross, собственная исполнимая котировка Alpha,
spread/fee/quote expiry. Второй — regime/context: CNY spot-minus-fixing
displacement (`basis` во внутренних feature IDs), рублёвое фондирование,
KASE FX/swaps/TONIA,
плановые продажи Нацфонда Казахстана, нефть, глобальный стресс и медленная
макростатистика. MOEX/KASE полезны, но их non-display/derived use и, возможно,
передача derived evidence жюри требуют договора или письменного разрешения;
обычная регистрация по email этого не решает. Bloomberg — возможный
лицензированный резервный канал и календарь ожиданий, но не открытый
воспроизводимый источник.

## 1. Что реально собрано

| ID | Источник | Даты / строк | Частота и доступность | Роль в модели |
|---|---|---:|---|---|
| `moex_cnyrub_tom` | MOEX ISS | 2020-01-03—2026-09-03 / 1 691 | daily close, conservative lag | главный исследовательский CNY/RUB spot |
| `moex_kztrub_tom` | MOEX ISS | 2020-01-03—2026-09-03 / 1 691 | daily close | direct KZT/RUB basis; active rows 1 519 |
| `moex_amdrub_tom` | MOEX ISS | 2022-06-27—2026-09-03 / 1 068 | daily close | direct AMD/RUB; active rows 1 067 |
| `moex_kgsrub_tom` | MOEX ISS | 2022-10-31—2026-09-03 / 978 | daily close | direct KGS/RUB; active rows 843 |
| `moex_uzsrub_tom` | MOEX ISS | 2022-09-12—2026-09-03 / 1 013 | daily close | direct UZS/RUB; active rows 652 |
| `moex_tjsrub_tom` | MOEX ISS | 2022-10-31—2026-09-03 / 978 | daily close | **непригоден как регулярный ряд**: лишь 17 active rows |
| `moex_imoex` | MOEX ISS | 2020-01-03—2026-09-03 / 1 674 | daily close | российский risk-on/off regime |
| `moex_rgbitr` | MOEX ISS | 2020-01-03—2026-09-03 / 1 676 | daily close | рублёвый duration/rate regime |
| `moex_rvi` | MOEX ISS | 2020-01-03—2026-09-03 / 1 678 | daily close | локальный stress/volatility gate |
| `moex_cny_fixing` | MOEX ISS | 2020-01-03—2026-09-03 / 1 681 | daily fixing/close proxy | CNY spot–fixing basis |
| `moex_rusfar` | MOEX ISS | 2020-01-09—2026-09-03 / 1 635 | daily close | RUB funding stress |
| `moex_rusfar_cny` | MOEX ISS | 2022-09-26—2026-09-03 / 906 | daily close | CNY funding/basis stress |
| `fed_h10_dollar_indexes` | Federal Reserve H.10 | 2020-01-01—2026-08-28 / 1 738 | daily observations, weekly release; lag 8d | global USD regime |
| `us_treasury_daily_curve` | US Treasury | 2020-01-02—2026-09-03 / 1 670 | daily after US close | global rates/risk regime |
| `eia_brent_daily` | EIA | 2020-01-02—2026-09-01 / 1 686 | observations daily, weekly update; lag 10d | oil shock/regime, not same-day alpha |
| `world_bank_commodity_prices` | World Bank Pink Sheet | 1960-01—2026-08 / 800 | monthly; lag 7d; latest-vintage workbook | oil/metals/food regime, eval with caveat |
| `kz_official_cpi` | Kazakhstan BNS | 2022-01—2026-08 / 56 | monthly; lag 7d; latest vintage | slow inflation/regime gate |
| `nbk_reserves` | NBK Open Data | 2020-01—2026-08 / 79 | monthly; lag 40d | FX buffer/National Fund regime |
| `nbk_current_account` | NBK Open Data | 2020-Q1—2025-Q2 / 22 | quarterly; lag 100d | external-balance stratification |
| `nbk_kase_monthly_rub` | NBK Open Data | 2025-01—2026-08 / 20 | monthly; lag 40d | RUB/KZT market depth regime |
| `nbk_interbank_monthly` | NBK Open Data | 2023-01—2026-08 / 412 | monthly; category keys incomplete | research only pending schema clarification |
| `nbk_daily_indicators` | NBK Open Data | 2026-02-24—2026-09-03 / 188 | daily, history too short | future shadow only; never model selection now |
| `cbr_international_reserves_weekly` | Bank of Russia | 2020-01-03—2026-08-28 / 348 | weekly; lag 7d | RUB macro regime |
| `cbr_business_climate_monthly` | Bank of Russia | 2020-01—2026-08 / 80 | monthly; lag 14d; current vintage | regime/eval-only sensitivity |
| `cbr_current_account_quarterly` | Bank of Russia | 2020-Q1—2026-Q1 / 25 | quarterly; lag 90d; revised | stratification, not live trigger |
| `nbk_business_activity_monthly` | NBK | 2017-01—2026-08 / 116 | monthly; lag 7d; current vintage | Kazakhstan cycle gate |
| `nbk_inflation_expectations_monthly` | NBK | 2016-01—2026-07 / 126 | monthly; lag 31d; current vintage | KZT policy/inflation regime |
| `ecb_new_ciss` | ECB | 2020-01-01—2026-09-02 / 1 749 | business-daily, about T+1; revisions possible | global systemic-stress gate |
| `cbr_key_rate` | Bank of Russia | 2020-01-03—2026-09-03 / 1 691 | effective daily; lag 1d in experiment | RUB policy regime |
| `nbk_base_rate` | NBK | 2020-02-04—2026-07-27 / 42 events | event dates; lag 1d | KZT policy regime |
| `project_official_fx_snapshot` | branch snapshots | CBR five corridors + KZT observations | source-specific clocks | target and mechanism audit |

Официальные endpoints: [MOEX ISS](https://www.moex.com/a8531),
[Fed H.10](https://www.federalreserve.gov/datadownload/Download.aspx?rel=H10),
[US Treasury yields](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?page=0&type=daily_treasury_yield_curve),
[EIA Brent](https://www.eia.gov/dnav/pet/hist/RBRTED.htm),
[World Bank Pink Sheet](https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/world-bank-commodities-price-data-the-pink-sheet),
[ECB CISS](https://data.ecb.europa.eu/data/datasets/CISS/data-information),
[NBK Open Data](https://data.nationalbank.kz/),
[Kazakhstan BNS dynamic tables](https://stat.gov.kz/en/industries/economy/prices/dynamic-tables/),
[Bank of Russia statistics calendar](https://www.cbr.ru/eng/statistics/indcalendar/).

## 2. Самое важное: механизм пяти target-рядов

С 27 декабря 2024 года Bank of Russia допускает построение официальных
курсов через cross-rates. Пункт 3.3 Указания 6956-У использует официальный
RUB-курс выбранной опорной валюты и cross национального банка либо
уполномоченной финансовой организации, доступный к 15:30 MSK; итог публикуется
с четырьмя десятичными знаками. 15:30 —
cutoff входных данных, **не** гарантированное время публикации результата:
точный publication time не регламентирован, обычно курс появляется до 18:00
MSK ([разъяснение CBR](https://www.cbr.ru/Reception/TopicalMessage/Page/2661)).
[Текст и формулы 6956-У](https://www.cbr.ru/Queries/XsltBlock/File/105012/-1/2531),
[объявление режима](https://cbr.ru/eng/press/pr/?id=40058). Действующий список
Bank of Russia от 4 июня 2026 прямо задаёт пары `USD/AMD`, `USD/KGS`,
`USD/KZT`, `USD/TJS`, `USD/UZS` для всех пяти коридоров:
[Annex OD-1012](https://www.cbr.ru/Content/Document/File/186003/Annex_OD-1012_en.pdf).

Для этих номиналов воспроизводимая формула имеет вид:

```text
CBR_RUB_per_nominal_LCY = round4(CBR_USD_RUB × nominal / local_USD_LCY)
nominal = AMD 100, KGS 100, KZT 100, TJS 10, UZS 10 000
```

Проверка совместимости в
[`mechanism_audit/summary.json`](mechanism_audit/summary.json):

- live snapshot 2026-09-04 — **5/5 точных** совпадений до четырёх знаков;
- KZT на выбранном strict-prior effective-date alignment после смены режима —
  **412/413 = 99.7579%**
  точных совпадений;
- средняя абсолютная относительная ошибка — `0.0213 bps`, максимум `2.3503 bps`.

Одна строка 2026-08-15 расходится на `2.3503 bps` и похожа на cutoff/source
alignment exception; конкретный источник denominator не доказан, поскольку
п. 3.3 допускает не только центробанк, но и уполномоченную финансовую
организацию. Поэтому безопасная формулировка — **почти точная совместимость с
USD-cross на выбранном alignment**, а не математическая identity на всей
истории. Alignment — выбранная arithmetic sensitivity, не идентификация
фактического source/clock ЦБ. Live 5/5 — вручную транскрибированный однодневный
snapshot без независимой raw/hash lineage всех пяти local inputs.

Это не «альфа» и не утечка, а структурный механизм contemporaneous official
reference. Он меняет архитектуру: target следует разложить на RUB-якорь,
локальный denominator и остаток относительно исполнимой котировки Alpha.
Однако NOW-h официального reference требует прогноза будущего пути **обоих**
cross-компонентов; Alpha residual дополнительно нужен для отдельной исполнимой
клиентской котировки/business utility. К decision time известна лишь
опубликованная часть. Поэтому сначала замораживается реальный decision clock,
затем для каждого входа явно задаётся `known / stale / nowcast`.

NBK отдельно документирует, что `USD/KZT` опирается на взвешенный курс
предыдущей утренней сессии KASE, прочие официальные курсы строятся как
cross-rates, официальный курс не является обязательством купить/продать, а
публикуемые данные финальны:
[NBK Exchange Rates metadata](https://nationalbank.kz/en/news/metadannye/7252).

## 3. Матрица допустимого использования

| Класс | Historical research/eval | Production feature | Условие |
|---|---|---|---|
| CBR/NBK/BNS official statistics | Да | Да, после проверки SLA/robots/attribution | хранить `observed_at`, `available_at`, vintage/source hash |
| NBK/BNS reuse | Да | В принципе да | условия разрешают копирование, изменение и software-use с атрибуцией: [NBK license](https://nationalbank.kz/en/page/data-usage-terms), [BNS terms](https://stat.gov.kz/ru/description/) |
| MOEX website/ISS market data | Локальный exploratory snapshot; **не распространять до legal sign-off** | **Нет по умолчанию** | MOEX относит programmatic/non-display/derived use к договорному, а website data не разрешает автоматически third-party derived use: [policy](https://www.moex.com/files/4a1jy8j83qc25vv9p286tzsmc1) |
| KASE prices/indices/indicators | Локальный exploratory snapshot; **не распространять до legal sign-off** | **Нет по умолчанию** | investment analysis/derived/non-display требует standard agreement: [KASE](https://kase.kz/en/information/non-display) |
| Bloomberg | Нет в открытом benchmark | Только по enterprise license | использовать как redundant feed, consensus/calendar/surprise, но не маскировать под open data |
| FRED | **Исключён** | Исключён | актуальные Terms запрещают FRED content для development/training ML и cache/archive: [FRED Terms](https://fred.stlouisfed.org/legal/) |

Юридическая оговорка не является заключением юриста. Перед пилотом владелец
данных/Legal должен письменно подтвердить конкретный endpoint, use case,
retention, derived-output и redistribution. Email-регистрация не заменяет
такого права.

## 4. Point-in-time классы

### A. Разрешены как live features

- цена/спред/fee/recipient amount/quote expiry самой Alpha на `decision_ts`;
- опубликованные к cutoff локальные `USD/LCY` национальных банков;
- законно лицензированные intraday CNY/RUB, USD/RUB proxy, KASE/MOEX FX;
- календарь публикаций и planned FX flows, если announcement timestamp сохранён.

### B. Historical eval или regime-gate

- latest-vintage CPI, current account, reserves, BAI, inflation expectations;
- World Bank commodities, CBR business climate/current account;
- дневные показатели с обновлением пачками после факта.

Для B допустимы два дизайна: (1) только стратифицировать ошибки/устойчивость;
(2) использовать как feature с консервативным release lag. Без immutable
vintage нельзя называть такой тест realtime-replicable: revision bias остаётся.

### C. Нельзя использовать в отборе текущей модели

- NBK daily indicators, начавшиеся лишь в феврале 2026;
- 2026 как «новый holdout»: он уже многократно просмотрен и неполон;
- будущие значения, выровненные по period-end вместо реального `available_at`;
- source rows без доказанного timestamp при intraday cutoff.

### CNY microstructure clock

Внутреннее имя `cnyrub_basis` в experiment — это
`log(EOD CLOSE CNYRUB_TOM) − log(CNYFIXME)` одной trade date. Это
post-fixing displacement, не cash-and-carry basis. По текущей странице MOEX
fixing определяется около 12:30 по наблюдениям 12:15–12:30 MSK; публичные
неавторизованные значения могут быть задержаны на 12 часов, realtime требует
подписки: [MOEX fixing](https://www.moex.com/en/fixing/). По расписанию валютного
рынка CNYRUB_TOM торгуется до 19:00, поэтому EOD close не существует в 15:30:
[MOEX FX schedule](https://www.moex.com/files/4fwbde0jzcmq517abt912ecj4b).
Старая опубликованная методика использовала окно 12:25:01–12:30, тогда как
текущая страница указывает 12:15–12:30. Исторический replay обязан хранить
`methodology_version` и event flag, а не переносить одну методику на все годы.

Ближайший к cutoff биржевой proxy — рассчитанный MOEX `CNYRUB_WAP0`: MOEX
описывает его как средневзвешенный `CNYRUB_TOM` за 10:00–15:30 и связывает с
установлением официального курса CBR, но это не идентифицирует фактический
CBR anchor и не доказывает availability к 15:30:
[MOEX WAP0 notice](https://www.moex.com/n65680),
[instrument page](https://www.moex.com/s3468). Право и SLA публичной
автоматизации всё равно проверяются отдельно.

Проверка неавторизованного MOEX ISS от 2026-09-04 показала важное ограничение:
метаданные `WAPS` отдают историю 2023-12-04—2026-09-03 и 700 календарных
строк, но на всех семи страницах `NUMTRADES=0`, а `WAPRICE=null`. Поэтому
бесплатная history инструмента не является пригодным историческим рядом WAP
на 15:30. Нужен лицензированный timestamped feed либо воспроизводимая
реконструкция из сделок с подтверждёнными правами. Машинный протокол проверки:
[`moex_wap0_access_probe_2026-09-04.json`](moex_wap0_access_probe_2026-09-04.json).

Следовательно, лаг T+1 в текущем backtest допустим только для заранее
замороженного next-day/morning decision clock. Для решения в 15:30 same-day
EOD close — look-ahead; нужен лицензированный timestamped mid/WAP/order book
`≤ decision_ts` и versioned исторический fixing methodology.

### Corridor-specific publication clock

| Компонент | Найденная механика | Безопасный статус до immutable archive |
|---|---|---|
| CBR output | 15:30 MSK — cutoff входов; exact publication time не регламентирован, обычно до 18:00; effective next calendar day | отдельный `decision_ts`; не путать cutoff/publication/effective date |
| AMD / CBA | Resolution 6L: публикация average rates 15:45–16:00 Yerevan = 14:45–15:00 MSK; входные other-FX около 15:00 Yerevan | потенциально до CBR cutoff, но мало запаса; проверить актуальный SLA/version |
| KGS / NBKR | setting 15:00–15:30 Bishkek = 12:00–12:30 MSK; historical notice — website после 16:00 local | обычно до cutoff; хранить фактический HTTP timestamp |
| KZT / NBK | USD/KZT основан на предыдущей утренней KASE session; published after setting | exact `published_at` отсутствует — только timestamped capture |
| UZS / CBU | с 10.08.2026 публикация около 17:30 Tashkent = 15:30 MSK | cutoff race; не считать same-day заранее доступным |
| TJS / NBT | есть daily read-only JSON API, точный publication timestamp не найден | T+1 до доказательства |

Источники: [CBR clock](https://www.cbr.ru/Reception/TopicalMessage/Page/2661),
[CBA Resolution 6L](https://www.cba.am/Storage/EN/regulations/Resolution%206L.pdf),
[NBKR rules](https://www.nbkr.kg/contout.jsp?item=2145&lang=RUS&material=39503),
[NBKR publication notice](https://www.nbkr.kg/searchout.jsp?item=31&lang=RUS&material=12209),
[CBU timing change](https://cbu.uz/ru/press_center/adverts/4257111/),
[NBT API](https://nbt.tj/en/kurs/document_gold_api.php).

## 5. Приоритет следующих источников

1. **P0, внутренний:** executable Alpha quote snapshots — bid/ask, spread, fee,
   recipient amount, expiry, provider and route. Канонический all-in rate для
   fixed notional `n`: `RUB_debited/LCY_received` (ниже лучше), с явным
   send/receive direction. Без этого bps официального курса не превращаются в
   клиентскую выгоду или contribution margin.
2. **P0, открытый:** машинно фиксировать публикации локальных центробанков с
   HTTP timestamp/response hash; для TJS timestamp считать неизвестным и
   использовать T+1, пока не доказано обратное.
3. **P1, договорный:** сначала MOEX `CNYRUB_WAP0`/timestamped mid с доказанным
   `observed_at/published_at ≤ decision_ts` и realtime CNYFIXME; окончание окна
   WAP ровно в 15:30 само по себе доступность к 15:30 не доказывает. Затем KASE
   USDKZT_TOD/TOM, RUBKZT, CNYKZT, swaps, TONIA,
   MOEX order book/RUSFAR/RUSFARCNY и roll-adjusted CNY/RUB/USD/RUB futures.
4. **P1, event data:** planned National Fund sales, interventions и rule-based
   conversion of export proceeds; хранить announcement timestamp и effective
   window, а не только месячное число. Для NBK тестировать ex-ante
   `planned_net_USD_supply / prior_20d_KASE_turnover / remaining_business_days`,
   не подмешивая будущий realised volume.
5. **P1, открытый lagged funding:** implied CNY/RUB overnight swap rate ЦБ —
   spread к RUONIA, delta, stress quantile и fallback flag. Это только D+1:
   публикация заявлена до 17:30 следующего рабочего дня, а при малом числе
   банков возможна резервная методика:
   [CBR methodology](https://www.cbr.ru/Content/Document/File/187851/IMPLIED_RATE_CNY_ENG.pdf).
6. **P2, regime:** Urals/Brent spread, metals/uranium/food indices, trade and
   current-account releases, fiscal/tax-payment calendar, global USD/stress.
7. **P2, China anchor:** CFETS/PBoC USD/CNY fixing и surprise относительно
   предыдущего close с реальным announcement timestamp:
   [ChinaMoney benchmark](https://www.chinamoney.com.cn/english/bmkcpr/).

Коридорные P1-кандидаты, которые стоит начать архивировать прямо сейчас:

- TJS: timestamped panel котировок банков NBT — median mid, best ask, MAD,
  p90 spread, stale/zero share и channel basis; доказанного публичного архива
  нет, поэтому только immutable forward-shadow:
  [NBT panel](https://nbt.tj/en/kurs/kurs_kommer_bank.php);
- AMD: daily cash/non-cash FX tape, forwards и swaps CBA, сначала lag1/lag2
  falsification: [CBA FX market](https://www.cba.am/en/Foreign%20Exchange%20Market/);
- KGS: межбанковские open/close, volume и settlement mix NBKR, только lag1:
  [NBKR interbank market](https://www.nbkr.kg/index1.jsp?item=118&lang=ENG);
- UZS: UZONIA policy spread, 7d–30d slope и stress/fallback flags, next-day:
  [CBU UZONIA](https://cbu.uz/en/monetary-policy/money-market-operations/uzonia/);
- общий RUB anchor: операции Минфина/ЦБ, government-account changes,
  Treasury deposits/repos и tax/month-end interactions, только опубликованное
  D−1: [CBR liquidity](https://cbr.ru/statistics/flikvid/).

Дополнительный timing-риск: UZS с 10.08.2026 публикуется около 17:30 Ташкента,
то есть примерно в 15:30 MSK — это cutoff race. Для TJS доказанного
publication timestamp не найдено. Их нельзя объединять одним blanket daily lag.

Стоп-критерий сбора данных: новый блок остаётся только если в nested/PIT ablation
даёт заранее заданный incremental выигрыш, устойчивый по годам и хотя бы двум
коридорам, либо улучшает калибровку/guardrail в отдельном режиме. «Экономически
правдоподобно» без OOT delta — недостаточно.

Наличие `CLOSE>0` в скачанном ISS-файле означает строку с ценой, но не
ликвидность. Для direct-pair guards нужны turnover, spread, depth,
last-trade time, quote age, zero/stale streak и session status. В текущем
research join разрешает carry-forward 7–45 дней для разных инструментов; это
годится лишь как явно помеченная sensitivity. Для production market feature
старое наблюдение ограничивается одним рабочим днём, а `age/stale` остаётся
отдельным признаком или причиной fail-closed.
