# AlphaTransfer: независимый quant/macro research report

Дата: **2026-09-04**. Статус: **final retrospective exploratory**.
Production и prospective promotion: **0 моделей**.

Канонический результат перенесён в каталог
[`../../final_solution/model_bundle/`](../../final_solution/model_bundle/) с
`10 000` bootstrap repetitions и `_SUCCESS.status=complete`. SHA-256 manifest:
`24a2ce0d0b76f895a3ef83d41ce20c3b21f7895206d2198e79ded2b2647e5361`.
Проверены 5 code, 34 input и 35 output hashes, exact output whitelist и
153 715 prediction rows: 141 765 development, 3 120 diagnostic-2026 и 8 830
post-hoc policy rows. `available_date > decision_date` и отрицательных source
ages не найдено.

## 1. Что можно и нельзя утверждать

Допустимо:

> На latest-snapshot данных и при **предположении** о доступности MOEX T+1
> compact CNY/RUB-basis block показывает сильную и устойчивую по development
> folds ретроспективную ассоциацию с h=5 NOW target. Это перспективная
> гипотеза для frozen prospective shadow.

Недопустимо называть это готовой моделью, predictive alpha, калиброванной
вероятностью, client savings/P&L или production/shadow-ready решением.
`3 635` corridor rows — не независимый N: в парном development сравнении лишь
`727` decision-date clusters. Значение `+47.69 bps` ниже относится к
официальному reference-rate backtest, а не к исполнимой котировке Alpha.

## 2. Дизайн исследования

- Target: NOW, `rate[T] ≤ min(rate[T+1:T+5])`; `h=5` CBR effective-date rows,
  использованных как publication-session proxy. Sensitivity сохранена для
  `h={1,3,10,20}`, но весь поиск остаётся post-selection.
- Development test years: 2023, 2024, 2025. Частичный и уже просмотренный 2026
  — только diagnostic.
- Split: rolling train → purged validation → purged test; хвосты очищаются на
  max forward horizon. Imputation/model/calibration/threshold не fit на test.
- Unit inference: decision date; пять коридоров даты остаются в одном block.
- Model-family endpoint: paired Brier delta vs соответствующий base; month/date
  blocks, sign-flip и Holm across 37 h=5 family rows.
- Policy endpoint: cell-standardized lift/hit/bps/regret; month-block intervals
  на 36 development months и simultaneous per-corridor screen.
- Calibration: monotone Platt на prior validation с identity fallback. Но
  calibration и threshold используют один validation interval — для
  confirmatory protocol нужен внутренний temporal split/cross-fit.
- Каждая внешняя серия joined назад по explicit availability proxy. Это
  предотвращает календарный lead внутри кода, но не доказывает исторический
  point-in-time: у части источников нет immutable `published_at`/vintage.

## 3. Structural mechanism — главный результат

С 27.12.2024 методика Bank of Russia допускает расчёт официального курса через
RUB-anchor и официальный cross национального банка/уполномоченной финансовой
организации, доступный к 15:30 MSK. Действующий Annex от 04.06.2026 перечисляет
`USD/AMD`, `USD/KGS`, `USD/KZT`, `USD/TJS`, `USD/UZS`.

```text
CBR_RUB_per_nominal_LCY = round4(CBR_USD_RUB × nominal / local_USD_LCY)
```

Проверка [`mechanism_audit/summary.json`](mechanism_audit/summary.json):

- вручную зафиксированный live snapshot 2026-09-04 — 5/5 совпадений;
- KZT history на выбранном strict-prior effective-date alignment —
  412/413 = 99.7579%;
- mean absolute relative error `0.0213 bps`, max `2.3503 bps`;
- исключение 2026-08-15 похоже на cutoff/source alignment, но источник
  denominator не доказан.

Корректная формулировка — **почти точная совместимость с USD-cross на выбранном
alignment**, а не математическая identity. Поскольку publication timestamps
нет, выбранный alignment не идентифицирует фактический denominator/source ЦБ;
live 5/5 — ручная однодневная arithmetic illustration без полной raw/hash
lineage. Следствие для архитектуры: отдельно
считать contemporaneous deterministic reference. Но NOW-h официального
reference зависит от будущих движений RUB-anchor и local `USD/LCY`: cross
engine сам по себе не является forecast. Отдельно для исполнимой клиентской
котировки нужен прогноз `Alpha executable quote − reference` residual. К
decision time каждый вход маркируется как `known/stale/nowcast`.

Первичные источники: [CBR 6956-U](https://www.cbr.ru/Queries/XsltBlock/File/105012/-1/2531),
[CBR Annex OD-1012](https://www.cbr.ru/Content/Document/File/186003/Annex_OD-1012_en.pdf),
[CBR regime notice](https://cbr.ru/eng/press/pr/?id=40058),
[NBK exchange-rate metadata](https://nationalbank.kz/en/news/metadannye/7252).

## 4. Feature-family ablation

Primary comparison panel: 3 635 rows = 727 dates × 5 correlated corridors.

| Challenger / comparison | Brier | Относительный результат | OOT stability | Решение |
|---|---:|---:|---:|---|
| `hgb_base` | 0.218714 | baseline | — | сохранить |
| `hgb_plus_cnyrub_basis` vs base | **0.190617** | **+12.846%** | 3/3 years, 15/15 cells | compact research hypothesis |
| CNY basis stump vs standard | 0.190776 | −0.084% incremental | 1/3 years, 6/15 cells | complexity winner не доказан |
| CNY basis + `ret1` vs basis | 0.191052 | −0.229% incremental | 1/3 years, 5/15 cells | исключить |
| full research vs CNY basis | 0.203905 | **−6.971%** | 1/3 years, 5/15 cells | исключить широкий stack |
| all contract market vs CNY basis | 0.213445 | **−11.976%** | 1/3 years, 4/15 cells | исключить |
| all public fast vs base | 0.231911 | **−6.034%** | 2/3 years, 4/15 cells | исключить |
| public KZ official logit vs logit base | 0.218628 | **−0.386%** | 0/3 years, 1/15 cells | public-only winner нет |

Для CNY basis paired Brier delta `−0.028097`, month-block CI95
`[−0.038115; −0.017824]`, Monte Carlo sign-flip `p=0.000100`, Holm
`p=0.003700` по
37 family rows. Block-length sensitivity 10/20/40 decision dates сохраняет
отрицательный CI. Это сильный retrospective result, но не снимает timing и
post-selection blockers.

Добавление full research и all-contract наборов статистически ухудшает
compact CNY model: их CI для Brier deterioration относительно CNY basis не
пересекают ноль. Вывод: продолжать **block-by-block ablation**, а не model zoo.

Название `CNY/RUB basis` исторически сохранено в config ID, но экономически это
`log(CNYRUB_TOM close) − log(CNYFIXME)` одной trade date, то есть
spot-minus-fixing displacement, не cash-and-carry basis.

## 5. Policy result и его граница применимости

У post-selected `hgb_plus_cnyrub_basis` corridor policy:

| Метрика, candidate layer | Estimate | Month-block CI95 |
|---|---:|---:|
| Signals | 663 | — |
| Cell-standardized lift | **1.435** | **[1.305; 1.574]** |
| Hit-rate delta | **+13.72 pp** | **[+9.62; +17.65]** |
| Forward reference-rate delta | **+47.69 bps** | **[+32.77; +64.02]** |
| Symmetric `±h` delta | +26.80 bps | [+16.86; +37.58] |
| Regret improvement | +35.39 bps | [+20.75; +50.28] |

Эти intervals условны на уже выбранной модели/policy и всего 36 month blocks.
`symmetric ±h` включает прошлое и не является forward business outcome.

### Per-corridor simultaneous screen

| Corridor | N signals | Lift, ordinary CI95 | Forward Δ, bps CI95 | Min weeks with 1–2 | Gate |
|---|---:|---:|---:|---:|---|
| AMD | 139 | 1.333 [1.178; 1.487] | +37.02 [21.82; 52.50] | 0.735 | cadence fail |
| KGS | 128 | 1.387 [1.217; 1.565] | +54.15 [36.69; 72.11] | 0.755 | cadence fail |
| KZT | 133 | 1.588 [1.321; 1.869] | +42.04 [22.65; 63.51] | 0.667 | cadence fail |
| TJS | 124 | 1.469 [1.281; 1.657] | +56.30 [36.99; 76.25] | 0.735 | cadence fail |
| UZS | 139 | 1.430 [1.264; 1.608] | +50.12 [32.60; 67.51] | 0.833 | cadence fail |

Каждый corridor проходит retrospective quality intervals, включая
Bonferroni-style simultaneous lower screens, но **каждый проваливает** заранее
заданный minimum weekly fulfillment `0.90`. Поэтому поле
`passes_corridor_candidate_policy_uncertainty_gate=true` нельзя читать без
`audit_scope=diagnostic_track_only`: family-level promotion всё равно false.

Synthetic all-five portfolio содержит 153 signal dates, lift 1.332 и forward
delta +37.24 bps. Это верхняя оценка market stream, не клиентский CRM estimate:
client eligibility, другие кампании и delivery не наблюдаются.

## 6. Quality–cadence frontier

Отдельный post-hoc frontier держит одну модель и меняет только frozen rolling
score quantile:

| Quantile | Lift | Min weekly fulfillment | Вывод |
|---:|---:|---:|---|
| 0.05 | 1.096 | 0.959 | cadence проходит, quality нет |
| 0.10 | 1.147 | 0.918 | cadence проходит, quality нет |
| 0.20 | 1.115 | 0.918 | cadence проходит, quality нет |
| 0.30 | 1.259 | 0.896 | оба чуть не проходят |
| 0.40 | 1.377 | 0.837 | quality проходит, cadence нет |
| 0.50 | 1.421 | 0.755 | quality проходит, cadence нет |

Проверенной точки с `lift≥1.3` и fulfillment `≥0.90` нет. Frontier — point
estimate diagnostic без inferential claim, но он показывает полезную
операционную развилку. `Force-fill` запрещён; нулевой сигнал в тихую неделю —
правильное решение. Полный файл:
[`cadence_quality_frontier.csv`](cadence_quality_frontier.csv), готовый визуал
для защиты — [`cadence_quality_frontier.svg`](cadence_quality_frontier.svg).

Важно: текущая cadence исключает partial boundary/unresolved-label weeks.
Production denominator обязан включать все недели nominal exposure, потому что
для воспроизведения факта сигнала future label не нужен.

## 7. Timing sensitivity — главный blocker

| MOEX calendar lag | Brier | Skill vs prior-year | Candidate lift | Candidate forward Δ |
|---:|---:|---:|---:|---:|
| 0 | 0.212729 | +3.10% | 1.116 | +21.04 bps |
| **1** | **0.190617** | **+13.17%** | **1.435** | **+47.69 bps** |
| **2** | **0.225511** | **−2.73%** | **1.065** | **+9.37 bps** |
| 3 | 0.217025 | +1.14% | 1.004 | +5.93 bps |
| 5 | 0.229994 | −4.77% | 1.083 | +5.14 bps |

Один дополнительный день availability уничтожает основной результат. Это не
«небольшая sensitivity», а causal-to-viability data contract. Нужны
`decision_ts`, exchange session/cutoff, actual `published_at`, timezone и
immutable snapshots. Для UZS новая публикация около 17:30 Ташкента совпадает
примерно с 15:30 MSK и создаёт cutoff race; для TJS timestamp не доказан, значит
до верификации безопасен T+1. Для CBR 15:30 — cutoff входных cross-rate данных,
а не publication time; точное время публикации не регламентировано. EOD MOEX
close формируется после окончания сессии и недоступен к cutoff, поэтому лаг T+1 допустим только для заранее
определённого next-day/morning decision clock, но не доказывает 15:30 nowcast.

Официальные timing references: [CBR publication clarification](https://www.cbr.ru/Reception/TopicalMessage/Page/2661),
[MOEX fixing clock](https://www.moex.com/en/fixing/) и
[MOEX FX trading schedule](https://www.moex.com/files/4fwbde0jzcmq517abt912ecj4b).
Текущий CNYFIXME — около 12:30, тогда как CNYRUB_TOM EOD close формируется
после сессии до 19:00 и не может быть финальным в 15:30; публичное fixing value
может иметь 12-hour delay. Более близкий к cutoff биржевой proxy — рассчитанный
MOEX `CNYRUB_WAP0` за 10:00–15:30, но он не является доказанным фактическим
CBR anchor и окончание окна не гарантирует публикацию к decision time:
[MOEX notice](https://www.moex.com/n65680).
Историческая методика fixing менялась: старый PDF задавал окно
12:25:01–12:30, текущая страница — 12:15–12:30. Поэтому replay должен
версионировать методику и отмечать даты изменений.

Однако отдельный unauthenticated ISS probe от 2026-09-04 не решил проблему:
при заявленной истории `WAPS` из 700 строк все `WAPRICE` оказались null и все
`NUMTRADES` равны нулю. То есть открытый endpoint даёт календарный каркас, но
не значения WAP. Результат сохранён в
[`moex_wap0_access_probe_2026-09-04.json`](moex_wap0_access_probe_2026-09-04.json).
До получения лицензированного intraday feed или законной реконструкции из
сделок CNY-блок остаётся diagnostic-only.

## 8. Calibration, dependence и 2026

- Monotone Platt улучшает Brier на validation во всех 15 CNY cells, но OOT ECE
  и slope заметно меняются по годам. В уже просмотренном 2026 calibrated ECE
  хуже raw. Score можно использовать для ranking, но нельзя называть
  калиброванной вероятностью без inner cross-fit и frozen OOT calibration gate.
- По 1 636 датам, где строится h=5 target, средняя pairwise target correlation
  пяти коридоров `0.705`; все пять согласны в `73.72%` дат. Общий RUB shock
  нельзя размножать на пять независимых доказательств.
- 2026 diagnostic снова выглядит положительно по Brier/lift, но это 156 дат и
  лишь 8 month blocks, период неполон и уже многократно просмотрен. Он не
  является confirmation или holdout.

## 9. Данные и интерпретация feature blocks

Собраны 31 source family и 76 provenance/data artifacts. Полная таблица с
частотой, coverage, lag, лицензией и ролью —
[`SOURCE_CATALOG_2026-09-04.md`](SOURCE_CATALOG_2026-09-04.md), машинный
provenance — [`data_manifest.json`](data_manifest.json).

| Блок | Результат | Дальнейшая роль |
|---|---|---|
| CNY/RUB spot–fixing/basis | единственный сильный fast block | лицензированный shadow после точного clock |
| Direct MOEX corridor FX | систематического incremental gain нет; TJS лишь 17 active rows | liquidity/basis guard, не общий predictor |
| KASE/NBK market/rates | current public history коротка/агрегирована | KZT-specific licensed experiment |
| Brent, commodities, CISS, Treasury, H.10 | broad stack не улучшил fast target | regime stratification, stress tests |
| CPI, reserves, current account, surveys | slow/current-vintage | historical eval с lag или regime only |
| Alpha executable quote | отсутствует | P0: настоящий outcome/utility |

Открытая статистика не означает автоматически открытые биржевые права.
MOEX/KASE raw и derived evidence нельзя распространять до legal/data-owner
sign-off; Bloomberg требует enterprise license. Email registration проблему не
решает. См. [`DISTRIBUTION_AND_LICENSE_README.md`](DISTRIBUTION_AND_LICENSE_README.md).

## 10. Рекомендуемые модели и эксперименты

Приоритет:

1. deterministic contemporaneous cross/no-change/climatology/rule baseline;
2. regularized logit/GAM/EBM на `RUB anchor + local denominator + residual`;
3. shallow HGB challenger; stump и depth-2 считать эквивалентными, пока их
   paired delta interval включает ноль;
4. hierarchical partial pooling по corridor residual;
5. discrete-time hazard для согласованных horizons;
6. quantile/regret model и abstention по expected utility;
7. state-space/Kalman joint nowcast RUB-anchor/local cross;
8. MIDAS для macro только с vintages/release timestamps;
9. dependent/sequential conformal uncertainty после prospective shadow.

Следующие обязательные experiments:

- residual target после нормализации Alpha quote:
  `log(RUB_debited/LCY_received)−log(reference)` для fixed notional, с fee,
  direction, route/provider и TTL;
- exact cutoff replay по каждому локальному центробанку;
- KASE USDKZT/RUBKZT/CNYKZT + swaps/TONIA + National Fund announcements;
- timestamped MOEX mid/WAP/order book `≤decision_ts`, realtime CNYFIXME и
  roll-adjusted CNY/RUB/USD/RUB futures displacement при наличии прав;
- на одной выборке разложить движение на `fix 12:30 → WAP0/mid 15:30` и
  `15:30 → EOD close`; lift только во второй части не deployable в 15:30;
- добавить официальный implied CNY/RUB overnight swap rate ЦБ только с D+1
  availability как funding-stress/fallback feature;
- CFETS/PBoC USD/CNY fixing surprise с announcement timestamp;
- по одному macro block за раз с nested/PIT ablation;
- inner temporal cross-fit calibration/threshold;
- fold-stratified stationary/month bootstrap с refit всей selection pipeline;
- multiplicity across model × horizon × policy, а не только displayed h=5 rows;
- adversarial missing/stale-feed, holiday, devaluation и methodology-change tests;
- full-exposure cadence и client-level CRM simulator.

Подробная очередь: [`MODEL_EXPERIMENT_ROADMAP.md`](MODEL_EXPERIMENT_ROADMAP.md).

## 11. Недочёты, оставшиеся намеренно видимыми

1. Истории — latest-revised snapshots, не полные real-time vintages.
2. `published_at` отсутствует у target и части внешних рядов; availability —
   консервативное допущение, а не доказанный SLA.
3. Threshold и Platt используют один validation period; test не загрязнён, но
   confirmatory selection uncertainty недооценена.
4. Policy bootstrap ресэмплирует 36 months без стратификации по outer fold и
   условен на выбранной policy; нужен pipeline refit/bootstrap.
5. Calibration gate не реализован, per-signal model explanation unavailable.
6. Full nominal-exposure cadence ещё не посчитана.
7. Direct TJS market series практически пуст; локальные denominators имеют
   неодинаковые cutoff и publication mechanics.
8. `_SUCCESS` отсутствует у прерванного `results-final-20260904-v1/`; любые его
   частичные файлы запрещено использовать.
9. Sentinel `development_corridor_candidate_uncertainty_h5.csv` говорит
   `no_diagnostic_track_available`, хотя точный смысл — «нет gate-eligible
   candidate»; реальные diagnostic rows лежат в отдельном файле. Это cosmetic
   wording issue, не изменение результата.
10. `CLOSE>0` использовался как признак наличия market row, но не доказывает
    ликвидность; нужны turnover/spread/depth/last-trade time. Допустимый в
    research carry-forward 7–45 дней слишком длинный для production-сигнала:
    там нужен максимум один рабочий день и явные `age/stale` guards.

## 12. Итог

Новый контур опроверг идею, что максимальный набор внешних факторов или более
сложная модель автоматически улучшат задачу. Он нашёл одну сильную компактную
гипотезу, но одновременно показал, что она зависит от одного дня source timing
и не совместима с требуемой недельной равномерностью. Поэтому рациональное
решение — deterministic mechanism + residual model + abstention, затем новый
immutable shadow. До этого момента production verdict остаётся **NO-GO**.
