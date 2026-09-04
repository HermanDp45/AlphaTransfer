# AlphaTransfer: model & experiment roadmap

Дата: **2026-09-04**. Цель — не угадать красивую модель, а принять решение
`send now / stay silent / wait` при жёстком CRM-бюджете и измеримой пользе.

## Целевая архитектура для shadow-валидации

```text
available local USD/LCY + joint RUB/local-cross nowcast + Alpha executable quote
              │
              ▼
deterministic cross-rate engine ──► expected official/reference rate
              │
              ▼
residual / uncertainty model ─────► P(NOW hit), quantiles, expected regret
              │
              ▼
utility + abstention policy ──────► corridor candidate + reason ledger
              │
              ▼
client eligibility / CRM ranker ──► delivered contact or suppression reason
```

Почему так: ручной all-five snapshot и KZT-история на выбранном alignment после
конца 2024 года почти точно совместимы **contemporaneously** с одним RUB-якорем
и наблюдаемыми proxy `USD/LCY`; all-five historical decomposition и фактический
denominator/source ЦБ ещё не установлены.
Поэтому pooled model на пяти почти одинаковых target-движениях переоценивает
независимость данных. Но official-reference NOW-h зависит от будущего пути
обоих cross-компонентов: deterministic engine не является forecast. Alpha
residual нужен отдельно для прогноза исполнимой котировки и business utility.

## Frozen baselines и challengers

| Приоритет | Модель | Зачем | Условие сохранения |
|---|---|---|---|
| B0 | prior-year/rolling climatology | proper-scoring baseline | обязателен всегда |
| B1 | deterministic cross-rate + rule bands | сильный структурный benchmark | не хуже ML по utility/cadence |
| B2 | transparent spline/logit/GAM | nonlinear CNY spot-minus-fixing displacement без лишней сложности | OOT Brier + policy gain |
| C1 | HGB depth=2 | минимально лучший point Brier (`0.190617`) в текущей ретроспективе | только frozen challenger; не production |
| C2 | HGB stump depth=1 | более простой практически эквивалентный вариант (`0.190776`) | предпочесть по парсимонии, если новый OOT не докажет gain depth=2 |
| C3 | hierarchical/partial-pooling GLM | общий RUB factor + corridor residual | лучше worst-corridor и calibration |
| C4 | discrete-time hazard | вероятность, что минимум наступит в 1/3/5/10/20 | согласованные горизонты, без пяти отдельных моделей |
| C5 | quantile/regret model | цена ожидания и abstention | улучшает decision regret, не только hit-rate |
| C6 | state-space/Kalman residual | joint online nowcast RUB-якоря/local cross | latency/stability лучше static model |
| S1 | CatBoost или EBM shadow | один мощный нелинейный sanity-check | не проводить model zoo |
| S2 | MIDAS mixed-frequency | строгая работа с monthly/weekly macro | только с vintages/release timestamps |

Не приоритетны сейчас: deep nets, Transformers, RL, HMM zoo, новости/LLM как
направленный predictor. Истории мало, effective sample size определяется
датами/эпизодами, а не `5 × rows`; основное ограничение — policy cadence,
executable quote и causal product value.

## Эксперименты, которые уже дали решение

| Эксперимент | Измеримый вывод | Решение |
|---|---|---|
| CBR cross compatibility | KZT 412/413 exact на выбранном alignment; live manual snapshot 5/5 | строить deterministic engine, не называть alpha; расследовать exception |
| Feature-family ablation 2023–2025 | CNY spot-minus-fixing displacement (`basis` в feature ID) — единственный сильный короткий блок; full stacks хуже | frozen compact challenger |
| Public Kazakhstan official stack | не улучшает base | не добавлять в fast model |
| Slow macro blocks | не дают надёжного incremental OOT gain | только regime/stratification |
| HGB depth 1 vs depth 2 | доказанного выигрыша глубины нет | не выбирать champion по этой ретроспективе; оба только challengers |
| Cadence-quality frontier | lift ≥1.3 и ≥90% недель одновременно не достигнуты | не force-fill; улучшить utility либо согласовать cadence |
| 2026 diagnostics | нестабильная calibration/drift | не считать holdout или confirmation |

Точные финальные значения и интервалы находятся в
[`QUANT_RESEARCH_REPORT_2026-09-04.md`](QUANT_RESEARCH_REPORT_2026-09-04.md) и
`results-final-20260904-v2/`. Каталог `results-final-20260904-v1/` —
намеренно сохранённый незавершённый run без `_SUCCESS.json`; использовать его
числа нельзя.

## Правильный evaluation protocol

1. Единица split и inference — календарная дата/market episode; пять
   коридоров одной даты остаются в одном block.
2. На каждом fold: `train → purged validation → purged test`; purge не меньше
   максимального forward horizon.
3. Imputation, scaling, feature selection, calibration, threshold и cooldown
   state обучаются/настраиваются только на прошлом.
4. Сырые `observed_at`, `available_at`, source hash и vintage фиксируются.
5. Основной development период — 2023–2025. 2026 уже просмотрен и является
   diagnostic. Новый confirmatory период начинается только после frozen spec.
6. Model-family screening: paired loss delta по общей OOT-панели + month/block
   bootstrap/sign-flip; Holm FWER по просмотренным семействам.
7. Policy inference: block-bootstrap по месяцам/эпизодам, одновременный
   corridor screen, концентрация по folds и gap/cadence diagnostics.
8. Любой historical-only/latest-vintage блок либо получает честный release lag,
   либо используется только для stratified eval.

Текущий research pipeline выполняет snapshot replay и не использует будущую
календарную дату в as-of join, но это ещё не доказательство исторической PIT:
intraday timezone/published timestamps и immutable vintages отсутствуют у части
источников. Поэтому `historical_point_in_time_verified=false` и
`production_data_rights_verified=false` должны оставаться fail-closed.

## Метрики зрелого offline-контракта

### Score layer

- Brier и log loss против prior-year climatology;
- calibration intercept/slope, ECE/reliability bins;
- discrimination как вспомогательная, не основная метрика;
- paired loss delta с block CI, а не случайный row-bootstrap.

### Decision layer

- `lift = signal hit-rate / same-cell random-day hit-rate`;
- абсолютная разница hit-rate, pp;
- обязательная выгода относительно `±h`, bps;
- forward advantage и expected regret, bps;
- `signals`, `hits`, eligible denominator, folds, corridors;
- 95% CI; 99%/simultaneous screen для пяти коридоров;
- все `h ∈ {1,3,5,10,20}`; primary horizon фиксируется заранее.

### Policy/operations layer

- candidate 1–2/коридор/неделю как self-check;
- доля полных недель с 1–2, silent-week share, median/p90/max gap;
- top-fold share, min-fold lift, policy Jaccard;
- отдельные причины suppression: low score, no explanation, stale,
  scenario conflict, corridor cooldown/cap, portfolio rank/shared cooldown,
  client CRM budget.

`1–2/коридор/неделю` не доказывает `1–2 контакта/клиент/неделю`: для второго
нужны клиентские eligibility и общий CRM ledger.

Performance denominator содержит только полностью разрешившиеся labels.
Операционная cadence должна дополнительно считаться на полном nominal exposure
span, включая unresolved tail, boundary/zero-signal weeks; текущий research
ledger этого production-утверждения не доказывает.

## Следующие измеримые работы

### P0 — до shadow

1. Добавить executable Alpha quote и определить utility в рублях.
2. Заморозить cutoff, primary horizon/clock, model, feature list, calibration,
   threshold, cooldown и tie-break до новых labels.
3. Реализовать per-signal explanation и полный suppression ledger.
4. Выбрать legal feed для MOEX/KASE либо public-only fallback.
5. Начать prospective immutable shadow; не переобучать по его outcomes.

### P1 — experiments

1. **Mechanism residual:** сначала привести UI quote к одной стороне и номиналу:
   `r_alpha_allin(t,c,n)=RUB_debited/LCY_received` для fixed notional `n`, затем
   прогнозировать `log(r_alpha_allin)−log(r_reference)`. Хранить fee,
   send/receive direction, notional tier, route/provider и TTL; иначе residual
   меняет знак и смешивает нелинейную комиссию со spread.
2. **Joint component nowcast:** CNY/RUB spot-minus-fixing displacement, USD
   proxies, local `USD/LCY`, RUSFAR, orderbook/liquidity при наличии лицензии.
   EOD MOEX close формируется после окончания сессии и недоступен к 15:30,
   поэтому clock задаётся до выбора lag.
   Первый timing experiment на общей выборке сравнивает CNYFIXME 12:30,
   `CNYRUB_WAP0` 10:00–15:30, mid/last 15:29:59 и EOD close, но включает WAP0
   в same-day решение только при доказанном
   `available_at=max(published_at, first_observed_at, access_ready_at) ≤ decision_ts`.
   Если lift живёт
   только в движении 15:30→close, он недоступен для 15:30 product decision;
   если в 12:30→15:30 — возможен честный сигнал только после фактической
   публикации cutoff-compatible snapshot.
3. **Local denominator timing:** availability audit по AMD/KGS/KZT/TJS/UZS;
   TJS до доказательства — T+1.
4. **KASE block:** USDKZT/RUBKZT/CNYKZT, swaps/TONIA, National Fund planned
   flows; ablate отдельно от общего stack.
5. **Regime gates:** нефть/global stress/rates/макро — только один блок за раз;
   критерий — incremental OOT utility/calibration, а не feature importance.
6. **Cadence frontier:** utility-оптимизация с abstention и client demand window;
   не ослаблять truth и не заполнять тихие недели ложными сигналами.

### P2 — после первых 8–12 недель operational shadow

Этот рубеж разрешает диагностировать pipeline, но не объявлять confirmatory
winner: promotion ждёт заранее рассчитанного information target по predeclared
non-overlapping decision-date/episode blocks с block-aware inference.

- block conformal/online calibration по corridor и regime;
- hierarchical shrinkage для редких коридоров;
- survival/hazard across horizons;
- adversarial stress: holidays, rate regime change, stale/missing feed,
  abrupt devaluation, methodology change;
- champion/challenger только по frozen sequential review dates.

## Что из современной литературы переносимо, а что нет

- [ECB WP 2151, *Exchange rate forecasting on a napkin*](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2151.en.pdf)
  напоминает, что простой калиброванный structural benchmark может обгонять
  более сложные time-series/panel-модели. Работа относится к monthly flexible
  FX развитых стран, поэтому не доказывает качество здесь, но усиливает
  требование сначала победить deterministic cross/random-walk baseline.
- [MIDAS](https://www.nber.org/papers/w10914) — разумный challenger для
  mixed-frequency macro, если появятся настоящие release vintages. Исходная
  работа прогнозирует volatility на других рынках; это архитектурная идея, не
  внешний claim о наших пяти коридорах.
- [Dependent-data conformal inference](https://arxiv.org/abs/1802.06300) и
  [sequential predictive conformal inference](https://arxiv.org/abs/2212.03463)
  уместны для abstention/uncertainty после накопления prospective shadow.
  Обычный IID split-conformal здесь невалиден из-за общей даты и перекрывающихся
  горизонтов.

Практический вывод литературы совпал с экспериментом: сейчас преимущество даёт
не model zoo, а структурная декомпозиция, правильный clock, компактная
регуляризация и честное измерение неопределённости.

## Promotion gates из frozen shadow в client pilot

Переход из operational shadow в client pilot допустим лишь при одновременном
выполнении:

- положительный Brier skill и **upper** CI paired delta < 0 после multiplicity;
- policy lift point ≥1.3, CI low >1;
- hit delta и `±h` bps CI low >0;
- forward point ≥0 и заранее заданный non-inferiority floor;
- минимум два коридора проходят simultaneous screen;
- cadence/coverage/concentration проходят;
- explanation, lineage, deterministic replay и legal source rights готовы.

Production GO дополнительно требует randomized ITT pilot на собственной
котировке и contribution margin. Ретроспективный backtest этого доказать не
может.
