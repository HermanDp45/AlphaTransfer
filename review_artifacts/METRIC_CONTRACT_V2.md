# AlphaTransfer: metric contract V2

> **Поправка от 2026-09-04.** Формальные определения truth и clocks ниже
> сохраняются, но иллюстративный `logit_full, h=20` больше не является текущим
> research candidate. Канонический h=5-аудит находится в
> [`quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md`](quant_macro/QUANT_RESEARCH_REPORT_2026-09-04.md).
> Любое продвижение теперь fail-closed при отсутствии immutable
> `published_at`/vintage, production data rights или per-signal explanation.
> Частота для policy должна считаться на полном nominal exposure span; недели
> с unresolved tail/boundary не исчезают из операционного denominator.

Статус: proposal для заморозки **до** следующего untouched-test. Контракт отделяет
правдивость сигнала, экономическую ценность момента, качество вероятности,
пригодность потока для CRM и причинный эффект будущего пилота.

Модель не обещает предсказать торговый курс. Она ранжирует объяснимые поводы для
фактологичной коммуникации только о прошлом и настоящем. `WINDOW_CLOSING` —
внутреннее имя сценария; клиентский текст сообщает уже наблюдённый отскок и не
говорит прямо или намёком, что окно «может закрыться».

## 1. Временная ось и курс

Для коридора `c`:

- `T_t` — timestamp принятия решения;
- `S_t` — последний опубликованный к `T_t` официальный курс, рублей за одну
  единицу валюты получателя; меньше — лучше отправителю;
- `Q_t(c,n) = RUB_debited / LCY_received` — исполнимый all-in курс Alpha для
  коридора `c` и fixed notional `n`, после fee; меньше — лучше. Направление,
  notional tier, route/provider и quote TTL обязательны — UI `LCY/RUB` имеет
  обратный знак и не сравнивается с `S_t` напрямую;
- `available_at = max(published_at, first_observed_at, access_ready_at)` по всем
  применимым часам источника, и `available_at ≤ T_t`; `effective_date` или
  окончание расчётного окна не заменяет фактическую доступность. Если
  обязательный timestamp неизвестен, same-day feature fail-closed;
- номинал источника (`1/10/100/10 000` единиц) нормализован до `RUB/1 LCY`;
- `h ∈ {1, 3, 5, 10, 20}`; единица всегда указывается в имени метрики.

В текущем exploratory experiment `h_cbr_row` — число последовательных CBR
effective-date records, используемых как proxy новых публикаций target-ряда;
настоящего `published_at` в snapshot нет. Повтор last-known курса на выходном
или празднике не становится новой CBR-строкой. Это удобный reproducibility
proxy, но не доказанный `h_pub` и не требование кейса.
Для следующего confirmatory-test primary clock/horizon ещё должен быть выбран
до открытия labels; §2.3 предлагает календарный `h_cal=5` как default,
соответствующий формулировке «дней» и пяти календарным дням решения. Все
`h={1,3,5,10,20}` на обоих clocks публикуются как sensitivity. Если продукт
принимает решение по свежей MOEX-сессии, а target ЦБ не обновился, это отдельный
`decision_clock`: baseline и cooldown обязаны использовать тот же clock и не
создавать несколько равнозначных решений на одной публикации ЦБ.

Команда выбирает primary evaluation target — официальный нормализованный ряд ЦБ
РФ — ради воспроизводимости, длины истории и единой шкалы пяти коридоров;
постановка не навязывает этот источник. NBK, MOEX, OXR и другие ряды —
source/clock robustness. Исполнимая котировка Alpha не является доступным
offline ML-target хакатона, но её snapshot обязателен в измерении будущего
пилота.

## 2. Truth-функции

### 2.1. `NOW_FAVORABLE`

```text
best_wait_gain_bps(t,h) =
  10 000 × max(0, S_t − min(S_{t+1}, …, S_{t+h})) / S_t

Y_NOW(t,h,δ) = 1[best_wait_gain_bps(t,h) ≤ δ]
```

Для NOW эта truth следует Q&A. Primary `δ=0`: за следующие `h` наблюдений не появился даже немного лучший
курс. `δ ∈ {10, 25, 35, 50, 75, 100}` bps — только заранее объявленная
sensitivity. Локальный минимум в симметричном окне `±h` может быть auxiliary
label, но не является truth из Q&A.

### 2.2. `WINDOW_CLOSING`

```text
future_rise_bps(t,h) =
  10 000 × max_k=1..h ((S_{t+k} − S_t) / S_t)

Y_CLOSING(t,h,δ) = 1[future_rise_bps(t,h) > δ]
```

Кейс требует, чтобы курс «действительно вырос», но точную форму функции и порог
не задаёт. `max-within-h` и строгое сравнение `>δ` здесь — командное допущение.
`δ=0` остаётся формальным benchmark; он слишком легко засчитывает микродвижение
и не допускается к продуктовому GO. До confirmatory-test `δ*` должен быть
зафиксирован из минимально значимого изменения исполнимой выплаты/маржи.
Диагностический `35 bps`, взятый из текущего trigger-config, post-selected и не
является таким обоснованием. Уже случившийся rebound допустим как **признак
trigger**, но никогда как future truth.

Robustness-вариант:

```text
endpoint_rise_bps(t,h) = 10 000 × (S_{t+h} − S_t) / S_t
Y_CLOSING_ENDPOINT(t,h,δ) = 1[endpoint_rise_bps(t,h) > δ]
```

Endpoint публикуется отдельно и не подменяет primary max-within-horizon.

### 2.3. Что ещё нужно заморозить до confirmatory test

Текущая версия — **draft contract**, а не фиктивно «замороженный» claim:

- Scenario предлагается зафиксировать как `NOW_FAVORABLE`.
- Product owner до открытия новых labels выбирает actionable clock/horizon.
  По смыслу кейса primary должен быть календарным (`h_cal`); publication-row
  horizons остаются reproducibility sensitivity. Если продукт не даст иного
  окна, рабочий pre-registration default — `h_cal=5`, следующие пять
  календарных дней. Пять business sessions — отдельный clock `h_bus=5` и не
  подменяет это определение.
- Команда заранее фиксирует один компактный model/feature contract, threshold
  rule, cooldown-state semantics и business weighting коридоров; никакого
  retuning на test.
- Confirmatory corridor family — все пять коридоров с Holm FWER `0.05`.
- Все остальные horizons, CLOSING, модели и ablations публикуются, но имеют
  robustness/exploratory status.

Сетка `logit_full h_cbr_row=1/20` и прежний `logit_core_vol h_cbr_row=10` уже выбраны
после просмотра OOT и не могут сами определить confirmatory claim. После
письменного выбора параметров этому документу нужна новая версия **до** открытия
новых labels. Любая последующая смена claim начинает новый holdout.

## 3. Eligible universe и random baseline

Для каждого `corridor × fold × h × decision_clock` заранее строится множество
`U(c,f,h)`. Дата входит в него, только если:

- target положителен, нормализован, версия источника зафиксирована;
- все входы имеют доказанный `available_at ≤ T_t`;
- строка не является искусственным повтором, если primary clock — публикации;
- доступны обязательные past-only признаки;
- полностью разрешился future horizon;
- дата находится в untouched test после purge;
- выполнены заранее заданные freshness/data-quality gates.

Random hit rate — prevalence соответствующей truth на всём `U(c,f,h)`. Сигнал
и baseline всегда имеют один коридор, fold, clock, source, horizon, freshness и
правила исключения. Дополнительный matched-baseline по месяцу, weekday и режиму
волатильности полезен как robustness, но не заменяет официальный random-day
baseline.

## 4. Point-in-time и walk-forward

В каждом fold:

```text
train → purged validation → purged untouched test
```

Обязательные свойства:

1. Features на `T_t` используют только записи с `available_at ≤ T_t`.
2. Normalizer, imputer, feature selection и calibrator обучаются только внутри
   train/validation соответствующего fold.
3. Model, features, hyperparameters, `δ`, threshold и policy выбираются без test.
4. Последние `h` target-наблюдений train удаляются, если label касается validation;
   последние `h` validation — если label касается test.
5. Test-строка оценивается только если весь её outcome разрешился в test window.
6. При общей модели горизонтов boundary gap равен `max(h)=20`.
7. Calendar folds синхронны для пяти коридоров.
8. Добавление данных после `as_of=T` не меняет features, selection, score,
   explanation или decision на T.
9. Для изолированного fold policy state на старте test прогревается frozen
   policy по pre-test истории; эти события не входят в метрики. Для
   последовательной deployment-симуляции immutable delivered/suppressed ledger
   переносится между folds, включая purged tail: новая fold-модель не должна
   переписывать фактически принятые старые решения.

## 5. Четыре уровня оценки

Нельзя смешивать метрики разных уровней:

1. **Score** — вероятность/score на каждом eligible-дне.
2. **Raw candidate** — score прошёл threshold и есть объяснимое основание.
3. **Policy signal** — применены conflict, freshness veto, cooldown и cap.
4. **Delivered contact** — online-уровень после общего клиентского CRM-бюджета.

Для raw и policy нужны отдельные таблицы. Suppression reason хранится явно:
`low_score`, `no_explanation`, `stale`, `scenario_conflict`, `cooldown`,
`corridor_cap`, `client_contact_budget`.

Self-check 1–2 сигнала в неделю относится к `policy signal` на коридор. Лимит
1–2 контакта на клиента суммарно — отдельный online-слой.

## 6. Обязательные offline-метрики

### 6.1. Правдивость

```text
hit_rate          = hits / signals
random_hit_rate   = random_hits / eligible_decisions
lift              = hit_rate / random_hit_rate
absolute_delta_pp = 100 × (hit_rate − random_hit_rate)
```

Всегда показываются `hits`, `signals`, `eligible_decisions`, число folds/временных
блоков и 95% CI. Lift без absolute delta и CI не интерпретируется.

При объединении `fold × corridor` cells primary — signal-mix standardization:

```text
p_signal = Σ hits_j / Σ signals_j
p_random_standardized = Σ(signals_j × baseline_hit_rate_j) / Σ signals_j
lift_standardized = p_signal / p_random_standardized
```

Для bps baseline cell means взвешиваются теми же `signals_j`. Это сравнивает
сигнал со случайным днём в тех же cells и не даёт Simpson mix между годами и
коридорами. Рядом обязательно публикуется exposure-pooled sensitivity
`Σbaseline_hits/Σeligible`: смена знака или решения между estimands означает
robustness failure. Для online business summary веса коридоров фиксируются по
ожидаемой eligible-аудитории до раскрытия outcomes.

### 6.2. Официальная выгода `±h`

```text
advantage_bps_pm(t,h) =
  10 000 × (mean(S_{t-h}, …, S_t, …, S_{t+h}) / S_t − 1)
```

Публикуются mean, median, доля `>0` и 95% moving/month-block bootstrap CI. Это
обязательная метрика кейса, но прошлое в окне не доказывает выгоду решения ждать.

### 6.3. Forward economics

```text
advantage_bps_forward(t,h) =
  10 000 × (mean(S_{t+1}, …, S_{t+h}) / S_t − 1)
```

Положительное значение: ожидание в среднем сделало перевод дороже. Отрицательное:
клиенту было выгоднее подождать. Рядом показываются `best_wait_gain_bps`,
`future_rise_bps` и asymmetric regret. Forward economics обязательна как
диагностика решения, но не заменяет официальную `±h` метрику.

Для contest offline-gate фиксируется временный non-inferiority guardrail:
point estimate `advantage_bps_forward ≥ 0`, нижняя граница block-95% CI
`> −25 bps`. `25 bps` — явно командный proxy до появления Alpha quote,
recipient amount и unit economics; для банковского пилота он заменяется
экономически рассчитанным MES до раскрытия результата.

### 6.4. Асимметричная цена ошибки

На всех eligible rows для одного frozen threshold:

```text
weighted_error_3 = (3 × FP + 1 × FN) / eligible_decisions
```

`FP` — policy would send при `Y=0`; `FN` — policy remains silent при `Y=1`.
Вес `FP:FN=3:1` — прозрачное командное допущение из-за дефицитного push-слота;
рядом обязательна sensitivity `1:1`, `2:1`, `5:1`. Threshold выбирается
лексикографически: сначала попадает в cadence-band `[1.0, 2.0]` policy
signals/corridor/week, затем минимизирует `weighted_error_3` на validation;
tie-break — меньшая частота. Для пилота вес заменяется отношением ожидаемой
стоимости лишнего контакта к упущенному инкрементальному revenue.

### 6.5. Вероятности и калибровка

На **всех** eligible OOT-днях отдельно для NOW/CLOSING:

- Brier score и primary Brier Skill относительно pre-test
  train/validation prevalence, доступной на cutoff;
- diagnostic Brier Skill относительно same-test-fold prevalence-oracle,
  недоступного в production; обе базы именуются явно и не смешиваются;
- log loss;
- calibration intercept/slope;
- reliability diagram и число точек в bins;
- ECE с оговоркой о чувствительности к binning;
- PR-AUC как ranking-метрика редкого класса.

Class-weighted sigmoid без post-hoc calibration и эвристическая strength не
называются probability/confidence.

### 6.6. Частота и кучность

На полном test-span, включая нулевые folds и недели:

- raw candidates/week и policy signals/week;
- silent-week share;
- median, p90 и max gap, включая границы периода;
- gap CV и доля интервалов `≤3` дней;
- доля событий в самом активном fold/месяце;
- counts по каждому fold и число активных folds.

### 6.7. Цена ожидания fast→slow

Для одного заранее определённого episode:

```text
wait_cost_bps = 10 000 × (S_slow − S_fast) / S_fast
```

Нужны `episode_id`, `parent_candidate_id`, `Δsessions`, `Δcalendar_days`, цена
ожидания, изменение hit/lift и число потерянных сигналов. `S_{t+1}` и `S_{t+3}`
после одного события — не сравнение fast/slow индикаторов. По умолчанию один
CLOSING на episode, чтобы не расходовать CRM-бюджет на повторы одной волны.

## 7. Устойчивость и ablation

Обязательные разрезы:

- пять целевых коридоров и все пять `h`;
- каждый OOT fold/год и worst fold;
- primary full history и заранее объявленные regime sensitivities;
- expanding против rolling 2–3 года;
- сетка `δ`, официальный source и source robustness;
- rules-only → compact logistic → один challenger;
- price-only / +calendar / +cross-source / +liquidity ablation;
- cutoff/prefix invariance и leave-one-corridor/year/source-out.

Коридоры зависимы и не дают пять независимых повторов. Source robustness на KZT
не заменяет переносимость на другие коридоры.

## 8. Неопределённость и multiple testing

- CI строятся блоками, сохраняющими временную зависимость.
- Lift и absolute delta считаются на одном bootstrap resample.
- Единственная confirmatory family следующего holdout: пять corridor-тестов для
  заранее записанного NOW scenario, business horizon и model contract; Holm
  FWER `0.05`. Пока поля из §2.3 не утверждены, confirmatory test не начат.
- Все пять горизонтов всё равно публикуются. Остальные `scenario×clock×h`,
  ablations и models — exploratory; FDR допустим только для их приоритизации,
  не для `GO_PILOT`.
- Pooled panel estimate с month/date blocks — secondary: он не может скрыть
  провал corridor-level transferability.
- Если победитель выбран после просмотра test-матрицы, текущие p-values
  корректируются минимум Holm/Bonferroni и всегда помечаются post-selection;
  это не заменяет новый holdout.
- Model/feature/threshold/horizon selection выполняются внутри nested walk-forward.
- Exploratory модели не повышают статус решения без нового untouched test.
- Широкий CI означает `INCONCLUSIVE`, а не `GO`.

## 9. Decision states

### Hard validity gates

- правильные NOW/CLOSING truth;
- zero look-ahead и purged boundaries;
- единый eligible universe;
- prefix invariance, включая model selection;
- counts-based aggregation;
- полный frequency denominator;
- воспроизводимый `signal(as_of=T)`.

Нарушение любого пункта даёт `INVALID` независимо от lift.

### Signal GO

Предлагаемый gate, который надо заморозить до final test:

- OOT lift point estimate `≥1.3` и Holm-adjusted 95% lower bound выше `1.0`
  минимум в двух из пяти коридоров;
- pooled month/date-block lower bound lift выше `1.0`;
- 95% CI absolute delta и официального `advantage_bps_pm` выше `0`;
- forward point estimate `≥0 bps`, block-CI lower bound `>−25 bps`;
- mean cadence в диапазоне `[1.0, 2.0]` policy signals/corridor/week и минимум
  `90%` full-exposure weeks имеют 1–2 сигнала в каждом заявляемом
  `corridor × fold`. Denominator включает zero-signal, boundary и
  unresolved-label недели; `signals=0` означает undefined hit/lift и failure
  coverage, а не нулевую точность;
- минимум три активных OOT folds, один fold создаёт не более 50% событий;
- simulated block-power не ниже 80% для альтернативы `lift=1.3` при
  family-wise `α=.05`; fallback до такой симуляции — не менее 36 OOT
  month-blocks и 20 signals в каждом заявляемом коридоре;
- Brier Skill положителен, если probability участвует в policy;
- каждый сигнал объясним и воспроизводим на cutoff.

Если `random_hit_rate=0`, lift undefined: показываются counts/absolute delta и
заранее заданно укрупняется test-period; бесконечный lift не даёт GO. CLOSING
не получает GO, пока экономически осмысленный `δ*` не зафиксирован до test.

Fail validity → `INVALID`; недостаточно выборки/стабильности → `INCONCLUSIVE`;
не пройдены predictive/economic gates → `NO_SIGNAL`; прошли offline gates →
`GO_PILOT`, но ещё не rollout.

## 10. Decision table

| Level | Corridor | Scenario | Clock / h / δ | Hits/N | Eligible | Lift [95% CI] | Δ pp [95% CI] | Signal `±h` bps; incremental Δ | Signal forward bps; incremental Δ | /week; silent | Active folds | Brier Skill prior/oracle | Worst fold | Status / причина |
|---|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|
| Policy dates | RUB/KZT | NOW | MOEX-row / 5 / 0 | 4/7 | 443 rows | 2.26 [0.73; 4.47] | +31.86 [−6.13; +77.65] | +81.96 [−3.20; +201.04]; Δ n/a | −0.36 [−150.16; +180.71]; Δ n/a | 0.077; 92.4% | 2/7 | n/a | n/a | **DIAGNOSTIC/INVALID**: не pub/day clock, даты выбраны дефектным evaluator, tiny N |
| Policy dates | RUB/KZT | CLOSING | MOEX-row / 5 / >35 | 9/10 | 443 rows | 1.42 [1.14; 1.72] | +26.57 [+8.80; +41.99] | +19.57 [−43.33; +73.74]; Δ n/a | +144.49 [+7.19; +250.54]; Δ n/a | 0.110; 90.2% | 2/7 | n/a | n/a | **DIAGNOSTIC/INVALID**: не pub/day clock, `δ` post hoc, selection invalid |
| Policy, cell-standardized | 5 corridors | NOW | CBR-row proxy / 20 / 0 | 156/774 | 4115 rows | 1.095 [0.945; 1.266] | +1.75 [−1.03; +4.81] | +26.65 [+0.38; +53.70]; Δ +16.46 [+6.57; +27.82] | +35.23 [−25.59; +97.36]; Δ +13.84 [+0.57; +28.28] | 0.916; n/a | 4/4 | −17.4% / −26.6% | 0.886 | **EXPLORATORY/POST-SCREEN**: no winner; adjusted lift/forward-Δ p=1, lift<1.3 |
| Policy, exposure-pooled sensitivity | 5 corridors | NOW | CBR-row proxy / 20 / 0 | 156/774 | 4115 rows | 1.015 [0.873; 1.178] | +0.30 [−2.48; +3.48] | +26.65 [+0.38; +53.70]; Δ +16.02 [+4.01; +29.33] | +35.23 [−25.59; +97.36]; Δ **−3.80** [−25.22; +17.44] | 0.916; n/a | 4/4 | −17.4% / −26.6% | n/a | **ROBUSTNESS FAIL**: incremental forward/regret меняют знак относительно standardization |

Для `CBR-row proxy / 20` incremental regret improvement также меняет знак:
`+6.43 bps` при cell-standardization против `−4.24 bps` exposure-pooled.

Текущие числа `edelkin_test` нельзя вставлять сюда как GO: они получены до
исправления truth, boundary purge, point-in-time и aggregation.

## 11. Online pilot

Offline GO разрешает только пилот.

- Unit — клиент; assignment постоянный.
- Выбранная аудитория — существующие regular/flexible senders с ранее
  использованным коридором/получателем. Механизм — repeat/share-of-wallet, а не
  привлечение новых клиентов. Это допущение до банковской сегментации.
- Treatment получает BAU CRM scheduler + frozen FX-candidate; holdout получает
  тот же BAU scheduler без FX-candidate, включая next-best campaign. Это
  измеряет opportunity cost, а не signal-vs-forced-silence.
- Анализ intention-to-treat, а не только среди delivered/opened.
- Sample size считается по фактической trigger frequency и заранее выбранному MDE.

Единственный primary — **incremental net contribution margin per ex-ante
targetable randomized client** до 90 дней после последнего разрешённого
контакта. 7/30/60 дней — secondary trajectory. Mechanism metric — incremental
completed net transfer volume; все недоставленные/неоткрывшие остаются в ITT.

Secondary: delivery/open, переход в форму, quote generated, completed transfer
за 24/72 часа, повторность, recipient amount, чек, time-to-transfer. Рост CTR или
перенос уже запланированного платежа без прироста кумулятивного объёма/маржи не
является успехом.

Guardrails: opt-out/жалобы, contact load, отмены после repricing, stale signal,
route errors, fraud/compliance incidents, снижение net volume/margin и вытеснение
других полезных коммуникаций.

### Данные, дизайн и срок

От банка до старта нужны: persistent assignment; corridor eligibility;
shadow-policy `candidate/suppression`; snapshots `quote_at_signal/open/confirm`;
send/delivery/open/deeplink/form/confirm ledger; completed transfers, RUB amount,
revenue и contribution margin; остальные контакты; opt-out/жалобы; route/fraud
statuses. Идентификаторы — технические токены в разрешённом банковском контуре,
не в offline submission.

Power считается по margin baseline/variance и business MES до запуска. Общие
рыночные триггеры создают cluster dependence; market date нельзя размножать на
пять коридоров. Stop collection требует одновременно planned client
information, не менее 50 заранее определённых non-overlapping episode blocks и
90-day outcome maturity. При текущем диагностическом темпе это около 30–50
активных недель плюс maturity; 8–10 недель годятся только для operational ramp,
а не финального market-regime вывода. Подробный causal contract и decision
states: [`PILOT_METRIC_CONTRACT.md`](PILOT_METRIC_CONTRACT.md).
