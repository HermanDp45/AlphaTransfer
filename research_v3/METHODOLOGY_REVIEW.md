# AlphaTransfer v3: независимая методологическая проверка и программа развития

Дата проверки: 2026-09-04. Этот документ проверяет существующее `final_solution`, а не приписывает будущим экспериментам уже полученный эффект. Прочитаны постановка, обе Q&A, `APPROACH.md`, `training/core_experiment.py`, `training/train_and_evaluate.py`, metric/pilot contracts, старый ML-review, quant summary и `product_artifacts/CLIENT_JOURNEY.md`. Научные и внешние утверждения ниже опираются на первичные источники, проверенные через web.

## 1. Главное решение

У текущего кандидата есть содержательный ретроспективный эффект, но его задача — предсказать качество момента по официальному reference — ещё отличается от задачи клиента. Наиболее перспективное развитие: совместно моделировать распределение изменения курса, личное окно допустимого перевода и ущерб от ожидания; выбирать `send / wait / abstain` под общий бюджет контактов. Более сильные модели и длинные train-окна проверять в том же временном протоколе, сохраняя отдельный результат по обязательным метрикам кейса.

Нужно сохранить то, что уже сделано правильно: будущий NOW-target, purge границ, prior-only калибровка, отдельные candidate и portfolio layers, date-block inference, отрицательные ablations и честный статус 2026. Старые `lift=10.57` и `100% CLOSING` не являются базой для сравнения новых моделей: прежний review установил, что там оценивались другие/некорректные события.

### Проверенная численная точка отсчёта

Источник: `final_solution/model_bundle/development_metrics.csv`, development OOT 2023–2025, `h=5` effective CBR rows, corridor candidates после threshold/cooldown.

| Метрика | HGB base | HGB + CNY spot-minus-fixing | Изменение |
|---|---:|---:|---:|
| Brier | 0.218714 | 0.190617 | −12.85% |
| Cell-standardized candidate lift | 1.0890 | 1.4352 | +0.3462 |
| Forward advantage против matched-cell random day, bps | +6.02 | +47.69 | +41.66 bps |
| Symmetric advantage против matched-cell random day, bps | −1.05 | +26.80 | +27.85 bps |
| Mean hindsight regret у сигналов, bps | 103.45 | 80.33 | −23.13 bps |
| Число candidates | 742 | 663 | −79 |

Это эффект **добавления CNY-блока в прежнем исследовании**, не прирост от нового v3-review. Число contacts и экономия реальных пользователей не наблюдались. `signal` в CSV — отдельный синтетический all-five portfolio; смешивать его с `candidate_signal` нельзя.

## 2. Что именно требует кейс и где наши собственные допущения

| Требование | Следствие для v3 |
|---|---|
| Открытые и воспроизводимые FX-данные; минимум пять лет как отправная точка | Public-only результат обязателен отдельно от hypothetically licensed result |
| Базовые momentum, level, reversal, calendar/holiday hypotheses | Нужна полная матрица индикатор × corridor × horizon, включая отрицательные результаты |
| Классический ML и локальные минимумы ±h | Future-only NOW остаётся главным truth по Q&A; symmetric local-min — отдельный auxiliary target/диагностика, не подмена NOW |
| NOW: сегодняшний RUB/LCY не хуже каждого из следующих h; CLOSING: последующий рост | Разные labels, разные eligible sets, разные калибровки и тексты |
| Метрики на h=1/3/5/10/20 | Зафиксировать h=5 для выбора, опубликовать все пять; не выбирать лучший horizon по test |
| Lift — hit rate / random-day hit rate того же corridor и периода | Сохранять matched-cell estimand, дополнить macro-average по cells и одинаковыми weights при сравнении policies |
| Выгода ±h обязательна, forward допустим с обоснованием | Выводить обе; forward нельзя молча объявить исходной метрикой кейса |
| Необычная кучность неприемлема, 1–2/неделю — самопроверка | Отчёт о silent weeks/months, gap и concentration; качество не компенсируется искусственной частотой |
| Общий лимит 1–2 на клиента, конкуренция с CRM | Персональная policy и общий contact ledger; пять коридоров не дают пятикратный бюджет |
| Исторический срез T без будущих данных | Воспроизводить **весь** путь features → model version → policy state → text; простая фильтрация итогового CSV недостаточна как тест causality |
| Бизнес-эффект на хакатоне не измеряется | Синтетический эффект — scenario, не доказанный uplift; будущий pilot с ITT holdout |

Порог `>=90% недель с 1–2 candidates в каждой cell` введён командой, а не Q&A. Не стоит объявлять любое полезное selective решение неработоспособным только из-за него. Сохранить этот показатель для comparability, но отдельно показать риск/выгоду при разных уровнях coverage. Q&A разрешает обоснованное молчание в тихие периоды; длительные концентрации всё равно требуют объяснения.

`CLIENT_JOURNEY.md` честно остаётся каркасом: AS-IS, обзор практик и часть продуктовых развилок помечены `[TBD]`. Нельзя представлять это как уже проведённое наблюдение реального приложения. Звёздочка не заменяет качество основной задачи.

## 3. Что проверка кода установила дополнительно

### 3.1. Train уже длиннее, чем предполагалось

`core_experiment.py:32` задаёт `TRAIN_WINDOW_YEARS=2`; `split_for_year` в районе строки 213 использует два календарных года train и предшествующий test году целый год validation. Например, test 2023 использует train 2020–2021 и validation 2022. Поэтому гипотезу «раньше учились на паре месяцев» к текущему `final_solution` применять нельзя. Нужна честная лестница 0.5/1/2/3/5 лет/expanding при одинаковом test и одинаковой доступной feature-history.

Оценка большей модели на более длинной истории должна отделять три эффекта: количество строк, возраст наблюдений и охват режимов. Длинное окно с забыванием старых наблюдений сравнивать с коротким и с expanding. Большое число строк пяти коридоров не равно большому числу независимых рыночных дней.

### 3.2. Текст про исторически выгодную зону не следует из текущего score

`product.py` выбирает candidate с максимальной вероятностью NOW, но без проверки historical percentile формирует «курс ... в благоприятной зоне по исторической шкале». Вероятность будущего роста может быть высокой и при исторически дорогом текущем курсе.

Независимое соединение сохранённых `hgb_plus_cnyrub_basis` candidates с `build_panel` даёт:

| Predicate, окно 60 строк ЦБ | Из 663 candidates | Доля |
|---|---:|---:|
| В нижних 20% текущего RUB/LCY | 201 | 30.3% |
| Выше исторической медианы | 391 | 59.0% |
| В верхних 20%, то есть исторически дорого | 277 | 41.8% |

Это не опровергает предсказание NOW; это опровергает автоматическое отождествление `score` с исторически благоприятным уровнем. На demo-дате 2025-12-16 `pr60` равен AMD 0.325, KGS 0.308, KZT 0.525, TJS 0.342, UZS 0.192.

Исправление: отдельный factual predicate и numerical evidence для каждого текста: `pr60<=q`, `ret3<0` и серия down-days, измеренный rebound от зафиксированного прошлого минимума. Если не выполняется ни один факт — abstain либо нейтральный текст с точными цифрами. Надпись «не прогнозирует курс» относится к клиентскому сообщению; внутренняя ML-модель действительно прогнозирует будущую метку и не должна описываться иначе.

### 3.3. Inner validation пока повторно используется

`fit_platt_calibrator` обучает Platt на validation и по Brier **на том же наборе** выбирает его против identity; `choose_frequency_threshold` затем на этом же году выбирает policy. Это не прямой test leakage: test остаётся последующим годом. Но gain самой калибровки in-sample оптимистичен, а один кризисный validation-год может нестабильно выбрать и slope, и threshold.

Лучше построить последовательные inner OOF scores: model-fit → calibration-fit → threshold-select. Для малого effective N использовать identity/temperature/Platt с shrinkage, без дорогого поиска; isotonic допускать только при достаточном числе независимых calibration dates. Окончательную калибровку оценивать на outer test, отдельно в selection region. ECE — диагностика с bin sensitivity, а не цель оптимизации.

### 3.4. Доступность задаётся датой-прокси

`train_and_evaluate.py:add_availability` делает `source_date+days`, затем backward asof join по датам, допускающий равенство. Это хороший консервативный каркас, но `available_date` не наблюдённый timestamp. Latest-revised macro snapshot остаётся latest-revised после добавления любого фиксированного лага. Утверждение «исключён любой look-ahead» требует source vintages и intraday timestamps, а не только проверки `available_date<=date`.

Существующая lag ladder полезна: дополнительный день MOEX разрушает большую часть эффекта. Следовательно, проверять `MOEX D−1`, `D−2`, publication cutoffs и source outages нужно **для каждого нового winner**, а не только для старого. Потенциально сильный эффект может оказаться обучением механики official fixing, которая слабо переносится на исполнимые котировки Альфы.

### 3.5. Пять коридоров не являются пятью независимыми подтверждениями

Сохранённый `target_date_dependence_summary_h5.csv`: средняя pairwise correlation labels 0.7053; все пять меток совпадают на 73.7% из 1,636 дат. Q&A не требует независимости коридоров: переносимость — допустимый вывод. Однако стандартные ошибки и effective N должны учитывать общий RUB shock.

Обучение pooled-модели обоснованно; доказательство превосходства использует paired date losses, date/month/episode block bootstrap, fold-specific readout. При горизонте h перекрытие будущих окон добавляет временную зависимость. 663 candidates не равно 663 независимых испытаниям.

### 3.6. Сравнивать policies нужно при одинаковом экономическом ограничении

Текущий random-day baseline корректен как описательный benchmark кейса. Но для сравнения deployable policies дополнительно нужны случайная schedule-policy, fixed weekly day, salary-cycle и простые rules с тем же eligibility, cooldown, budget и exposure. Иначе изменение частоты или выбора corridor меняет популяцию сравнения.

Cell-standardized lift текущей policy использует её собственные signal weights. Для сравнения моделей публиковать также общие predeclared weights: равные corridor×year или ожидаемая смесь клиентов. Чистый uplift качества — paired исходы одного exposure universe; отдельная policy-value метрика может законно выбирать иное подмножество.

### 3.7. Fast/slow pairing пока описательное

`fast_slow_confirmation_audit` ищет следующий slow portfolio-trigger в течение 10 строк в том же fold/corridor; его код честно помечает результат exploratory. Это не идентифицирует одно рыночное движение, допускает использование одной slow-даты несколькими fast triggers и сравнивает поздний hit только в подтвердившейся подвыборке.

Для решения «ждать или сейчас» нужен frozen stopping rule с deadline: отправить по fast, либо ждать до первого slow/до D, либо abstain. Учесть **все** fast starts, включая отсутствие slow, ухудшение курса, истёкший клиентский deadline и censored tail. Ввести episode identity по информации на fast-time; позднее подтверждение не должно задним числом отбирать удобные старты.

### 3.8. Historical preview ограничен сохранёнными predictions

`load_candidates` читает demo CSV или отфильтровывает precomputed OOT файл. Это воспроизводимый historical preview; он не равен загрузке model artifact и независимому replay на усечённом input. Центральный путь пока не содержит сохранённого fitted model для будущей даты. Новый research runner должен сохранять preprocessing/model/calibrator/threshold и selection cutoffs либо доказуемо воспроизводить их с prefix input.

## 4. Постановка, которая лучше соответствует клиентской ценности

Обозначим `R_t` = RUB за единицу валюты получателя; меньше — выгоднее. Клиент имеет уже запланированный перевод суммы `A_i`, earliest date `e_i`, deadline `d_i`, личную цену ожидания и историю контактов. При фиксированном рублёвом бюджете к получателю приходит `A_i/R_t` до учёта комиссий; production заменяет это на наблюдённый all-in recipient amount.

### A. Обязательный контракт кейса

`NOW_h(t)=1{R_t <= min(R_{t+1},...,R_{t+h})}`. При materiality `δ` отдельно считать `1{min_future/R_t−1>=δ}`. Нельзя заменить strict NOW на более лёгкую «в среднем выгодно» и объявить рост lift улучшением прежней метрики.

`CLOSING_h,δ(t)=1{R_{t+h}/R_t−1>=δ}` — один прозрачный terminal endpoint. Альтернатива first-passage через верхнюю границу допустима как отдельный target; нельзя незаметно менять между endpoint и max-path. Past rebound trigger вычислять независимо от future outcome.

`SYMMETRIC_MIN_h` — отдельный local-min auxiliary. `symmetric_bps` кода включает сам день T в среднее 2h+1: это надо сохранять в спецификации, чтобы команды сравнивали одну величину.

### B. Распределение и риск вместо одной бинарной метки

Предсказывать не только `P(NOW_h)`, но и conditional distribution `G_h = R_{t+h}/R_t−1`, minimum future improvement, и regret. Минимальная реализация: quantile HGB/CatBoost либо pooled distributional residuals; более сложная — joint discrete hazard first adverse move. Согласованность `P(NOW_1)>=P(NOW_3)>=...>=P(NOW_20)` должна выполняться. Независимые classifiers этого не гарантируют; monotone projection или hazard head исправляет incoherence.

Headline для клиента: expected recipient advantage и downside, определённые относительно **конкретного** feasible baseline (например, planned salary date). VaR/ES уместны только как дополнительные loss summaries при достаточном числе tail observations; не подменять ими обязательный lift.

### C. Deadline-aware решение

Для контекста `z_i(t)` оценивать полезность отправки сейчас против ожидания:

`U_send = expected timing value × P(actionable and incremental response | z_i,t) − contact cost − fatigue cost − occupied CRM slot value`.

При срочном deadline proactive wait запрещён policy. При отсутствии наблюдённой response-model синтетический `P(response)` — параметр сценария. Для оптимизации достаточно прозрачного динамического правила/finite-horizon DP поверх forecast distribution; RL до появления валидных logged propensities и причинных outcomes не нужен.

Отдельно оптимизировать factual eligibility и financial utility: высокий forecast score не делает ложный факт истинным. При re-open заново проверять факт и executable quote, показывать both timestamps и нейтральный changed-state.

### D. Синтетические клиенты: полезны для policy, не создают финансовую информацию

Если synthetic user features сгенерированы независимо от будущего FX условно на исторических market/calendar данных, они не добавляют знания о будущем FX. Их преимущество может возникнуть через иной выбор горизонта, допустимые даты, цену ожидания, cash availability и budget. Размножение одной market-date на 100 тысяч клиентов не увеличивает market N.

Реалистичность без банковского customer dataset нельзя доказать полностью. Можно доказать согласованность с наблюдёнными ограничениями и воспроизведение конкретных внешних aggregate moments, а остальное назвать scenario uncertainty. Исследование на matched payroll/remittance administrative data показывает связь переводов с доходами и информацией семьи, но выполнено в UAE и не задаёт параметры российского CIS-сегмента. Источник: [Joseph et al., Asymmetric Information and Remittances](https://www.nber.org/papers/w20986).

Сценарии: weekly, twice-monthly/salary-linked, monthly, irregular/urgent; renewal intervals с индивидуальной heterogeneity, sticky corridor/recipient, amount variation, deadline, notification fatigue. Нужно не одно удобное распределение, а grid реалистичных долей/эластичностей. Train simulator на прошлом, сравнивать на новых synthetic clients **и последующих market dates**; держать одинаковый random seed/latent baseline across policies. Обязательные negative controls: response=0, no flexibility, no FX knowledge, shuffled user features. Польза должна исчезать там, где по конструкции ей неоткуда взяться.

## 5. Что из исследований действительно применимо

| Источник | Результат источника | Применение и предел переноса |
|---|---|---|
| [Meese & Rogoff, 1983](https://www.sciencedirect.com/science/article/pii/002219968390017X) | В исследованных валютных рядах macro-модели не превосходили random walk OOS | Сильные naïve FX baselines обязательны; это не теорема невозможности прогнозировать CIS fixing |
| [Evans & Lyons, 2005](https://www.nber.org/papers/w11042) | Micro-based FX model превосходила RW/macro в их трёхлетней выборке | Новые быстрые market/flow данные потенциально ценнее очередного revised macro-level; локальная проверка обязательна |
| [Elmachtoub & Grigas, Smart Predict then Optimize](https://arxiv.org/abs/1710.08005) | Prediction loss и decision loss различаются; SPO учитывает downstream objective/constraints | Выбирать forecast/policy по явной utility; теоремы работы не дают автоматической гарантии для nonstationary FX |
| [Bergmeir, Hyndman & Koo, 2018](https://robjhyndman.com/publications/cv-time-series/) | Обычный CV может быть корректен для определённого autoregressive setup с некоррелированными errors | У нас overlapping labels, exogenous clocks и regime shifts; blanket «random KFold всегда разрешён/запрещён» неверен, outer chronological здесь необходим |
| [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html) | Post-processing улучшает вероятностную калибровку ряда нейросетей; temperature scaling прост и эффективен в их задачах | Проверить raw/temperature/Platt на temporal calibration и selected region; результат из vision не гарантия FX |
| [Geifman & El-Yaniv, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | Selective classification явно управляет risk/coverage | Строить risk–coverage frontier; их вероятностные гарантии нельзя переносить на зависимый FX без проверки предпосылок |
| [Gibbs & Candès, 2021](https://arxiv.org/abs/2106.00170), [2024](https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf) | Адаптивные интервалы для distribution shift, долговременная coverage и online adaptation | Учитывать delayed h-step feedback; nominal coverage не означает conditional coverage на selected pushes или малую ширину |
| [Bailey et al., Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | Оценивается overfitting процесса отбора из многих стратегий | Вести полный trial ledger, включая модели/окна/targets; post-hoc p-value одной строки не описывает объём поиска |
| [Gorishniy et al., TabM, ICLR 2025](https://arxiv.org/abs/2410.24210) | Parameter-efficient ensembles улучшают tabular DL performance/efficiency в их benchmarks | Один небольшой TabM — разумный neural challenger к trees; не объявлять superiority до нашего OOT |
| [Hollmann et al., TabPFN, Nature 2025](https://doi.org/10.1038/s41586-024-08328-6), [TabPFN-3 official changelog, May 2026](https://docs.priorlabs.ai/changelog/tabpfn-3) | Tabular foundation models сильны на общих таблицах; к дате проверки актуален v3 | Проверять pinned version, training-data contamination, local license, compute; foundation model не устраняет временную зависимость и drift |
| [Dudík, Langford & Li, 2011](https://arxiv.org/abs/1103.4601) | Doubly robust off-policy evaluation использует reward model и behavior policy | После randomized pilot полезно исследовать alternative policies; без реальных propensities синтетические логи не идентифицируют настоящий uplift |

Текущие SOTA работы дают кандидатов и методы проверки, а не основание обещать улучшение в процентах заранее. Для данного effective sample разумно сначала проверить long-history shallow tree/regularized additive/hazard/quantile models; затем TabM/TabPFN как ограниченные challengers. «Больше архитектур» не заменяет полученный независимый период.

## 6. Данные, ранжированные по возможной ценности

| Приоритет | Данные/эксперимент | Почему могут помочь | Честный comparator |
|---|---|---|---|
| P0 | Реальные timestamped all-in Alpha quotes на fixed notional, fee, expiry, recipient amount | Закрывают разрыв reference → customer value; может изменить ranking моделей | Та же policy на official и executable outcome |
| P0 | Publication snapshots/vintages ЦБ и локальных central banks | Проверяют доступность и deterministic cross-rate mechanics | Known-at-T против latest revision и conservative delay |
| P1 | Intraday RUB-anchor, spot-minus-fixing displacement и source-age indicators | Существующий edge чрезвычайно timing-sensitive | Existing D−1 CNY block, D−2, same-day validated cutoff |
| P1 | KASE/local USD/LCY, PBoC fixing surprise, announced interventions/FX-sale calendars | Разделяют общий RUB-factor и local residual; добавляют release surprise | Single-family PIT ablation, no-change residual |
| P1 | История CBR до 2020 с режимными сегментами | Больше уникальных дат, больше stress/recovery regimes | Identical 2023–2026 test + rolling/expanding/decay ladder |
| P2 | Праздники, payroll/tax calendars, фактически объявленные изменения методики | Calendar/context для liquidity и клиентского actionable window | Simple calendar-only/rule baseline, publish-time aware |
| P2 | Licensed order flow, liquidity spreads/depth, FX forwards/swaps | Microstructure может нести новую короткую информацию | Paired dates и latency-adjusted baseline; отдельно cost/rights |
| P3 | Широкий revised macro/news feature stack | Может помочь режимам, но мало независимых releases и большой overfit risk | Один блок за раз; release-vintage, time-stamped news, no future labels |

### Отдельный hypothetically licensed трек

Термины «юридически нечистые» и «закрытые» не взаимозаменяемы с качеством сигнала. FRED — агрегатор множества серий, многие из которых существуют у первичного публичного производителя. Его [актуальные условия](https://fred.stlouisfed.org/legal/terms/) отдельно ограничивают ML-training; условия series copyrights и условий сервиса нужно различать. [ALFRED real-time period](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) позволяет методологически отделить исторически доступную vintage от текущего знания, но доступность функции сама по себе не разрешает любой use.

Для такого трека вести `source/provider/field/license_status/observed_at/vintage`. Если доступны права/экспорт Bloomberg, использовать реальные выгрузки с provenance; [Bloomberg Data License](https://professional.bloomberg.com/products/data/data-license/) — отдельный enterprise product. Если выгрузки отсутствуют, результат «не запускался: нет данных», а не нулевой или придуманный gain.

Эксперимент состоит из четырёх строк на одинаковых test dates: public baseline; тот же экономический ряд из первичного источника; licensed extra fields; public+licensed ensemble. Это отличает улучшение информации от смены периода, frequency, лагов или текущих revisions. Положительный эффект reported как **«что было бы при наличии допустимого доступа и прав»**; он не меняет public-only scorecard. Снятие правовых ограничений не снимает point-in-time требования.

## 7. Конкретный план оценки нового решения

1. **Freeze experimental matrix.** Одна таблица всех запланированных model×features×train-window×calibration×target variants. Primary h5, development 2023–25, 2026 diagnostic. Новые просмотренные исторические данные остаются retrospective; нельзя переименовать уже использованный 2026 в untouched holdout.
2. **Reproduce existing baseline.** HGB base и HGB+CNY на текущих snapshots с numeric parity. Любое отличие версий библиотек явно отражать; новые comparisons запускать в одном окружении.
3. **Train-window ladder.** Rolling 0.5/1/2/3/5 лет, expanding; decay half-life 0.5/1/2 года. Same validation/test. Отдельно frequency of retrain: yearly baseline, quarterly/monthly с outcome-maturity purge.
4. **Bounded model challenge.** HGB, regularized logistic/GAM, CatBoost/XGBoost, simple convex ensemble; один TabM/TabPFN при доступности. Model/ensemble weights/early stopping выбирать только внутри прошлого. Сравнивать compute/latency.
5. **Objective challenge.** Binary NOW, terminal return/quantile, first-adverse-move hazard и deadline policy. Сохранять common mandatory metrics; новые labels не считать прямым улучшением старой accuracy.
6. **PIT stress.** Prefix invariance full pipeline, future-data perturbation, plus-one-source-day, revised-vintage sensitivity, source missingness/outage, pre/post methodology-change. Freshness semantics фиксировать до outcomes.
7. **Policy challenge.** Та же модель при fixed thresholds, temporal calibration, forecast utility и factual gate. Сравнить fixed cadence random/rule policies при одинаковом cap; не fill gaps future weekly top-k. Ex-post top day недели — только oracle upper bound.
8. **Synthetic users.** Renewal/periodic/urgent segments, deadlines, realistic shared CRM budget, response/latency/fatigue scenarios. Baseline — тот же латентный organic transfer plan. Два out-of-time уровня split и минимум три adverse scenarios.
9. **Inference.** Paired loss differences агрегировать на market date. Bootstrap месяца/episodes целиком со всеми corridors; стратифицировать по outer year. Опубликовать несколько block lengths, point differences, CIs, minimum cell, full trial count. Holm только обозначенного family — sensitivity screen, не возврат confirmatory status.
10. **Package.** Для каждого подхода `status=improved/negative/inconclusive/not_run`, exact delta, eligible dates, signal count, calibration/policy selection window, dependencies, seed, hashes. Final selection frozen для будущего prospective readout.

### Три числа на защите и technical appendix

Показывать три параллельных решения: (1) информативен ли сигнал — mandatory lift с uncertainty; (2) есть ли полезная величина — matched future/symmetric bps плюс downside; (3) пригодна ли policy — coverage/cluster gaps при hard cap и фактических factual predicates. Синтетический net customer value вывести отдельным scenario panel, не смешивая с рыночным эффектом.

В appendix: Brier/logloss, calibration intercept/slope и selective bins, все horizons/corridors/years, paired incumbent deltas, materiality sensitivity, downside regret, cadence distribution, source lag ladder, public/licensed split, список всех попыток и negative outcomes.

В реальном пилоте первичен persistent client-level ITT net margin/volume с BAU CRM-control и отдельной клиентской ценностью, как уже описано в `PILOT_METRIC_CONTRACT.md`. После randomization нельзя оставлять только открывших push. Число необходимых episodes выводить из power/information simulation; прежние «50 episodes» — planning assumption, а не универсальная статистическая теорема.

## 8. Что именно привнёс этот review

Review не обучает модель, поэтому **прирост model Brier/lift от него не заявляется**. Он:

- установил реальный baseline train2y+val1y;
- численно выявил несоответствие текста historical-zone и 69.7% candidates вне нижних 20% 60-дневного окна;
- отделил case requirements от самовведённого 90% weekly-fill gate;
- задал исполнимую deadline/utility постановку и правила честного synthetic evaluation;
- указал конкретные inner-validation, source-clock, fast/slow и replay ограничения;
- подготовил первичные научные основания для следующей experimental matrix и критерии, при которых усложнение будет отвергнуто.

Это улучшение валидности решения и клиентского обещания. Численное улучшение качества v3 должны дать отдельные воспроизводимые experiments с измеренными deltas, включая отрицательные результаты.
