# V4: частота клиента и политика выбора рыночной возможности

Статус: **retrospective scenario research, не причинный uplift и не production policy**. Выполнено 2026-09-05. Основной результат — разделение полезного frequent tradeoff и провалившейся общей гипотезы. Все показатели ниже воспроизводятся из CSV, отрицательные результаты сохранены.

## Что реально добавлено к V3

V3 давал readiness gate после одинакового рыночного сигнала. Здесь каждой группе разрешено выбирать собственные **горизонт h1/3/5/10/20, порог вероятности и минимальный промежуток между рыночными возможностями**. Настройка использует только прошлые OOT годы. Реально скачаны микроданные World Bank L2KGZ и получены агрегаты регулярности вместо ссылок на каталог; сохранена сопоставимость рынков, пользователей, расходов и объёма между политиками.

**Частым пользователям** group policy дала 12.0% больше контактов (10.667→11.947) и 14.1% больше условной net timing value (164.75→188.02 RUB/client-period), при падении common-h5 quality на 2.62 п.п. Это прозрачная цена частоты, а не улучшение accuracy FX модели.

**В фиксированной основной смеси** результат хуже universal: 621.93→260.84 RUB/client-period (-58.1%), контакты 7.626→4.580, h5 quality 44.98%→38.91%, relevance 53.99%→61.45%. Улучшение relevance само по себе не доказывает денежную полезность. Политика без ограничений cadence также не превзошла universal; проблема включает нестабильность выбора параметров по короткой истории отдельных групп.

## Сегментация из чатов — другой признак

Источник: `Researches/chat_research_kz/report.md`, §3; `Presentation Artifacts/Презентация_Д4.md`, слайд6; `product_artifacts/CLIENT_JOURNEY.md`.

| Гипотеза из чатов | Продуктовый смысл | Что доступно этому эксперименту |
|---|---|---|
| S1 регулярный отправитель с гибким окном | Предлагать дату в уже известном допустимом окне | Частота симулируется; гибкость и intent — отдельные сценарные переменные |
| S2 срочная потребность | Сразу текущая сумма и надёжный маршрут | Срочные потребности никогда не сдвигаются; preview подавляет push |
| S3 ищет работающий маршрут | Доступность, total cost, fallback | Нельзя лечить проблему маршрута прогнозом курса |
| S4 сравнивает банк/P2P/наличные | Сравнение исполнимого net результата | Предпочтение внешнего канала не выводится из банковской истории |
| S5 суперчастый участник чата | Роль не установлена | Количество сообщений не означает число переводов |

S1/S2 могут быть состояниями одного клиента в разные дни. Три группы frequent/monthly/occasional — ось частоты внутри потенциальной аудитории, а не доли пяти JTBD. Ни raw чаты, ни идентификаторы авторов не экспортированы. Чатовые counts и qualitative samples не использованы как population weights.

## Первичные данные и веса

[World Bank RPP baseline report](https://documents1.worldbank.org/curated/en/552541540823620142/pdf/131455-RPPbaselinesurveyFINAL.pdf): полевой опрос 2016, Kyrgyzstan819 и Tajikistan1053 респондента, figure5.1.2.3, PDFpage20/printed19. Скачан PDF, числа проверены визуально. KG: ≥monthly40%,4–6/year32%,2–3/year21%,yearly6%; TJ41/34/17/7. Округлённая сумма99% нормализуется. Это получатели переводов, преимущественно из РФ, не банк-клиенты-отправители. Несколько отправителей у одной семьи дополнительно мешают прямому переносу частоты.

Основная fixed mixture: frequent4/99,monthly36/99,occasional59/99. Наблюдаемы только первые две группы вместе40/99; распределение 10% monthly-bin в frequent — **плановое допущение**, не оценка. Sensitivity меняет frequent долю этого bin от0 до50%, держит одинаковые веса для всех политик. Частота 7/14 дней внутри frequent остаётся сценарной, не доказанной российской payroll статистикой.

[World Bank Listening to Kyrgyz Republic 2021–2025](https://microdata.worldbank.org/catalog/6523), Ref KGZ_2021-2025_L2KGZ_v02_M, [DOI](https://doi.org/10.48529/swmc-aq82). Скачан официальный CSV ZIP 7,673,631 bytes: 62,235 household-month и 324,050 individual-month строк. Используется `mig_living_remittance` — yes/no отправки домохозяйству за прошлый месяц. Есть 8,257 наблюдаемых migrant-month и 1,490 мигрантов; пропуски не превращены в no. Исходные rows находятся только во временном исследовательском каталоге, в repository только агрегаты. Автор данных: World Bank; скачано 2026-09-05. Публичный v02 релиз содержит поздние наблюдения: даже подвыборка до2024 не объявляется доступной в2023.

| period | previous_sent | consecutive_pairs | unweighted_p_next_sent | derived_household_weighted_p_next_sent |
|---|---|---|---|---|
| pre2024_observations_released_later | 0 | 1899 | 0.324 | 0.341 |
| pre2024_observations_released_later | 1 | 1689 | 0.700 | 0.696 |
| full2021_2025 | 0 | 2529 | 0.319 | 0.332 |
| full2021_2025 | 1 | 2288 | 0.708 | 0.701 |

Последовательные пары — один и тот же мигрант в соседних календарных месяцах; апрель2024 отсутствует и не склеивается. Используется population weight `popw`; для мигрантских распределений дополнительно показан производный household weight `popw/hhsize`, усреднённый по наблюдениям. Это descriptive adjustment, **не предоставленные longitudinal migrant weights**. Нет design-based population CI. Условие ≥6/12 наблюдений и ≥1 перевода создаёт отбор; attrition и возвращение мигрантов важны.

| band | migrants | eligible_migrants | unweighted_share | derived_household_weighted_share |
|---|---|---|---|---|
| sporadic_le25pct_months | 87 | 447 | 0.195 | 0.175 |
| intermittent_25to75pct_months | 268 | 447 | 0.600 | 0.608 |
| regular_gt75pct_months | 92 | 447 | 0.206 | 0.217 |

Новая empirical sensitivity относит >75% observed-month senders к monthly-or-frequent bin, а остальные к occasional. Внутри bin10% условно frequent. Это перевод survey definition в stress scenario, **не идентификация частоты внутри месяца**. Новые данные подтверждают persistence, но показывают существенные месяцы без переводов: регулярный календарь V3 не прошёл полноценную внешнюю валидацию. Значения intent TPR/FPR, response35%, окно2/5/10дней, urgent20%, чек18/45/90тысRUB и fatigue остаются гипотезами.

[IOM Kyrgyz return-migrant survey July2024](https://kyrgyzstan.iom.int/sites/g/files/tmzbdl1321/files/documents/2024-11/kg_return-migrant-survey_r3_eng.pdf): официальный поисковый индекс содержит9% >monthly,57% monthly,18% quarterly,15% circumstances,1% nonresponse среди отправлявших. PDF вернул403; поэтому это отдельно помеченный sensitivity, не основной downloaded источник. Распределение не переносится на Alfa и не используется как известное в2023. Категория circumstances условно объединена с occasional; это ещё одно допущение.

## Протокол и защита от ложного улучшения

- Заморожен `baseline_reproduction`, один model family, по всем пяти h. Никакого дополнительного обучения FX модели на синтетике.
- Universal выбирает одну из60 комбинаций h×quantile(.25,.50,.75)×cadence(3,7,14,28 calendar days). Пороги вычисляются по прошлым score distributions, не test.
- Group-aware: frequent cadence3/7,monthly7/14,occasional14/28; каждая группа выбирает свою комбинацию. Цель: gross timing RUB минус1RUB/contact; минимум2 контакта/client-period в обучающей смеси. Unconstrained group использует все60 комбинаций для каждой группы: добавлен **после просмотра первого результата**, явный exploratory followup.
- Train2023→test2024; train2023–24→test2025; train2023–25→diagnostic2026. Все старые h20 label horizons завершены до следующего года; тест проверяет это по исходным CBR session positions. Сами V3 scores и рынки ранее были просмотрены: это nested prior-year retrospective selection, не нетронутый prospective holdout.
- Контакты разрешены только на **пересечении** доступных дат всех h. Так h1 не получает дополнительные tail dates. Common market rows by year: {'2023': 1135, '2024': 1140, '2025': 1135, '2026': 705}. Всего 823 уникальных дат, а не множество независимых synthetic examples.
- Все политики получают одинаковые cohorts, random uniforms, потребности, суммы и FX paths. Training seed7101 отличается от evaluation8101/8102/8103; evaluation180 клиентов/segment, 540 total. Синтетическая population composition не меняется между политиками.
- Reuse `research_v3.behavior.simulate`: готовность `(phase OR observed intent) AND balance`, recent-transfer gate, максимум2 контакта в rolling7 calendar days. У пользователя один primary corridor; cap общий для предложений симуляции. Внешний CRM traffic и multicorridor portfolio клиента не симулированы, их лимиты нужны в боевом gate.
- Cadence применяется к хронологическому потоку **market candidates до readiness**, поэтому она causal, но может пропустить редкое удачное окно готовности. Это конкретная проверенная политика, а не доказательство бесполезности любой personalization.
- Organic перевод происходит в срок; можно только продвинуть его внутри funded window. `A*(FX_due/FX_exec−1)` — conditional customer timing-value proxy. Net вычитает contact cost и execution drag. Count и RUB volume неизменны во всех arms; bank incremental conversion, revenue, profit и causal uplift **не оценены**.
- CBR daily — reference rate/publication proxy, не исполнимый bank quote; одинаковые real FX paths недостаточны для финансового P&L доказательства.

## Сопоставимые метрики, 2024–2025

Client-period — от1января до последней общей score date года, не полный год. Основной pooling усредняет client-periods, сценарий и mixture фиксированы; `contacts_per_week` нормирует экспозицию. Gross/net здесь RUB на клиента за период. **h5_quality** всегда оценивает одну и ту же h5 цель. Own-h quality сохранена в CSV для диагностики, но не используется для заявления роста качества: более короткая цель проще. Relevance — доля контактов, попавших в латентное окно потребности, известное только симулятору.

| policy | contacts | contacts_per_week | h5_quality | relevance | gross_timing_value_rub | net_scenario_value_rub | timing_bps_all_volume |
|---|---|---|---|---|---|---|---|
| group_aware | 4.580 | 0.095 | 0.389 | 0.614 | 265.416 | 260.836 | 5.763 |
| group_expected_budget | 4.580 | 0.095 | 0.389 | 0.614 | 265.416 | 260.836 | 5.763 |
| group_unconstrained_expected_budget | 4.218 | 0.087 | 0.381 | 0.567 | 190.389 | 186.172 | 4.134 |
| group_unconstrained_exploratory | 8.298 | 0.172 | 0.384 | 0.551 | 302.298 | 293.999 | 6.564 |
| random_expected_budget | 4.421 | 0.092 | 0.281 | 0.567 | 45.320 | 40.899 | 0.984 |
| universal | 7.626 | 0.158 | 0.450 | 0.540 | 629.558 | 621.932 | 13.669 |
| universal_expected_budget | 3.685 | 0.076 | 0.446 | 0.566 | 337.677 | 333.993 | 7.332 |
| v3_readiness | 5.718 | 0.119 | 0.425 | 0.578 | 468.204 | 462.486 | 10.166 |
| weekly_expected_budget | 4.498 | 0.093 | 0.268 | 0.592 | -45.105 | -49.603 | -0.979 |

`v3_readiness` — контроль V3 gate на одинаковом новом common grid и frozen baseline h5 candidates. Его цифры не являются прямой заменой sealed V3 report с другим покрытием дат/модельным артефактом/смесью.

## Частота против качества по сегментам

| segment | policy | contacts | h5_quality | relevance | net_scenario_value_rub |
|---|---|---|---|---|---|
| frequent | group_aware | 11.947 | 0.424 | 0.672 | 188.021 |
| monthly | group_aware | 6.992 | 0.345 | 0.596 | 169.424 |
| occasional | group_aware | 2.608 | 0.450 | 0.626 | 321.549 |
| frequent | group_unconstrained_exploratory | 11.947 | 0.424 | 0.672 | 188.021 |
| monthly | group_unconstrained_exploratory | 12.711 | 0.384 | 0.544 | 196.634 |
| occasional | group_unconstrained_exploratory | 5.358 | 0.378 | 0.542 | 360.594 |
| frequent | universal | 10.667 | 0.451 | 0.675 | 164.747 |
| monthly | universal | 9.447 | 0.441 | 0.555 | 379.371 |
| occasional | universal | 6.308 | 0.458 | 0.511 | 800.930 |

Frequent действительно чаще получает предложения и выигрывает по timing proxy с потерей h5 quality. Но у редкого клиента есть мало потребностей: отбрасывание рыночных дат по длинному cooldown лишает его значительной части ценности. «Редкие переводы» не означают «редко проверять рынок»; правильнее редко отправлять пуши и внимательно проверять короткое окно реального intent.

### Frequent: чувствительность response/cost и поддержка рынками

| scenario | policy | contacts | h5_quality | gross_timing_value_rub | net_scenario_value_rub |
|---|---|---|---|---|---|
| base | group_aware | 11.947 | 0.424 | 199.968 | 188.021 |
| base | universal | 10.667 | 0.451 | 175.414 | 164.747 |
| contact_cost10 | group_aware | 11.947 | 0.424 | 199.968 | 80.496 |
| contact_cost10 | universal | 10.667 | 0.451 | 175.414 | 68.747 |
| execution_drag25bps | group_aware | 11.947 | 0.424 | 199.968 | 63.151 |
| execution_drag25bps | universal | 10.667 | 0.451 | 175.414 | 50.549 |
| high_response | group_aware | 12.142 | 0.425 | 367.765 | 355.623 |
| high_response | universal | 10.814 | 0.452 | 362.343 | 351.529 |
| low_response | group_aware | 11.842 | 0.424 | 61.482 | 49.640 |
| low_response | universal | 10.603 | 0.451 | 61.464 | 50.861 |
| narrow_flexibility | group_aware | 7.097 | 0.425 | 35.644 | 28.547 |
| narrow_flexibility | universal | 6.319 | 0.454 | 33.669 | 27.350 |
| weak_intent | group_aware | 9.781 | 0.415 | 101.042 | 91.261 |
| weak_intent | universal | 8.606 | 0.437 | 92.257 | 83.652 |
| zero_response | group_aware | 11.794 | 0.424 | 0.000 | -11.794 |
| zero_response | universal | 10.556 | 0.451 | 0.000 | -10.556 |

| policy | unique_market_dates | synthetic_contacts |
|---|---|---|
| group_aware | 167 | 4301 |
| group_expected_budget | 167 | 4301 |
| group_unconstrained_expected_budget | 166 | 2236 |
| group_unconstrained_exploratory | 167 | 4301 |
| random_expected_budget | 450 | 2261 |
| universal | 157 | 3840 |
| universal_expected_budget | 156 | 1806 |
| v3_readiness | 168 | 2824 |
| weekly_expected_budget | 92 | 2235 |

Контакты искусственных пользователей повторно используют одни и те же рыночные даты. Всего common development grid455 уникальных дат (228в2024,227в2025); они также serially dependent и коррелированы между коридорами. Число эффективно независимых рыночных наблюдений меньше и не известно. +14% — модельная разность при response35%, не измеренный treatment uplift; таблица показывает смену величины/знака при других предположениях.

## Expected budget controls и стабильность

Target — минимум prior weighted contact means universal/group. Каждая arm получает Bernoulli thinning, калиброванный на предыдущих users/years; current/test quota не читается. Random sender выбирает дни без FX score, weekly — calendar gap7; readiness одинаковый. Prior calibration receipts содержат target, achieved mean и thinning probability. Это **matched expected budget**, не точное ex-post равенство: на новых годах budget drift виден в таблице и нельзя приписать всю разность только selection quality. В первой основной паре universal_expected даже меньше контактов, чем group_expected, но больше value; это не даёт формального causal superiority.

| year | policy | contacts | h5_quality | net_scenario_value_rub |
|---|---|---|---|---|
| 2024 | universal | 8.570 | 0.495 | 966.508 |
| 2024 | group_aware | 5.011 | 0.376 | 231.230 |
| 2024 | group_unconstrained_exploratory | 8.996 | 0.383 | 366.841 |
| 2025 | universal | 6.681 | 0.392 | 277.356 |
| 2025 | group_aware | 4.148 | 0.404 | 290.442 |
| 2025 | group_unconstrained_exploratory | 7.600 | 0.385 | 221.158 |
| 2026 | universal | 5.571 | 0.447 | 234.702 |
| 2026 | group_aware | 2.828 | 0.396 | 128.056 |
| 2026 | group_unconstrained_exploratory | 4.177 | 0.477 | 171.682 |

2025 и2024 расходятся по знаку общего эффекта. 2026 неполный и diagnostic. В `corridor_year_diagnostics.csv` сохранены все пять corridor×year cells; в `seed_diagnostics.csv` Monte Carlo разброс, который нельзя трактовать как неопределённость новых рыночных режимов. Paired8-week block diagnostic h5-quality difference group−universal95% interval: [-11.854706121586082, -5.826807471199846, 0.21864882000328748]; единица resampling — календарная неделя со всеми коридорами/клиентами внутри. Это условный интервал фиксированных selected policies, без refit, **не confirmatory CI** и не CI причинного эффекта.

## Нулевые и слабые поведенческие сигналы

| scenario | policy | contacts | h5_quality | relevance | gross_timing_value_rub | net_scenario_value_rub |
|---|---|---|---|---|---|---|
| base | group_aware | 4.580 | 0.389 | 0.614 | 265.416 | 260.836 |
| base | group_expected_budget | 4.580 | 0.389 | 0.614 | 265.416 | 260.836 |
| base | universal | 7.626 | 0.450 | 0.540 | 629.558 | 621.932 |
| base | universal_expected_budget | 3.685 | 0.446 | 0.566 | 337.677 | 333.993 |
| contact_cost10 | group_aware | 4.580 | 0.389 | 0.614 | 265.416 | 219.620 |
| contact_cost10 | group_expected_budget | 4.580 | 0.389 | 0.614 | 265.416 | 219.620 |
| contact_cost10 | universal | 7.626 | 0.450 | 0.540 | 629.558 | 553.299 |
| contact_cost10 | universal_expected_budget | 3.685 | 0.446 | 0.566 | 337.677 | 300.832 |
| execution_drag25bps | group_aware | 4.580 | 0.389 | 0.614 | 265.416 | 125.519 |
| execution_drag25bps | group_expected_budget | 4.580 | 0.389 | 0.614 | 265.416 | 125.519 |
| execution_drag25bps | universal | 7.626 | 0.450 | 0.540 | 629.558 | 389.890 |
| execution_drag25bps | universal_expected_budget | 3.685 | 0.446 | 0.566 | 337.677 | 213.257 |
| high_response | group_aware | 4.606 | 0.389 | 0.611 | 552.529 | 547.923 |
| high_response | group_expected_budget | 4.606 | 0.389 | 0.611 | 552.529 | 547.923 |
| high_response | universal | 7.710 | 0.450 | 0.486 | 1026.792 | 1019.082 |
| high_response | universal_expected_budget | 3.707 | 0.446 | 0.536 | 574.699 | 570.992 |
| low_response | group_aware | 4.565 | 0.389 | 0.616 | 92.274 | 87.709 |
| low_response | group_expected_budget | 4.565 | 0.389 | 0.616 | 92.274 | 87.709 |
| low_response | universal | 7.559 | 0.450 | 0.583 | 193.809 | 186.249 |
| low_response | universal_expected_budget | 3.660 | 0.446 | 0.588 | 91.719 | 88.059 |
| narrow_flexibility | group_aware | 2.955 | 0.386 | 0.243 | 47.409 | 44.454 |
| narrow_flexibility | group_expected_budget | 2.955 | 0.386 | 0.243 | 47.409 | 44.454 |
| narrow_flexibility | universal | 4.899 | 0.450 | 0.242 | 149.394 | 144.495 |
| narrow_flexibility | universal_expected_budget | 2.387 | 0.441 | 0.233 | 72.600 | 70.213 |
| weak_intent | group_aware | 4.489 | 0.383 | 0.329 | 146.494 | 142.005 |
| weak_intent | group_expected_budget | 4.489 | 0.383 | 0.329 | 146.494 | 142.005 |
| weak_intent | universal | 7.612 | 0.447 | 0.291 | 351.218 | 343.606 |
| weak_intent | universal_expected_budget | 3.644 | 0.442 | 0.298 | 153.376 | 149.732 |
| zero_response | group_aware | 4.559 | 0.389 | 0.617 | 0.000 | -4.559 |
| zero_response | group_expected_budget | 4.559 | 0.389 | 0.617 | 0.000 | -4.559 |
| zero_response | universal | 7.524 | 0.450 | 0.606 | 0.000 | -7.524 |
| zero_response | universal_expected_budget | 3.645 | 0.446 | 0.599 | 0.000 | -3.645 |

Zero-response обязан давать gross0 и net=−contact cost; это проверено. Weak intent использует TPR.30/FPR.15, balance sensitivity.75, phase noise7days вместо базовых. Policy choices остаются замороженными. Узкое окно, contact cost10 и execution drag25bps — stress tests. Если стоимость контакта трактуется как CRM бизнес-cost, складывать её с customer timing value можно только как joint objective, **не bank contribution margin**.

## Sensitivity смеси без изменения cohort composition

| mix | universal | group_aware | group_minus_universal |
|---|---|---|---|
| frequent_heavy_stress | 335.293 | 201.541 | -133.752 |
| iom2024_returnees_scenario | 500.380 | 221.823 | -278.556 |
| l2kgz_regular10pct_frequent_scenario | 704.642 | 288.890 | -415.752 |
| v3_balanced | 448.350 | 226.331 | -222.018 |
| wb2016_kg_frequent_half_of_monthly | 587.245 | 263.842 | -323.404 |
| wb2016_kg_frequent_quarter_of_monthly | 608.924 | 261.963 | -346.961 |
| wb2016_kg_no_frequent | 630.603 | 260.085 | -370.519 |
| wb2016_kg_planning | 621.932 | 260.836 | -361.096 |
| wb2016_tj_same_split | 617.457 | 259.318 | -358.139 |

Для каждого mix одни и те же веса применены ко всем arms **до вычисления ratios**. Policy fit и бюджет main-mixture не перенастраиваются под новое распределение: sensitivity показывает переносимость одной политики, а не выигрыш от смены состава выборки. Formal population weights банка отсутствуют. Нет признака-указания национальности/этничности клиента: в production сегмент определяется только собственным consented поведением.

## Выборы и воспроизводимость

| test_year | segment | universal | group_aware | unconstrained |
|---|---|---|---|---|
| 2024 | frequent | h1_q0.50_cad3 | h10_q0.50_cad3 | h10_q0.50_cad3 |
| 2024 | monthly | h1_q0.50_cad3 | h20_q0.50_cad7 | h3_q0.25_cad3 |
| 2024 | occasional | h1_q0.50_cad3 | h20_q0.25_cad14 | h20_q0.50_cad7 |
| 2025 | frequent | h5_q0.50_cad3 | h3_q0.50_cad3 | h3_q0.50_cad3 |
| 2025 | monthly | h5_q0.50_cad3 | h20_q0.25_cad7 | h5_q0.25_cad3 |
| 2025 | occasional | h5_q0.50_cad3 | h3_q0.25_cad14 | h5_q0.50_cad3 |
| 2026 | frequent | h3_q0.50_cad3 | h3_q0.50_cad3 | h3_q0.50_cad3 |
| 2026 | monthly | h3_q0.50_cad3 | h1_q0.25_cad7 | h3_q0.50_cad3 |
| 2026 | occasional | h3_q0.50_cad3 | h3_q0.25_cad14 | h3_q0.75_cad3 |

`selected_policy_receipts.json` хранит fit years, max prior date и все абсолютные thresholds. `manifest.json` — SHA256 пяти input files, rates и V3 simulator; исходники не мутировали. Seven meaningful tests: poison future scores, chronological schedule prefix invariance, label maturity h20, zero response with unchanged worlds, future/invalid preview refusal, urgency and cap persistence.

Команды: см. README. По умолчанию mode=universal. `preview.build_segment_policy_preview` композиционно использует V3 behavior API и selected receipt; оставляет probability как есть, проверяет доступность receipt/scores/context и добавляет причины threshold/cadence. Даже bank_observed context возвращает production_eligible=False. Новая ветка не меняет sealed V3/final_solution.

## Продуктовое решение

1. Оставить universal market policy + readiness базовой исследовательской arm; сегменты S1–S5 определяют потребность/экран, а не автоматически вероятность выгоды.
2. Frequent cadence можно показывать как отдельный явно обозначенный выбор «больше возможностей / ниже доля h5 удачных дат». Здесь оценён tradeoff, а не универсальный выигрыш.
3. Для monthly/occasional искать **событие готовности**, затем оценивать рынок достаточно часто в этом окне; долгий календарный cooldown до readiness не принимать без проверки.
4. Чтобы заявить причинную пользу, нужен prospective user-level randomized holdout с неизменным общим budget, real executable quote, delivered/opened/intent/purchase, deadline и net amount. Randomize profile policy, проверять incremental completed-transfer volume и bank CM отдельно от переноса уже запланированных переводов. Срочные случаи не включать в wait-treatment.
5. Самый сильный новый evidence asset — реально доступная панель Кыргызстана: она позволяет проверять missing months и persistence и проектировать irregular recurrent simulator. Но 7/14-дневная frequent frequency и push response по этим данным не идентифицируются.
