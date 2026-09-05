# Дополнительные данные: что фактически проверено

Дата исследования: 4–5 сентября 2026. Статус: **retrospective exploratory**.

**Крупного подтверждённого прироста от покупки общего macro/risk feed не обнаружено.** Мы действительно обучили модели на ранее исключённом FRED и на CBOE, проверили открытые GPR и восстановили FRED inflation series прямо из US Treasury. Лучшая новая гипотеза — шестипризнаковый блок инфляционных ожиданий с лагом 7 календарных дней: Brier улучшился на **1.07%**, но доверительный интервал включает ноль, а reference-rate выгода стала ниже на **3.96 bps**. Это кандидат для следующего исследования, а не доказанная победа по всем метрикам.

Зато получен практический результат по доступу: **FRED T10YIE/T5YIFR полностью воспроизводятся из первичных данных US Treasury**. На 1 920 общих датах значения совпали точно, а на 4 415 OOT прогнозах совпали вероятности и решения моделей — для обоих лагов. Для этого блока лицензия FRED не добавляет информации.

## Сравнение с существующим решением

Ниже — h=5, 2023–2025, одинаковые train/validation/test и модель HGB. Все строки используют базовые признаки и CNY basis. Brier меньше — лучше; hit rate, lift и reference gain относятся к **коридорным кандидатам** (`candidate_signal`), а не к доставленным пушам или исполнимой экономии клиента.

| Дополнение к старому CNY basis | Лаг, календарные дни | Brier | Δ Brier к старому | Hit rate | Lift | Reference gain, bps |
|---|---:|---:|---:|---:|---:|---:|
| Старый `hgb_plus_cnyrub_basis` | старый профиль | 0.190617 | 0 | 45.25% | 1.435 | 47.69 |
| FRED inflation / точная Treasury-реплика | 2 | 0.192735 | +0.002118 | 42.03% | 1.333 | 37.45 |
| CBOE VIX, VIX9D, VVIX, OVX, VXEEM | 2 | 0.194184 | +0.003567 | 41.31% | 1.294 | 36.89 |
| GPR daily + monthly Russia/Ukraine | 14 / 15 после конца месяца | 0.193105 | +0.002489 | 40.73% | 1.277 | 27.63 |
| Все три новых блока | как выше | 0.199785 | +0.009169 | 40.56% | 1.275 | 27.22 |
| **FRED inflation / Treasury-реплика** | **7** | **0.188583** | **−0.002033** | **45.76%** | **1.456** | **43.72** |
| CBOE volatility | 7 | 0.190303 | −0.000313 | 41.68% | 1.301 | 31.79 |
| GPR | 30 / 45 после конца месяца | 0.201204 | +0.010587 | 44.73% | 1.374 | 37.40 |

Улучшение FRED/Treasury +7: **Δ Brier = −0.002033**, относительное улучшение **1.067%**, 95% month-block CI **[−0.00736; +0.00282]**. Улучшились 2/3 года и 8/15 год×коридор ячеек. Mean weekly coverage 1–2 сигнала составляет 86.86%; это также не закрывает старый gate 90%. Выбор лага происходил в заранее исполненной sensitivity-сетке, но вся работа остаётся exploratory: выбирать лучший просмотренный лаг и объявлять его новым holdout запрещено методологией.

Для слабого `hgb_base` (Brier 0.218714) новые основные блоки тоже не дали устойчивого преимущества: FRED 0.227607, CBOE 0.220618, GPR 0.220304. Следовательно, неудача дополнений не объясняется только слишком сильным CNY baseline.

Полные таблицы: `RESULTS_COMPARISON.csv`, `development_metrics.csv`, `development_paired_audit.csv`, `development_stale_metrics.csv`, `development_stale_paired_audit.csv`. В них сохранены raw/calibrated proper scores, частота, покрытия, сожаление и policy-метрики старого evaluator.

## «Что было бы, если бы можно было свободно использовать»

**FRED — эксперимент выполнен.** Использованы T10YIE и T5YIFR с первичными и увеличенными лагами. На уровне отдельных серий указано `Copyrighted: Citation Required`, однако общие FRED Services Terms отдельно ограничивают ML-use, включая non-commercial use. Это разные уровни условий. Здесь выполнен ровно запрошенный локальный counterfactual; право коммерческого применения и распространения сырых данных этим не установлено. Вывод — небольшой exploratory эффект при недельном лаге можно получить без FRED, восстановив показатели непосредственно из Treasury. Источники: [FRED terms](https://fred.stlouisfed.org/legal/), [T10YIE](https://fred.stlouisfed.org/series/T10YIE), [T5YIFR](https://fred.stlouisfed.org/series/T5YIFR).

**ICE BofA credit через FRED — ограниченный эксперимент выполнен.** С апреля 2026 обе страницы источника предупреждают, что свободная история содержит только 3 года. Реальная выгрузка BAMLH0A0HYM2 и BAMLEMCBPIOAS начинается **2023-09-05**; эти данные не выдавались за полный train для 2023–2025. Отдельно выполнено matched comparison: train 2024, validation 2025, diagnostic 2026. Basis Brier **0.223178 → 0.222582**, относительное улучшение **0.267%**, Δ **−0.000596**, CI **[−0.01028; +0.00754]**. Reference gain **32.28 → 25.12 bps**. Это не основание покупать длинный ICE feed ради ожидаемого сильного буста. По заметкам ICE внутреннее использование и распространение третьим сторонам регулируются отдельно. Источники: [US HY OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2), [EM corporate OAS](https://fred.stlouisfed.org/series/BAMLEMCBPIOAS).

**CBOE — эксперимент выполнен.** Пять индексов скачаны из официальных CSV, включая ожидаемую волатильность нефти OVX и emerging markets VXEEM. Из них построены levels, 5-session changes, VIX9D/VIX и VXEEM/VIX. Публичный download не приравнивался к свободной commercial ML-license: сайт предусматривает personal non-commercial use, отдельное согласие и исключение fair use. Поэтому это отдельный research/counterfactual tier. Основной lag=2 ухудшил результат; lag=7 дал лишь −0.16% Brier и более слабую policy-выгоду. FRED VIXCLS совпадает с CBOE VIX на **2 212 датах**. Источники: [historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data), [CBOE terms](https://www.cboe.com/terms).

**Bloomberg — эксперимент не выполнен, эффект неизвестен.** В переданных материалах нет авторизованного экспорта Bloomberg, а публичные страницы не предоставляют нужную длинную bid/ask или intraday историю. Bloomberg Data License предоставляет historical datasets и программную доставку в рамках клиентского доступа; наличие платного продукта само по себе не означает запрета research. Доступ не обходился, численный «буcт Bloomberg» не придуман. Официально доступен Virtual Data Room для оценки данных. [Bloomberg Data License](https://professional.bloomberg.com/products/data/data-license/), [DATA GO terms](https://data.bloomberg.com/tos/).

Для предметного запроса в Bloomberg/MOEX нужен не абстрактный «доступ ко всем данным», а следующий пакет: timestamped CNY/RUB bid/ask/trades до и после 12:30/15:30 MSK, соответствующие версии fixing, USD/CNH, исполнимые local USD/LCY quotes и точное разрешение internal research/ML. Тест: зафиксировать decision cutoff, latency и executable quote target; сопоставить incremental Brier/log loss, regret и net recipient units с прямым MOEX/CBR/reference baseline. Сравнивать стоимость feed с ожидаемой **инкрементальной** ценностью на eligible перевод, а не с общей суммой переводов. Все эти эксперименты требуют самого экспорта; текущая работа не доказывает их эффект.

## Открытые данные и воспроизводимость

**US Treasury.** Дополнительно скачаны real yield curves за 2019–2026 и nominal curve 2019; nominal 2020–2026 взяты из frozen bundle. `T10YIE = nominal10 − real10`; T5YIFR восстановлен по опубликованной формуле FRED, с округлением до 0.01 процентного пункта. Прогрев до 2020 обязателен: без 2019 года первые обучающие изменения ставки имеют иное количество пропусков и даже одинаковые ряды могут породить разные модели. После исправления прогрева direct/FRED predictions совпадают точно. [Treasury methodology and publication](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics).

**GPR.** Использованы freely available CC BY индексы Caldara–Iacoviello, а не тексты газет. По странице авторов daily series обновляется еженедельно, monthly — в начале следующего месяца; есть задержка газет и пересмотры. Lag 14/30 дней не превращает latest snapshot в истинный vintage. Найден официальный [архив версий](https://github.com/iacoviel/iacoviel.github.io/tree/master/gpr_archive_files); listing 689 файлов сохранён. Он не подменял latest snapshot в данном эксперименте. [Журнал](https://github.com/iacoviel/iacoviel.github.io/blob/master/gpr_files/log.txt) документирует исправление пропущенных статей в феврале 2024. Даже на latest данных с консервативным лагом основной блок не помог; vintage replay остаётся отдельной проверкой, если впоследствии появится полезная GPR-гипотеза. [Страница данных, лицензия, график обновлений](https://www.matteoiacoviello.com/gpr.htm?trk=public_post_comment-text).

**Продление MOEX для экспериментов с длиной обучения.** Получено 1 693 наблюдения CNY/RUB TOM **2013-04-15 — 2019-12-30** и 106 наблюдений CNY fixing **2019-08-01 — 2019-12-30** в `cny_history/`. Это совместимый с frozen bundle формат. Запрос был за 2010–2019: более ранний fixing endpoint не вернул. Поэтому нельзя интерпретировать десять лет CNY-basis training как десять лет полного basis coverage. Скрипт `fetch_cny_history.py` переиспользует прежний ISS-loader и сохраняет receipt с hash/units/availability. Эффект на длинные модели должен измеряться в matched history ablation; этот отчёт не приписывает ему метрики другого эксперимента.

## Дополнительный post-hoc тест: длинный train + Treasury

После основной матрицы выполнена ограниченная комбинация с длинными моделями основного исследования. Импортирован extended CBR panel из `research_v3/models`, train=120 месяцев, validation=12, early stopping отключён, как в исходном long-run. Дополнены только шесть Treasury lag7 признаков. Оба длинных baseline воспроизведены с max probability difference **1.11e−16**.

| Вариант | Brier 2023–2025 | Lift | Reference gain, bps |
|---|---:|---:|---:|
| Long 120 месяцев | 0.184850 | 1.371 | 40.08 |
| Long 120 + Treasury lag7 | **0.183295** | 1.343 | 31.52 |
| Long 120 + extended CNY history | 0.187480 | 1.342 | 38.19 |
| Long 120 + extended CNY + Treasury lag7 | 0.187197 | 1.326 | 35.33 |

Первый combo снижает Brier на **0.842% против matched long baseline** и на **3.84% против старого CNY basis**. Улучшение Brier есть в **3/3 годах и 13/15 ячейках**, но paired 95% CI Δ **[−0.004148; +0.000898]** включает ноль. Lift и reference gain ухудшаются; в diagnostic 2026 Brier также немного ухудшается: **0.208406 → 0.208938**. Поэтому это дополнительный результат по proper scoring, а не улучшение клиентского продукта по всем критериям.

Для extended CNY incremental Δ Brier **−0.000282**, CI **[−0.002920; +0.002307]**. Macro features в combo по-прежнему доступны **только с 2020**, более ранние строки имеют NaN; десять лет official training не превращаются в десять лет полной external-data coverage. Метрики, прогнозы и проверки находятся в `long_combo/`. Этот combo выбран после основных результатов и явно помечен post-hoc.

**Уточнение baseline для доверительных интервалов.** Указанный выше CI long120 + Treasury относится к **добавочному эффекту Treasury против long120**, а не к полному эффекту против старого решения. Отдельный основной аудит [`../models/external_combo_vs_incumbent_ci.csv`](../models/external_combo_vs_incumbent_ci.csv) получил общий Δ Brier **−0.007322** к original incumbent, 95% CI **[−0.014351; −0.000610]**, 2/3 улучшившихся года и 12/15 ячеек: этот total-effect CI не включает ноль. Incremental CI **[−0.004148; +0.000898]** включает ноль. Оба сравнения retrospective/post-selection, не скорректированы за весь перебор и не являются подтверждением на новой выборке. В 2026 общий Δ **+0.002311** ухудшается, CI **[−0.008411; +0.014146]** включает ноль.

## Протокол и ограничения

- Старые `hgb_base` и `hgb_plus_cnyrub_basis` воспроизведены **точно**: max probability difference 0 на 3 635 development rows, candidate decisions идентичны frozen bundle.
- 58 уникальных model×year fits основной матрицы: 40 основных, 16 lag-sensitivity, 2 matched ICE. Ещё 16 fit в отдельном long combo, итого **74**. HGB depth=2, 120 trees, old core unchanged; 2 года train, 1 год calibration/threshold validation, тест на следующий год; h=5 хвосты очищены. ICE имеет отдельный явно указанный train=2024.
- 2023–2025 — development OOT. Уже просмотренный, неполный 2026 — diagnostic. 3 635 строк — **727 дат**, а не 3 635 независимых испытаний. Month-block bootstrap сохраняет все пять валют даты вместе, 10 000 повторов.
- US close series доступны через lag 2 / 7 календарных дней; GPR daily 14 / 30, monthly month-end+15 / +45. Backward as-of joins, без backfill из будущего, со staleness caps. Проверено 0 `available_date > decision_date`.
- Протокол наследует ограничения старого решения: effective-date CBR вместо доказанного publication timestamp, latest snapshots, calibration и policy tuning на одном prior validation interval, post-selection/exploratory сравнение, reference-rate target вместо банковской котировки. Лаговый аудит проверяет механику join, а не полную историческую availability.

## Запуск

Из каталога `AlphaTransfer` с Python 3.11 и numpy/pandas/scikit-learn/pyarrow/openpyxl/xlrd/bs4:

```bash
python research_v3/external_data/fetch.py
python research_v3/external_data/experiment.py
python research_v3/external_data/fetch_cny_history.py
# После подготовки extended CBR panel основным models-исследованием:
python research_v3/external_data/long_combo.py
python research_v3/external_data/verify_and_summarize.py
```

Вся работа ограничена этим каталогом; `final_solution` не редактировался. Сеть нужна только двум fetch-скриптам. `experiment.py` ограничивает native threads единицей. `source_receipt.{json,csv}`, `experiment_manifest.json`, `verification.json`, `_SUCCESS.json` фиксируют входы, проверки и outputs. Сырые FRED/ICE/CBOE входы относятся к локальному исследовательскому сценарию и не являются автоматически разрешённым публичным набором для сдачи.

Импортируемый API находится в `benchmark.py`: импорт не скачивает данные, не запускает обучение и ничего не пишет. `load_frozen_panel()` читает panel, `augment_panel(panel)` добавляет Treasury lag7, `evaluate(...)` возвращает predictions и fold metrics, `score(...)` агрегирует прежние метрики. Например:

```python
from research_v3.external_data.benchmark import (
    load_frozen_panel, evaluate, score, BASIS_FEATURES, TREASURY_LAG7_FEATURES,
)
p = load_frozen_panel()
predictions, cells = evaluate(
    p, BASIS_FEATURES + TREASURY_LAG7_FEATURES, "treasury_lag7",
    train_years=2, years=(2023, 2024, 2025, 2026),
)
result = score(predictions[predictions.fold_test_year <= 2025],
               cells[cells.fold_test_year <= 2025])
```

Для root long-window harness нужен `disable_early_stopping=True`: автоматическое sklearn early stopping иначе меняет обучение после 10 000 train rows. API восстанавливает временные настройки evaluator после выполнения; для параллельных запусков применяются отдельные процессы.

**Что использовать дальше:** испытать прямой Treasury inflation lag7 совместно с более длинным train и проверить новую будущую выборку; исследовать timestamped CNY intraday и собственные банковские котировки, где решается реальная задача клиента. Добавлять все macro/news-признаки одновременно или покупать feed только из-за его известного бренда текущие цифры не поддерживают.
