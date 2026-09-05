# V4 segment market-policy experiment

[REPORT.md](REPORT.md) содержит метод, отрицательные и положительные результаты, пределы источников и продуктовые решения. Вся ценность — **сценарная**, не причинный business uplift. Default preview: **universal + V3 readiness**. Group policy экспериментальна и проиграла в основной взвешенной смеси.

## Воспроизведение

Из корня `AlphaTransfer` (нужны numpy/pandas; 1 CPU thread, несколько минут):

```bash
python research_v4/segments/experiment.py
python research_v4/segments/build_report.py
python -m unittest discover -s research_v4/segments -p test_segments.py -v
```

Использованный runtime: `/private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python`. Дефолт: training35users/segment, seed7101, evaluation60users/segment×3seeds8101–8103. H1/3/5/10/20 scores `research_v3/models/baseline_reproduction_*` неизменяемы. SHA256 в `results/manifest.json`.

Microdata raw не включены в repository. Для воспроизведения агрегатов откройте [официальные условия WB](https://microdata.worldbank.org/catalog/6523/get-microdata), примите исследовательские условия и сохраните ZIP вне repository:

```bash
python research_v4/segments/fetch_microdata.py --accept-research-terms --output /private/tmp/at-kg-microdata.zip
python research_v4/segments/aggregate_microdata.py --zip /private/tmp/at-kg-microdata.zip
```

Скачивание через штатный Accept flow, без входа/обхода ограничений. Raw сохранён только в `/private/tmp`. На диск исследования записываются агрегаты без household/member IDs. Download hash: `94708c35092c33c77907c3652ab68db237d57176d4ef34c7f441597d518d6400`. Если WB обновит ZIP или access flow, остановитесь и проверьте версию/метаданные; не подменяйте архив молча. Публичные отчёты/metadata: `fetch_sources.py`. IOM PDF403 отражён в download manifest, данные оттуда используются только как помеченный индексированный sensitivity.

## Артефакты

- `data/l2kgz_*`: реально посчитанные месячные агрегаты, regularity bands, transitions, methods, provenance; `data/raw/wb_rpp_baseline2016.pdf` — скачанный официальный первичный отчёт.
- `results/selected_policy_receipts.json`: все fit years и пороги;2024fit2023,2025fit2023–24,2026diagnosticfit2023–25.
- `results/prior_policy_frontier.csv`: полный prior grid60configs×3segments/all, без выбора лучшего на test.
- `results/weighted_development.csv`, `weighted_by_year.csv`, `by_segment_development.csv`: фиксированные веса и ratios из взвешенных числителей/знаменателей.
- `results/frequent_response_cost_sensitivity.csv`, `frequent_market_support.csv`: цена частоты для frequent, actual unique market-date support.
- `results/mixture_sensitivity.csv`, `corridor_year_diagnostics.csv`, `market_block_diagnostic.json`: переносимость и реальные рыночные единицы неопределённости.
- `results/client_outcomes.csv.gz`, `base_contacts.csv.gz`: **синтетические** клиентские выходы; не банк-транзакции и не новые независимые рынки.
- `artifact_manifest.json`: hashes code/report/tables. Пересоздаётся `build_report.py` после завершения изменений.

## Preview

```bash
python -m research_v4.segments.preview --context research_v4/segments/examples/context.json --scores research_v4/segments/examples/scores.json --as-of 2026-08-04
```

`--policy group_aware` включает экспериментальную адаптацию, `group_unconstrained_exploratory` — дополнительную проверенную гипотезу. JSON контекста в примере synthetic. Рыночные вероятности в примере взяты из frozen scores; для bank_observed контекста production всё равно запрещён. API `build_segment_policy_preview(context, market_scores, as_of, receipt, mode='universal')` использует проверенный V3 behavior API по импорту, возвращает причины отказа, horizon/threshold/cadence, **не меняет вероятность и ничего не отправляет**. Форматы контекста и scores см. примеры; `last_market_candidate_at` — последняя рассмотренная рыночная возможность, а не контакт. Это соответствует cadence до readiness в симуляции.

Политики настроены ретроспективно на уже просмотренных рынках; 2026 diagnostic. Нужны future prospective validation, real quotes и CRM holdout перед production.
