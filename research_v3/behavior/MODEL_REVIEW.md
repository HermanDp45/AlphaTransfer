# Независимое ревью V3 models/experiment.py и интеграции

Проверка 2026-09-05; чтение кода и сохранённых результатов. Основной код моделей не менялся этим ревью. Findings переданы владельцу `/root`, который исправляет provenance/metrics/integration.

## Подтверждено

- Baseline reproduction h=5: 3 635 development date×corridor rows совпадают с исходным `hgb_plus_cnyrub_basis`; max absolute probability delta = **0**, candidate disagreements = **0**.
- Все находившиеся на момент проверки h5 prediction-файлы имеют одинаковый development date×corridor set.
- `temporal_split` хронологически разделяет train/calibration/test и purges последние h строк на коридор у train и calibration. Очевидного доступа к test labels при fit, calibration или threshold choice не обнаружено.
- Survival factorization использует future outcomes только как train labels; realized day-1 outcome не поступает в inference.
- Utility objectives используют train-only targets; imputation/scaling обучаются внутри Pipeline на train. Cross-sectional common-factor target агрегируется только по train dates.
- Monthly scoring сохраняет полный месячный поток до сведения к совместным annual evaluation dates; месячное непуржирование test-tail само по себе не является утечкой, поскольку последующее включение labels в fit защищено новым purge.
- `forward_delta_bps` соответствует signal-count-weighted fold×corridor standardization. Brier/log-loss считаются на полном совместном score grid.

## Найдено и передано владельцу

1. **Coverage denominator.** `summarize()` использовал legacy `core.cadence_diagnostics`, исключающий boundary weeks. Это допустимо как точная реплика V2, но должно называться legacy full-week coverage. Для operational claim нужны отдельные observed-boundary-week показатели, а unlabeled tail нельзя объявлять измеренным. Проверенный пример: annual_recent_calibration_3m legacy mean/min 88.37%/77.55%; all observed weeks 86.48%/74.51%. Владелец добавляет separate metrics, сохраняя legacy comparability.
2. **Cache provenance.** Panel pickle и model predictions раньше принимались по `exists()`; spec/source/code fingerprints не проверялись, а протокол перезаписывался. При изменении кода это может смешать stale outputs и новый receipt. Владелец добавляет hashes и пересобирает результаты.
3. **Interpretation of shorter validation.** `validation_months` меняет не только объём calibration: train-end сдвигается ближе к test и вместе с ним меняется train window. `annual_recent_calibration_3m` нельзя интерпретировать как чистый calibration-only ablation; это совместное обновление recency/calibration. Утечки в этом нет.
4. **Optional behavior integration.** Условие `preview_ready is False` не добавляло suppression для rejected/unavailable (`None`) и для готовой simulation. В historical API `eligible_to_send=False` оставался безопасным, но production-like product с будущим promoted scorecard мог пропустить supplied invalid/synthetic context. Нужны отдельные reasons по status/source; отсутствие optional context можно оставить disabled. Повторено: future context даёт behavior.status=rejected без behavior reasons в общем suppression. Владелец исправляет.
5. **Empty relevant corridors.** `set(corridors or CURRENCY_COPY)` превращает явный `[]` во все направления. Подтверждено: вызов `decision(..., corridors=[])` вернул пять строк. Нужна проверка `None` отдельно от пустого списка. Валидация threshold должна жить и в callable API, не только в CLI.

Никакого статуса подтверждённого model/behavior uplift это ревью не создаёт. 2026 остаётся inspected diagnostic; synthetic client N не увеличивает число market regimes.
