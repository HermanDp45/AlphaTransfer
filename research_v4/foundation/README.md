# Foundation model benchmark

This directory owns the V4 pretrained numerical time-series experiment. It reads frozen V3 data/code and writes only here. `REPORT.md` contains the completed numerical results; `SOURCES.md` audits models, weights, licenses and contamination.

## Reproduce

Use Python3.11 and install `requirements.txt`. The original run used `/private/tmp/alphatransfer-tabm-venv/bin/python`, CPU2 threads. The public pretrained weights require about588MB. All inference/training executes offline after the model download.

```sh
python fetch_models.py
HF_HOME=/private/tmp/alphatransfer-hf HF_HUB_OFFLINE=1 python benchmark.py
HF_HOME=/private/tmp/alphatransfer-hf HF_HUB_OFFLINE=1 python assess.py
```

Pinned checkpoints are downloaded into `/private/tmp/alphatransfer-hf`; the code loads their explicit snapshot paths. No authentication token is required. Changing the cache directory requires updating `HF_CACHE` in `benchmark.py`. Downloading model artifacts is the only network step and reads no case data.

The source data are `research_v3/models/data/cbr_extended.csv`, public CBR nominal-normalized RUB per foreign-currency unit. Frozen V3 panel data are read for target alignment, rate-only baseline features, and the existing CNY/RUB basis feature in the HGB augmentation experiments. No FRED/Bloomberg field enters a foundation-model fit.

## Frozen evaluation design

- Target: NOW iff every one of the next5 CBR observations has rate at least today's rate. Rates are RUB/unit: smaller is favorable for buying the foreign currency with RUB.
- Outer2023–25 are retrospective development;2026 is a previously inspected diagnostic partial year. No year is relabeled pristine holdout.
- Backbone inputs: last256 log-rate observations, five currencies at the same decision date. Future target values are masked internally. Batches contain separate task groups; `cross_learning=False` prevents cross-date leakage.
- Zero-shot backbone weights remain fixed. A learned NOW adapter means the full decision system is **not** zero-shot.
- Pooled FT: actual full-weight AdamW,300 steps, batch20 target series/four5-currency tasks, learning rate1e-5 with linear decay, context256, minpast64. The available training interval is10 calendar years ending before the preceding calibration year. This is bounded stochastic training, not a claim of convergence or exhaustive window visitation.
- KZT stage2: starts from the pooled Synth checkpoint; full-weight100 additional steps on the same allowed dates of KZT only, learning rate2e-6. Inference still uses the identical five-currency same-date group, and the event-head training protocol stays pooled. Only KZT results of this stage are promoted to the summary.
- Event adapter: logistic regression C=.1, forecast-only14 summaries and corridor; or existing HGB architecture on base/basis features plus those summaries. Both fit on the same purged2-year train rows as V3. FT backbone and adapter share train data; this is ordinary supervised head fitting, and may overfit. No calibration/test labels train the backbone/head.
- Platt fit and frequency thresholds use the purged preceding year, as V3. Cooldown is replayed over its complete score history including the unlabeled tail, then carried into test.
- Training, calibration and complete outer-year label sets remove their last5 rows per corridor when required. Test pairs and labels must exactly match both V3 incumbents.

## Scores and artifacts

`forecast_cells.csv` scores untouched marginal distributions: mean pinball loss across the common13-quantile grid and5 horizons, median MAE, and80% interval coverage. Loss units are **log-rate basis points**, not realized transfer savings. Quantiles are monotonically rearranged before evaluation and adapter construction. The two forecasting baselines are no-drift Gaussian random walk with trailing60-return volatility, and trailing256-observation empirical5-step paths.

`summary.csv` and `cells.csv` score NOW Brier, logloss, AUC, fixed10-bin calibration error and actual candidate policies. `paired_intervals.csv` uses10,000 year-stratified whole-month resamples with all currencies together. Candidate and incumbent policies retain their respective signals; every draw recomputes each year×corridor random-day baseline and signal-count weights. These are exploratory conditional-model intervals, not multiple-comparison-adjusted evidence of an unseen-market edge.

`checkpoints/*/*/fit_receipt.json` records dates, hyperparameters, weight hashes and actual parameter changes. Safetensors and config files are real locally trained models. `forecasts/*.npz` stores raw quantile forecasts for independent re-analysis; `output/*_event_adapter.joblib` stores trained decision heads. `causality_checks.json` verifies that perturbing a later independent group does not change an earlier prediction beyond floating-point tolerance.

Memory and storage: the two downloaded base models total588MB; retaining all annual pooled and KZT full checkpoints occupies several GB. This preserves reproducibility without re-running the neural fits. The main run used a single2-thread process; MPS was unavailable. CPU timing is recorded per fit and forecast cache.
