# Exact original V3, annual H3/H5 2024–2026

Six separately trained pooled-five-corridor HGB models. Read [REPORT.md](REPORT.md) for exact parity and the distinction between H3 training and frozen-H5-policy H3 rescoring.

From AlphaTransfer root:

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python research_v4/robust_selection/v3/experiment.py
PYTHONDONTWRITEBYTECODE=1 /private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python research_v4/robust_selection/v3/diagnostics.py
PYTHONDONTWRITEBYTECODE=1 /private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python research_v4/robust_selection/v3/verify.py
PYTHONDONTWRITEBYTECODE=1 /private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python research_v4/robust_selection/v3/report.py
```

The six HGB fits are fast and always run afresh. No unverified cache is reused. All writes stay in this new directory. Runtime: Python 3.14.7, sklearn 1.9.0, pandas 3.0.5; one CPU thread.

## Root input contract

`raw_predictions.csv.gz`: one row per `(config_id, train_horizon, fold_test_year, split, date, corridor)`.

- `config_id=v3`; `train_horizon` is 3 or 5; `fold_test_year` is 2024/2025/2026; `cutoff` is annual Jan 1.
- `split`: `validation`, `history`, `test`. History overlaps validation intentionally and also includes its purged immature tail.
- KEEP: `date,corridor,target,forward_bps,symmetric_bps,regret_bps,session_ordinal,label_available_date,rub_per_unit,ret1,pr60`.
- `raw_probability`: original HGB score; `probability` and `original_v3_probability`: exact original pooled-five-corridor calibration. Never silently substitute local corridor calibration when evaluating the strict original 0.5 baseline.
- `candidate_signal`, `signal`: historical original tuned policy and shared portfolio; aliases `original_candidate_signal`, `original_signal`.
- `strict05_candidate_signal`, `strict05_signal`: original calibrated 0.5 threshold plus cooldown3, with prior-history initialization; shared portfolio explicitly separate from per-corridor candidates.
- Immature history outcomes are masked. Test outcomes are offline evaluation fields and must not enter policy selection.

`warmup.csv.gz`: split=`warmup`, exactly 63 last full-panel dates strictly before calibration start, including the training purge tail. All outcomes masked; same checkpoint and original pooled calibration. Per-row `in_sample_training_warmup` identifies overlap with fitted training rows. Warmup is optional policy-state initialization, not OOT validation. The original V3 historical parity does not consume it.

H3 and H5 have different test tail lengths. Use `matched_horizon_rescore.csv` or explicitly intersect `(date,corridor,fold_test_year)` for cross-horizon comparisons. Original metrics retain each horizon's historical test set. Lift/forward delta aggregate with candidate-count weights over year/corridor cells; raw pooled alternatives are explicitly named in rescoring diagnostics.

## Evidence

- `historical_parity.json`: raw probabilities, original calibration, all horizon-specific utility fields and candidate/portfolio flags versus sealed historical V3 predictions.
- `model_receipts.json`, per-fold receipts and `checkpoints/`: actual model artifacts, training fingerprint, maturity dates and original policies.
- `colleague_h3_reproduction.csv/json`: exact 39/60 H3 hits from frozen H5 strict signals over 727 KZT dates in 2023–2025; not an H3 model fit.
- `verification.json`: independent checkpoint, temporal, utility and scheduler checks.
- `artifact_manifest.json`: SHA seal of this package. Source V3/V4 files remain unchanged.
