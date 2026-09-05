# NOW + CLOSING research product profile

[REPORT.md](REPORT.md) contains the final results; [SCENARIO_CONTRACT.md](SCENARIO_CONTRACT.md) separates the two targets, UI facts and integration assumptions.

Default research profile: **NOW remains the primary claim on every existing contact; CLOSING is an optional secondary annotation with its own model probability and endpoint target. No extra notifications.** Aggressive joint scheduling and exclusive routing are preserved as diagnostics, not silently promoted.

Run from the AlphaTransfer repository root using Python with pandas/numpy/scikit-learn/scipy/pyarrow/threadpoolctl (the experiment used `/private/tmp/alphatransfer-tabm-venv/bin/python`):

```bash
python research_v4/final_sprint/product/closing_experiment.py
python research_v4/final_sprint/product/joint_evaluate.py
python research_v4/final_sprint/product/verify.py
python -m unittest discover -s research_v4/final_sprint/product -p 'test_scenario_adapter.py'
python research_v4/final_sprint/product/build_report.py --export-adapter-config --seal
```

The report export writes only the explicitly agreed new `final_solution/final_sprint/closing_annotation.json`; legacy product.py is untouched. Omit `--export-adapter-config` for a local-only report. `joint_evaluate.py` accepts `--selection`, `--now-predictions`, `--now-history`, `--output-dir`; all inputs are snapshotted for reproducibility. The selected NOW model is inherited from root's explicit retrospective ranking, not selected afresh by this package.

Key files:

- `results/closing_predictions.csv.gz`, `closing_history.csv.gz`: exact date/corridor scores, separate NOW/CLOSING labels, mean-forward and endpoint bps; all unmatured history outcomes masked.
- `results/closing_metrics.csv`, `model_receipts.json`, `checkpoints/`: four fixed CLOSING fits and standalone diagnostics.
- `results/joint/metrics.csv`, `annotations_metrics.csv`, `annotation_month_intervals.csv`: all joint variants and the default nonexclusive profile.
- `results/joint/input_snapshot/`, `policies.json`, `manifest.json`: final selected NOW inputs and past-fitted thresholds.
- `results/joint_catboost_control/`: preserved earlier CatBoost results, input snapshots and engine source.
- `scenario_adapter.py`, `scenario_schema.json`, `test_scenario_adapter.py`: working stdlib proposal API and11 meaningful tests.
- `results/example_dual_input.json`, `example_dual_preview.json`, `closing_annotation.json`: runnable sample and integration config.
- `results/verification.json`, `artifact_manifest.json`: numeric/temporal checks and hashes.

CLOSING head uses `R[t+5]>R[t]` with R=RUB per recipient currency unit, h=5 effective CBR observations, tau=0. The Halyk BANK SELL quote is a feature, never the target or an executed RUB→KZT price. Fresh bank data is the user's final deployment assumption; delay remains a diagnostic, not a selection gate. Both the root ranking and later annotation design inspected2026; point-estimate success is retrospective, not a prospective guarantee.
