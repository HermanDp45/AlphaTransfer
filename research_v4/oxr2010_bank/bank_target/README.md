# Actual Halyk BANK SELL RUB prediction

[REPORT.md](REPORT.md) explains the findings and limitations in Russian. [PROTOCOL.md](PROTOCOL.md) specifies the primary future-mean target; [NOW_ADDENDUM.md](NOW_ADDENDUM.md) specifies the explicitly post-readout NOW-event follow-up.

**q is KZT paid per RUB purchased from the bank. Lower is better for the RUB buyer. This is not the BANK BUY RUB side needed for the original RUB→KZT remittance product.** The archive anchor is only assumed available the following day; its executability then is unknown.

Run from the AlphaTransfer root:

```bash
python research_v4/oxr2010_bank/bank_target/experiment.py
python research_v4/oxr2010_bank/bank_target/now_followup.py
python research_v4/oxr2010_bank/bank_target/verify.py
python research_v4/oxr2010_bank/bank_target/build_report.py
```

Python used: `/private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python`, with numpy/pandas/scipy/scikit-learn/threadpoolctl. One CPU thread. Inputs are existing immutable Halyk raw/clean quotes, the CBR panel, and the expanded OXR snapshot in the parent folder. Only this folder receives writes. No API or network access is needed.

Key outputs:

- `results/development_summary.csv`, `results/common_march5_summary.csv`: primary task, all fixed arms on identical bank dates.
- `results/now/development_summary.csv`, `results/now/common_march5_summary.csv`: additional NOW-event classification.
- `results/paired_month_intervals.csv`: paired calendar-month bootstrap for both tasks.
- `results/source_audit.json`, `results/coverage_by_year.csv`, `results/timeline.csv`: source cleaning, maturity and coverage.
- `results/source_start_control.json`: source2010/2018 feature identity; independent verification also retrains and compares predictions for both starts.
- `results/checkpoints/`, `results/now/checkpoints/`: fitted models and past-only calibrators.
- `results/model_receipts.json`, `results/now/model_receipts.json`, `results/verification.json`: SHA, fitting windows, parity and future-poison checks.
- `artifact_manifest.json`: local hashes, written last by the report builder.

The 14-day future-window density filter is an ex-post evaluation condition, not an online eligibility rule. Three exact decimal ties were corrected after independent verification caught a floating-point comparison bug. Negative findings are preserved. Older OXR does not manufacture pre-2020 bank labels. This package does not establish causal client uplift or executed bank P&L.
