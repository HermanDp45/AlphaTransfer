# Independent OXR implementation audit

**Result: 124 checks passed, 0 failed. No actionable implementation defect found.** This audit read the final 95 checkpoints and independently replayed 11 representative checkpoints. It did not modify `experiment.py`, `assess.py`, the source snapshot, predictions, metrics or sealed V3/V4 artifacts. No tree model was fitted; the 11 Platt fits only reproduced the already saved calibration from its original validation data.

Machine-readable evidence: [verification.json](verification.json). Reproducible implementation: [verify.py](verify.py). Execution log: [verify.log](verify.log). The separate [source audit](../source_audit.md) documents official publication semantics and the remaining vintage limitation.

## Exact baseline identity

| Frozen reference | OXR-run baseline | Period | Matched rows | Maximum probability difference | Candidate/portfolio signal differences |
|---|---|---|---:|---:|---:|
| V3 `baseline_reproduction_h5` | `v3_24m` | 2023–2025 | 3,635 | 0 | 0 |
| V3 `basis_train_120m_h5` | `v3_120m` | 2023–2025 | 3,635 | 0 | 0 |
| V3 `baseline_reproduction_h5` | `v3_24m` | January 2026 freeze | 780 | 0 | 0 |
| V3 `basis_train_120m_h5` | `v3_120m` | January 2026 freeze | 780 | 0 | 0 |

Targets and forward-return proxy values are also exactly identical. The 24-month variant correctly uses `panel_v2.pkl`, preserving the original feature warmup; the 120-month variant uses `panel_extended.pkl`. March 2026 is a new explicit checkpoint, compared on its own 610 target rows; it is not relabelled as the original annual model. Within every cutoff, all 19 configurations share exactly the same date/currency/target/forward-value rows.

## Timing and labels

The implementation uses `max(published_at_utc, UTC date + 1 day) + 24h/48h`, then backward as-of at 10:05 Moscow time. This implements the recommended D+2/D+3 decision eligibility. The seven-day as-of tolerance is measured from the buffered availability time; `oxr_age_days` is measured from publication. Neither is a count of CBR rows.

OXR returns are computed in source calendar-day order. Basis differences and basis rolling statistics are computed after alignment, in CBR observations. The code explicitly uses a 20-observation window with a 10-observation minimum for volatility/z-scores; the 20-day return still requires 20 prior source observations. These are different feature definitions and should retain their respective units in descriptions.

All 95 train/calibration splits were reconstructed from the cached target panel. Every fifth future observation used by a training label precedes validation start; every fifth future observation used by calibration precedes the exact January/March checkpoint. All saved prediction rows agree with the reconstructed eligible test rows. Model inputs exclude target, forward return, regret and symmetric-return columns. Code inspection confirms the imputer/scaler are fitted through the model pipeline on the training fold.

Four adversarial source checks multiplied values whose buffered availability lay in the future, using both 24h/48h buffers at 2023-01-01 and 2026-03-01. Every earlier OXR feature, including the checkpoint day's features, remained exactly unchanged. Three target checks altered all official rates from the 2023, January 2026 and March 2026 cutoffs forward: all retained training and calibration labels remained exactly unchanged after regeneration.

Cooldown history uses only probabilities, dates, corridor and session ordinals. Poisoning its target/return fields did not change selections. Its warm start is a reconstruction under the checkpoint's frozen policy, consistent with the reference benchmark; it is not evidence of actual historical customer contacts.

## Checkpoint and decision replay

The sample includes both 2023 baselines, the primary basis model in 2023/2025/January 2026/March 2026, a 48h full-feature model, two full-family history controls, and the two later basis-family history controls. All saved raw/calibrated probabilities reproduce within **1.12e-16**. Independently refitted Platt calibration reproduces calibrated probabilities with **zero difference**. Per-corridor thresholds, candidate masks and portfolio masks reproduce exactly.

All 95 source/checkpoint/prediction hashes and all 95 feature-frame fingerprints match the current files. Importing the experiment and assessment modules caused no file mutations. `selection.json` agrees with the development prediction hash and the independently recalculated 2023–2025 minimum Brier among the three declared 120-month primary-delay families. The extra ten basis-history fits have the explicit `exploratory_followup_after_initial_2026_readout` status and do not change that selection.

## Metrics and intervals

Independent point-score calculations agree for Brier, candidate lift and candidate forward-return delta on all 95 checkpoints, for both the five-currency scope and KZT. The reference candidate baseline is estimated within year/currency cells and weighted by candidate exposure, matching the original benchmark. These remain official-rate target/return proxies, not realized transfer savings.

The monthly paired bootstrap resamples common time blocks across currencies and stratifies by year. Both policies share each draw. For each draw, the implementation recomputes the year/currency target rate and forward-return baseline before calculating candidate lift and forward delta. An independent direct row-replication implementation with 128 deterministic verification draws matches its tensor-based interval endpoints: maximum Brier endpoint difference **2.41e-17**, lift **1.81e-16**, forward delta **4.67e-15 bps**. The production assessment uses 10,000 draws; the smaller count is only an independent arithmetic check.

The 20/60-date variants are non-overlapping blocks split at year boundaries, rather than a moving-block bootstrap. They retain common dates across currencies and estimate Brier uncertainty only. The January/March comparison intersects target dates before pairing. Source-history comparisons hold family and training-window length constant while changing source start; the later basis controls are correctly marked post-readout sensitivity analyses.

## Interpretation and remaining limitations

No statistical promotion follows from passing software verification. These intervals condition on saved predictions and policies; they do not include model-selection uncertainty, retraining uncertainty or a correction across the many explored configurations. The small number of 2026 calendar blocks limits interval precision. Known 2023–2026 history remains retrospective/exploratory even when a particular new selection file was written before reading its new test table.

Historical source vintages and retrieval timestamps were not archived, so the conservative publication-lag model cannot prove revision-free historical availability. The reference rates have no execution costs or bank spreads. Data absent before June 2018 remain absent; results from a 2018/2020/2022 truncation experiment cannot quantify the benefit of buying 2010–2018 history.

**Open implementation findings: none.** Remaining items are documented research and interpretation limits, not blockers requiring code edits.

Reproduce from the `AlphaTransfer` directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /private/tmp/alphatransfer-ml.Ci5ycv/venv/bin/python \
  research_v4/continuation/oxr/audit/verify.py
```

Final audited code SHA-256 values:

- `experiment.py`: `d98c009be6d119929a63bd70f0f771df05157551e8b445770c6b6e0318c8487f`
- `assess.py`: `c341ec30e51a8d8789ab5e8c9b45d271005af78042315ba9ad03521cdce7065f`
- `verify.py`: `ea565280819ac2895b9553e257a561be4903e3954db26ffa45276e17ebf988aa`

The verifier ran with pandas 3.0.5 and one computational thread.
