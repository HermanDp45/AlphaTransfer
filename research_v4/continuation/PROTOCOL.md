# V4 continuation: protocol before new evaluation

This addition tests OXR on the available 2018-06-17–2026-09-02 data, 10-year NOW heads on Chronos forecasts, and robustness to bank-source latency. No pre-2018 OXR data are available. Original V3 and completed V4 experiment artifacts are read-only; their prior manifest is preserved here.

- Same h5 event and CBR reference labels. All comparisons pair identical corridor-dates. Brier primary, policy metrics separately; no interpretation as executable bank savings.
- Development remains 2023–2025. New candidate choices use development only and are saved before continuation 2026 score tables. The 2026 period was already inspected in V3/V4 and is a retrospective temporal test, not a pristine holdout.
- January freeze: model train before 2025, prior2025 calibration/policy tuning with h5 maturation purge, predictions starting2026. March freeze: prior12-month calibration endingMarch1, preceding10-year model train; no target requiring a rate at/after cutoff is used. Also compare both frozen versions on common March-onward dates.
- No claim that model weights are updated with test labels. Observed past rates may update rolling input features at each future decision. No outcome-dependent choice of model or calibration during test.
- Annual exact incumbent reproduction, forecast/checkpoint reconstruction, feature future-perturbation and label-maturity checks. Paired date-block intervals, not IID currency-row intervals. All intervals exploratory and not adjusted for the complete historical search.
- OXR max(published_at_utc, end of the completed UTC day) +24h is primary availability, +48h stress; join to decision10:05MSK and max7-day staleness. Preserve conservative lag and match currency/rate direction. No fallback to another provider under the OXR name.
- OXR predefined families: returns, basis, full; train24m/120m; two availability delays. Baseline, source-availability-only and shorter source-history controls included. Model capacity and prior-year cadence protocol frozen to V3. Candidate for2026 selected among120m primary-lag variants only by development Brier; non-winner controls retained.
- Foundation and robustness detailed protocols live in their subdirectories and are fixed before their new outcomes are evaluated.
