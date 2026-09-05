# Independent design and corrected-delay audit

**All inspected actionable findings are addressed.** The initial design audit passed 45 checks; the final canonical-control and simultaneous-interval audit passed another 25, with zero failures. [Initial verification](design_verification.json), [final verification](final_verification.json), [initial reproducer](verify_design.py), [final reproducer](verify_final.py). These verifiers perform no fitting, make no network requests, and write only in `data_audit`.

## Addressed: rolling feature state at delay onset

The original fixed-model stress joined a delayed source over the whole history, including dates before the declared onset. Consequently, post-as-of OXR basis differences/z-scores and bank/OXR premium differences used a counterfactual delayed prehistory. This was a scenario-definition error, not exposure to future labels. At January 2023, basis z20 differed on the first 19 dates in each corridor, with maximum absolute difference 0.7798; March 2026's maximum was 0.6467.

The root engine now splices normal observations before the cutoff and delayed observations from the cutoff forward, then recomputes only these derived rolling features. Native trailing source returns remain unchanged. The repair replays saved checkpoints without refitting and preserves the original selection and normal contact state.

An independent online buffer implementation verified all 30 combinations of two source starts, five cutoffs and three delayed scenarios. Pre-cutoff frames match exactly. The largest discrepancy between direct buffer arithmetic and the corrected rolling implementation is `3.855e-13`.

For all saved normal views, before/after repair probabilities, raw probabilities, candidate masks, portfolio masks, labels and forward outcomes are **exactly identical**: 50,250 row evaluations in the main branch and 22,110 in the Treasury branch. All 120 checkpoint and updated prediction hashes match their receipts, and both original selection files still reference the retained original development files. These are repeated row evaluations across configurations/cutoffs, not independent observations.

## Historical controls

The following normal controls reproduce previous saved results exactly across 2023–2025 and the January 2026 freeze:

| New control | Previous frozen reference | Rows | Maximum probability/signal/target/forward difference |
|---|---|---:|---:|
| v3_120m | V3 basis_train_120m | 4,415 | 0 |
| oxr_basis_2018_120m | Previous OXR primary basis120m | 4,415 | 0 |
| kzt_local_120m | V4 KZT pooled120m | 883 | 0 |
| kzt_shrink_120m | V4 KZT residual_shrink120m | 883 | 0 |
| halyk_local_120m | V4 KZT pooled120m Halyk lag1 | 883 | 0 |
| halyk_shrink_120m | V4 KZT residual_shrink120m Halyk lag1 | 883 | 0 |

Foundation's `synth_no_oxr` and `small_no_oxr` also exactly reproduce their previous exact-V3-extended foundation references on 3,635 development rows each. Static inspection found no new discrepancy in the declared pooled-fit/KZT-only-calibration scope, mature-label split, frozen forecast cache, currency units or explicit OXR/Halyk lags. The later lag1 diagnostic configuration addendum is separate from the original protocol; its provenance was flagged to the foundation author for inclusion in final receipts.

## Separate BANK SELL target

The bank-target design correctly states its distinct direction: KZT paid to buy RUB at a BANK SELL quote. It does not call this a RUB-to-KZT transfer receipt. Its target averages the next five actual clean bank observations, excludes windows exceeding 14 calendar days, treats ties as zero, and requires each training/calibration label to mature before the next boundary. The code reconstructs local effective dates from both ISO and epoch timestamps, uses trailing-only features, and converts CBR/OXR to KZT per RUB.

Bank effective date+1 is explicitly an assumed availability rule; neither verified publication timestamps nor execution at the last observed quote are established. Since bank labels start in 2020, OXR2010 and OXR2018 must give identical bank-target features after warmup; the implementation checks that identity. No actionable design error was found in this read-only inspection. The bank-target author's fuller executable verification remains separate evidence.

## Addressed: common-feature rounding and attribution

The 2010 and 2018 OXR sources overlap exactly, but pandas' incremental rolling arithmetic gives tiny post-warmup feature differences. For D+2 after 2019: maximum `oxr_vol20` difference `8.856e-16`, maximum `oxr_basis_z20` difference `1.637e-12`. For D+3 the maximum z20 difference is `2.137e-12`. All other OXR returns, basis differences, coverage/age and log levels are exact.

Swapping only these two common post-2019 columns while keeping each fitted model fixed produced zero bin/probability changes on 37,479 calibration and 31,155 test row evaluations. On 325,655 training row evaluations, however, 3,333 changed the fitted pooled HGB bin assignment and 34 changed the final raw probability by more than `1e-12`, with maximum difference 0.090814. Rows are repeated across checkpoints; these counts are not unique dates. [Routing evidence](common_feature_roundoff_routing.csv).

The swap exposed potential sensitivity in the training history and motivated a separate numerical attribution control. It did not itself demonstrate that the measured source-extension result was caused by rounding.

The root branch then declared and fitted seven canonical configurations across five cutoffs, retaining their unchanged 2018-history references. In all four OXR-delay/bank-lag source views, OXR features from 2018-09-01 onward are copied from the 2018 panel into the 2010 panel; earlier rows remain exact. At the first common row, the shortest source history already contains 74 calendar observations and the shortest CBR basis history contains 54 observations, exceeding both complete 20-observation windows. This boundary is therefore past warmup, rather than a truncation of meaningful initial features.

The independent final verifier reconstructed all 35 canonical training feature fingerprints, verified their source/checkpoint/prediction hashes and strict label maturities, and compared the complete saved predictions against the original counterparts. **All 50,250 model/view row evaluations are exactly equal** in raw and calibrated probabilities, candidates, portfolio signals, targets and forward outcomes. No refitting was performed by this auditor. The previous attribution concern is closed for these seven explicitly tested configurations, including the primary history-extension controls; this is not a claim that arbitrary future models are immune to floating-point effects. The original routing diagnostic and the post-readout nature of this additional control remain documented.

## Simultaneous-interval implementation

The current normal-view HGB family contains 27 configurations including the V3 long-history reference, hence 26 paired contrasts. All share the same KZT dates and targets within each cutoff. The implementation resamples common blocks across models, separately within each year, and centers the paired Brier differences before taking the maximum absolute standardized bootstrap statistic. It uses the resulting 95th-percentile maximum to widen each marginal interval. This accounts for dependence among the contrasts within the stated family.

An independent direct-index implementation reproduced all 12 track/block combinations: development, January 2026, and common-date January/March views, each with calendar-month, 20-date and 60-date blocks. It used sampled block sums directly, instead of the root implementation's weight matrix. The largest difference across estimated effects, standard errors, critical values and interval endpoints was `3.997e-15` over 10,000 replicates per combination.

These are approximate, fixed-model simultaneous sampling bands for the declared normal-view HGB/KZT family. They do not cover all previous V3/V4 or neural experiments, task/metric selection, latency-scenario selection, or uncertainty from retraining models. They are not a new confirmatory test on uninspected 2026 data. In particular, an unadjusted negative interval for Treasury+Halyk should not be described as established robust superiority when its family-adjusted development interval and its January 2026 20-date interval include zero.

Correct-side BUY history feasibility is documented separately in [halyk_buy_feasibility/REPORT.md](halyk_buy_feasibility/REPORT.md).

The final audit handoff is recorded in [final_handoff_receipt.json](final_handoff_receipt.json). Prior sealed research and `final_solution` were not changed by this audit.
