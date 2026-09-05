# Independent audit plan: architecture comparison, 2023–2025

This is a read-only audit of the new branch. No model fits, source downloads, or changes to sealed prior experiments are authorized for the auditor. Existing selected TabM architecture/seed were informed by prior 2025/2026 research, so this comparison is retrospective even when every forecast is temporally causal.

## Matched experimental contract

- Main factorial: TabM versus HGB, each with BASE15 and full33, KZT-only training/calibration/evaluation, the same prior120 months of training followed by12 months of calibration, CBR NOW h5, and identical train/validation/test index sets and labels.
- The shared preprocessing must be fitted only on outer train: median imputation, Gaussian quantile transform, deterministic per-feature missing indicators. Verify transformed feature order, missingness, numeric dtype and values. The neural inner epoch-selection preprocessor uses only inner train; its final refit preprocessor uses the full outer train. HGB must not gain pooled-corridor examples, sample weights, residual adapters or extra source columns in the pure architecture contrast.
- Reusing2025 checkpoints requires exact original train-row and feature fingerprints, matching feature lists, preprocessing hashes and saved model parameters. Byte hashes alone do not prove the model matches the newly assembled feature panel. Build from frozen binary panels, or use explicit round-trip CSV parsing; prior tests found ordinary CSV parsing can move quantile-transformed features substantially despite1-ULP input changes.
- Seed2 is the declared main comparison. Seeds0/1 measure sensitivity of full33 TabM, not independent historical observations. An ensemble must average raw predictions and calibrate on the same preceding year, without choosing weights on the evaluated year.
- Historical V3 pooled BASE15 remains a separate system control. It differs in training scope/preprocessing/calibration and cannot identify a pure architecture effect.

## Point-in-time and state checks

- Reconstruct the actual fifth-future observation date. Every outer training label must mature before calibration starts; every calibration label before the annual prediction cutoff. Check the inner neural split separately.
- Check that no evaluated labels, future rows or future feature perturbations affect fitted preprocessing, chosen epochs, Platt calibration, policy thresholds or initial contact state.
- Keep the prior-year history through the final pre-cutoff observation for cooldown/contact state; mask unready history outcomes. Replay saved raw/calibrated probabilities and selected actions from checkpoints without tree/neural fits.
- Reconstruct OXR, Halyk and Treasury availability assumptions through the already audited frozen source builder. No backfilling of pre-source-history gaps. The full33/base15 factorial changes the feature set, not the label grid.
- Test each annual contact stream independently with its own historical state. The threshold0.5 control must declare whether it acts on calibrated or raw probability; use the same cooldown/week-cap rules and correctly initialized history for both architectures.

## Multi-year aggregation

For a cell g=(fold year,corridor), let n_g be eligible dates, s_g selected dates, p_g the eligible event rate, and u_g the eligible mean forward outcome. Aggregate Brier over all eligible rows. Aggregate policy lift as:

`sum(selected successes) / sum_g(s_g * p_g)`.

Aggregate forward advantage as:

`sum_selected(forward - u_g) / sum_g(s_g)`.

Do not average annual lifts or divide all selected hits by an unrelated unweighted annual prevalence. Report raw selected hit rate and its selection-exposure-weighted baseline separately. Zero-signal and degenerate-baseline cells should be explicit rather than silently removed.

Construct the full calendar-week range separately inside each evaluated year/corridor, including partial boundaries and weeks with zero contacts. Sum compliant-week counts and denominators afterward. A continuous2023–2025 range incorrectly creates extra zero weeks across the annual evaluation gaps. Expected frozen KZT cohorts are recorded in [expected_cohorts.json](expected_cohorts.json).

For uncertainty, resample the same calendar-month blocks jointly for both models, stratified by fold year. Recompute cell prevalence and baseline forward means inside every draw. Compare paired Brier and policy differences on identical rows; retain temporal dependence and all models' paired alignment. Fixed-prediction intervals do not account for the earlier selection of architecture, seed or metrics.

## Final receipt

The executable audit should record exact source/code/checkpoint hashes; matching annual cohorts; feature/preprocessor fingerprints; label-purge assertions; historical2025 reproduction; checkpoint/Platt/policy replay; independently reconstructed point metrics, year-separated weekly counts and paired bootstrap arithmetic. Clearly separate new trained models, reused checkpoints, inner epoch-selection fits, and policy-only transformations.
