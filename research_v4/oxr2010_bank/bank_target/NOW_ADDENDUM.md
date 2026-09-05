# Additional actual bank NOW-event control

This bounded addendum was requested after the primary future-mean experiment's results were inspected. It is not part of the initial primary hypothesis and is not selected using 2026 outcomes.

On the exact same clean Halyk quote rows, features, maturity boundaries and five cutoffs, set `bank_now=I(q_anchor <= min(q_next1,...,q_next5))`. The bank SELLS RUB for KZT; lower KZT/RUB is preferable to the RUB buyer. Equality is a success: the current archived quote is no worse than every one of the next five actual observed bank sell quotes. No epsilon/tie parameter is tuned. This matches a no-better-future-opportunity NOW event in the correctly fixed buyer direction.

Refit only the six fixed classification arms (two identical prior-validation prevalence controls and four HGB feature groups). No regression refit is needed. Calibration is fit to the same purged prior twelve months. Keep both all-date Brier and past-only baseline skill; source start2010/2018 remains a deterministic identity control. All methods are reported, no 2026 selection.

The anchor q was observed the previous effective day, so a positive event does not prove the quote was executable at decision time. The target is a counterfactual comparison of archived quotes, not an executed NOW order. The <=14-calendar-day future-window filter conditions evaluation on future archive density; it is not an online eligibility rule. The primary future-mean results remain unchanged and must be reported even if this follow-up looks better.
