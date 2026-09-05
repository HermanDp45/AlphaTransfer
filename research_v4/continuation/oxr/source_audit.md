# OXR: independent source and timing audit

Audited 2026-09-05. Existing local data and public documentation only; **zero authenticated API requests and zero quota consumed**. `OXR_APP_ID` was absent from the auditor's environment; no credential values or account-usage identifiers were inspected. The user confirmed that earlier local history does not exist, so no further search or download was attempted. Model fitting and the later independent implementation audit are separate tasks.

## Verified local snapshot

`data/open_exchange_rates/rub_cis_daily.csv` contains **27,000 rows, 3,000 complete calendar dates, 2018-06-17–2026-09-02**, and nine currencies: AMD, AZN, BYN, KGS, KZT, MDL, TJS, TMT, UZS. There are no duplicate date/currency cells, missing calendar dates, nulls or nonpositive rates. The existing target benchmark covers only AMD, KGS, KZT, TJS and UZS.

CSV SHA-256: `d30d226fbaa0b2d89a2e7eacf011b90af93eb60150572ec66c2fb1ff26a25db2`. A read-only SQLite query against the `rates` table found the same 27,000 rows and identical text fields; maximum numeric CSV/SQLite difference was `2.8422e-14`, consistent with float serialization. Neither the database nor its WAL was changed. The structured receipt is [source_audit.json](source_audit.json).

The downloader constructs RUB crosses from two USD-base source legs: `rub_per_quote = rub_per_usd / quote_per_usd`. Its maximum relative identity error is `1.9318e-14`; the inverse-column identity error is `1.9429e-14`. This is a reference cross-rate, with no direct market transaction, transfer quote, bid/ask or execution fields. OXR describes a blend of provider prices and does not disclose its constituent sources; therefore independence from official rates or other vendors cannot be presumed. [OXR FAQ](https://openexchangerates.org/faq)

## Publication timestamp and proposed PIT protocol

`published_at_utc` is produced directly from the API payload's Unix `timestamp` by `scripts/download_oxr_cis.py:214`, rather than being replaced with a fixed end-of-day marker. Its UTC calendar date equals the CSV `date` for every observation; all nine currencies share one timestamp per date. Times range from **23:59:00 to 23:59:59 UTC**; 1,724 dates end at second 59 and 462 at second 58. The downloader's SHA-256 is `48f44fe96d63f7f31353d63213acfff50987d0be943db4ba09efa6df00bda08c`.

The official historical endpoint describes the last published rates of the requested UTC day and defines `timestamp` as publication time. A request for the current UTC day can instead return a partial-day snapshot. None of this local file's dates is the current date. [Historical endpoint documentation](https://docs.openexchangerates.org/reference/historical-json)

The FAQ places availability of a completed historical file after the following UTC midnight. This supports a daily availability boundary, but does not guarantee service uptime. [OXR FAQ](https://openexchangerates.org/faq)

For source UTC date `D`, use timezone-aware timestamps and the root benchmark's **10:05 Europe/Moscow = 07:05 UTC** decision time:

```text
complete_day_floor = max(published_at_utc, D + 1 calendar day at 00:00 UTC)
known_at_primary   = complete_day_floor + 24 hours
known_at_stress    = complete_day_floor + 48 hours
known_at_strong    = complete_day_floor + 96 hours  # optional sensitivity
```

The primary allows this source day at the `D+2` decision, at least **31 hours 5 minutes 1 second after its latest possible publication**. The stress allows `D+3`; the optional stronger stress allows `D+5`. The extra 24 hours in the primary are an explicitly chosen operational buffer, not a vendor SLA. All offsets mean elapsed calendar days, not rows in the CBR trading calendar. Backward as-of requires `known_at <= decision_at`; no backward filling from a later source observation is admissible.

These conservative lags address end-of-day timing. They cannot establish a revision-free historical vintage: neither retrieval time nor original HTTP payload vintages is retained in the exported rate table. Describe results as a **retrospective ablation with documented publication timing and conservative lag assumptions**, not an audited live data feed. In particular, `published_at_utc` must not be reinterpreted as the time this research project downloaded the record.

For feature construction, compute trailing OXR returns and volatility on its daily source series before as-of alignment. Keep the initial 20-day return warmup missing, document any smaller minimum count for rolling volatility, enforce a documented staleness limit, and never forward-fill a future observation into earlier dates. `log(OXR/CBR)` must divide `rub_per_quote` by CBR RUB per **one** unit of the same currency, using only the CBR value available at the decision time. CBR nominal quoting units require normalization. Return horizons should explicitly distinguish OXR calendar days from CBR target observations. Longer training windows have genuinely missing OXR history before June 2018.

## January and March 2026 benchmark boundaries

These counts describe the **raw source**, not the purged CBR target evaluation panel; interval ends are exclusive:

| Source period | Calendar days | Rows, all nine currencies |
|---|---:|---:|
| 2023-01-01 to 2026-01-01 | 1,096 | 9,864 |
| 2023-01-01 to 2026-03-01 | 1,155 | 10,395 |
| 2026-01-01 to 2026-03-01 | 59 | 531 |
| 2026-03-01 to 2026-09-03 | 186 | 1,674 |

For either cutoff, filtering only the training row's date is insufficient: **every fitted/calibrated label's full future target window must end before the cutoff**. Preserve the baseline horizon purge, fitted-transform boundaries, calibration chronology and decision cadence. Recompute baseline and candidate on identical target date/currency rows, and report actual post-purge sample counts. A March checkpoint may use January/February outcomes only if their labels had fully become observable before March; this does not make those months an independent validation set after they were reviewed.

The 2023–2025 development period and previously inspected 2026 outcomes remain **exploratory, retrospective evidence** under both boundaries. Renaming or moving a cutoff cannot make known history an untouched holdout. A useful ablation on the available 2018–2026 source can justify the cost of acquiring earlier data; it cannot measure or guarantee the incremental benefit of an unavailable 2010–2018 extension.

## Access and remaining scope

No OXR API endpoint was invoked. Fetching a date range is not quota-free: the time-series documentation counts each returned day as a request. [Time-series documentation](https://docs.openexchangerates.org/reference/time-series-json)

Existing authorized local research use is the scope of this audit. Do not label the snapshot freely redistributable just because a CSV exists: OXR's old license page redirects licensing interpretation to its current terms. No production redistribution entitlement is inferred here. [OXR license notice](https://openexchangerates.org/license)

**Addressed:** snapshot integrity, CSV/SQLite agreement, cross-rate units, raw publication-field provenance, UTC/MSK cutoff interpretation, conservative lag recommendation, source coverage and historical evaluation status. The subsequent independent [implementation audit](audit/REPORT.md) passed 124 checks over the final 95 checkpoints, with no open implementation findings. **Open by design:** historical revision/vintage uncertainty. No new model trials were run in this source audit.
