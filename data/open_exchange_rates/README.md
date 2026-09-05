# Open Exchange Rates: RUB/CIS daily history

The downloader stores daily end-of-day reference rates for these pairs:

`RUB/AMD`, `RUB/AZN`, `RUB/BYN`, `RUB/KGS`, `RUB/KZT`, `RUB/MDL`, `RUB/TJS`,
`RUB/TMT`, `RUB/UZS`.

The Open Exchange Rates Free plan fixes the API base to USD. The downloader
therefore derives a RUB-base cross-rate for each quote currency:

```text
quote_per_rub = quote_per_usd / rub_per_usd
```

`quote_per_rub` is the number of units of the quote currency per 1 RUB.
`rub_per_quote` is the inverse value. These are blended reference rates, not a
bank's executable transfer rates.

Belarus redenominated its currency on 2016-07-01 at `10,000 BYR = 1 BYN`.
OXR observations through 2016-06-29 use the historical `BYR` code; the
downloader divides those source values by 10,000 and exposes a continuous
`RUB/BYN` series. OXR's 2016-06-30 end-of-day response already uses `BYN`.
`source_quote` preserves whether the source observation was `BYR` or `BYN`.

OXR does not return either `TMT` or its predecessor `TMM` before 2011-01-01.
Those dates are retained for the other eight pairs, but `RUB/TMT` is absent
rather than imputed from another provider or a fixed official rate.

Run from the `AlphaTransfer` directory. By default, the App ID is requested
interactively without echoing or storing it:

```bash
python3 scripts/download_oxr_cis.py --end 2026-09-02
```

For automation, the script can alternatively read `OXR_APP_ID` from the
environment. It checks the account quota for free, downloads the newest possible
contiguous period, and can be rerun after the monthly quota refresh to extend the
history backward. Progress is checkpointed in SQLite and exported to
`rub_cis_daily.csv`. The database also records a conservative local monthly
request count, because the provider's usage endpoint may update with a delay.

To fill a bounded range without automatically extending the history, pass both
dates explicitly:

```bash
python3 scripts/download_oxr_cis.py --start 2010-01-01 --end 2010-03-30
```
