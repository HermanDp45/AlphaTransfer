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
