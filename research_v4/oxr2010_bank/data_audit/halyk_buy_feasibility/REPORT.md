# Halyk BANK BUY RUB: bounded feasibility check

The official public endpoint contains the **correct bank side for a customer selling RUB for KZT**, but the observed response supplies only two recent dates. No historical BUY dataset was fabricated or substituted for the existing SELL archive.

On 2026-09-05, two small successful public requests were made to [Halyk's currency-history endpoint](https://back.halykbank.kz/common/currency-history): its ordinary response and a bounded probe with `?date=2020-01-05`. Both returned identical 3,320-byte bodies, SHA-256 `ff10c117e00df925eda42e652045eba2c14cdaf20edb515bf8e98c467a7a8d26`. The requested historical date was **not honoured**: both bodies contain 2026-09-04 and 2026-09-03 only. This rules out that particular query pattern; it does not prove that no other historical interface exists.

| Effective date | Channel | BANK BUY, KZT per RUB | BANK SELL, KZT per RUB |
|---|---|---:|---:|
| 2026-09-04 | privatePersons | 4.47 | 5.28 |
| 2026-09-03 | privatePersons | 4.47 | 5.28 |
| 2026-09-04 | cards | 5.0662 | 5.4662 |
| 2026-09-03 | cards | 5.0739 | 5.4739 |

These are channel-specific indicative quote fields. They do not by themselves establish a remittance's executable net receipt, applicable amount tier, branch, fees or settlement time. `buy` is the appropriate bank-side direction; a larger KZT/RUB BUY quote means more KZT per RUB before any additional costs.

The locally frozen official [exchange-rates page](https://halykbank.kz/en/exchange-rates) and its JavaScript distinguish the current buy/sell widget from the historical chart. The observed historical request is `/api/exchangerates/{currency}/{date-range}`; its visible label specifies BANK SELL. The saved chart payload has only `date`, `date_at`, `value`, and `currency`, with no BUY field. The frontend's long-range chart fetch includes no side selector. No BUY-history interface was identified in that inspected frontend.

The practical next input is a bank-supplied historical BUY archive with channel/amount/publication metadata, or prospective snapshots of the public buy/sell endpoint. Two available dates cannot support a 2020–2026 BUY-side backtest. No monitoring or recurring downloads were started here.

Evidence: [successful network receipt](network_receipt.json), [current raw response](current.json), [historical-date probe](requested_2020_date.json). An earlier sandbox DNS failure is retained in [receipt.json](receipt.json); only the two later requests returned data. No authenticated requests, bulk downloads, normalized-file edits or model fits were performed.
