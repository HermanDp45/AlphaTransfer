# V4: реальные криптофиатные рынки и независимый эффект на FX

Сильного независимого улучшения существующей модели от криптоданных не обнаружено. Выполнены 120 годовых обучений: 24 первоначальных конфигурации и 6 компактных последующих проверок, каждая на 2023–2026 годах. Улучшение старой короткой модели от EXMO RUB premium небольшое и статистически неустойчивое; официальные валютные признаки без криптоданных дают больший эффект. Преимущество длинного обучения нельзя приписывать добавленному криптоисточнику.

## Что действительно скачано

Это биржевые spot OHLCV, агрегированные из рыночной активности. Здесь нет исторических P2P-объявлений, цен конкретного банка или гарантированно исполнимых переводов. Ни USDT/KZT, ни USDT/RUB не построены умножением официального FX на USDT.

| Источник, пара | Первая–последняя дата | Дневных баров | Отсутствующих календарных дней внутри интервала |
|---|---|---:|---:|
| Binance USDTRUB | 2019-12-25–2024-01-30 | 1498 | 0 |
| Binance BTCRUB | 2019-12-02–2024-01-30 | 1521 | 0 |
| Binance BTCUSDT | 2017-08-17–2026-09-01 | 3303 | 0 |
| Binance USDCUSDT | 2018-12-15–2026-09-01 | 2653 | 165 |
| Binance USDTKZT | 2026-05-04–2026-09-01 | 121 | 0 |
| EXMO.me USDT_RUB | 2017-09-06–2026-07-24 | 3241 | 3 |
| EXMO.me USDT_KZT | 2020-01-14–2026-07-17 | 2254 | 123 |
| EXMO.me BTC_RUB | 2017-01-01–2026-07-14 | 3482 | 0 |
| EXMO.me BTC_USDT | 2017-09-06–2026-08-24 | 3242 | 33 |
| EXMO.me USDT_USD | 2017-10-26–2026-07-13 | 3180 | 3 |

Последняя дата означает фактическую границу возвращённой истории, а не установленную нами причину прекращения торгов. Современный Binance `exchangeInfo` помечает RUB-пары `BREAK`; USDTKZT — `TRADING`. Binance KZT не имеет истории для обучения и оценки 2023–2025: скачан для аудита, в обучаемые группы не включён. У EXMO current order-book запрос для двух фиатных пар вернул пустой объект. Это ограничивает применимость исторического сигнала сегодня.

Официальный [архив Binance](https://github.com/binance/binance-public-data/blob/master/README.md) содержит monthly/daily файлы и контрольные суммы. Проверены пять доступных monthly ZIP против API: 151 дневной бар, полное совпадение close и volume, 5/5 SHA256 совпали с `.CHECKSUM`. Учтён переход архивных timestamp с milliseconds на microseconds с января 2025. Запрошенные отсутствующие архивы USDTRUB Jan2025 и USDTKZT Jan2025 вернули 404.

[Binance объявил прекращение RUB на P2P](https://www.binance.com/en/support/announcement/detail/3016096ace174be381fa22b6636e2c5f), но этот документ сам по себе не доказывает дату прекращения spot. Границу нашей spot-выборки подтверждают скачанные API и архивы; P2P и spot в выводах не смешиваются.

## Проверка EXMO и ограничения качества

EXMO предоставляет документированный публичный [`candles_history`](https://www.postman.com/exmo-finance/exmo-finance-s-public-workspace/request/2a45r5n/candles-history) с периодами `from`/`to` в Unix seconds и разрешением `D`/`60`. Во всех непустых годовых запросах полученные timestamp находятся внутри запрошенного интервала и совпадают с UTC midnight. Ответы API с rate-limit error, несмотря на HTTP 200, сохранены отдельно; после последовательных запросов с backoff данные восстановлены. Ошибки не приняты за отсутствие исторического рынка.

На четырёх датах 2020/2022/2023/2025 для каждой из RUB/KZT пар скачаны часовые бары. Все 8 дневных OHLC точно воспроизводятся агрегацией часов; максимальная погрешность суммирования volume — 5.83e-11. Для KZT 2020-03-16 есть только 6 часов с торговлей, для остальных проверенных дней — 24. Это подтверждает устройство агрегата и реальную разреженность; проверка не доказывает отсутствие wash trading и репрезентативность для розничных переводов.

В 2022 году [EXMO.com и EXMO.me стали отдельными компаниями](https://exmo.com/blog/en/product/exmo-coms-product-release-recap-main-updates-of-the-year). Однако текущие API `.com` и `.me` отдают одинаковую RUB-историю: проверенные 366 дней 2020 и 365 дней 2023 имеют нулевую разницу close/volume. В эксперименте это один исторический backend. Его ответы не объявлены двумя независимыми биржами, и их экономическая принадлежность после разделения требует уточнения у поставщика.

USDT_KZT — тонкий рынок: 13 плоских дневных баров, 40 баров с volume <100 единиц базового актива, 9 баров с high/low >1.5. В 2025 встречается close 1199.82 KZT, существенно выше обычного уровня валютного рынка. Это биржевая цена из ответа API, а не исправленный официальный FX; такие наблюдения могут отражать ошибку/манипуляцию/сегментацию, различить которые по OHLCV нельзя. Все 2023–2025 годы содержат ежедневные raw KZT бары; в 2026 осталось 78 баров с большими разрывами. Ни одна отсутствующая котировка не интерполирована; бары с некорректным OHLC или нулевым volume исключаются.

Сведения по годам: `data/exmo_yearly_quality.csv`; согласованность часов: `data/exmo_hourly_daily_audit.csv`; границы запросов: `data/exmo_request_bounds_audit.csv`; API aliases: `data/exmo_com_me_parity.csv`.

## Point-in-time протокол

Цель и решения сохранены из V3: h=5 строк публикационного proxy ЦБ; HGB, годовые train/validation/test, purge, калибровка и candidate cooldown неизменны. Development: 2023–2025, 3635 строк / 727 дат / 15 year×corridor cells. 2026: 780 строк / 156 дат, уже многократно просмотренная диагностика. Train — 2 или 10 лет; для long отключён automatic early stopping, как в V3. Обе baseline воспроизведены точно: по 4415 вероятностей и candidate flags, max absolute probability difference=0.

UTC-бар с датой открытия D закрывается в 03:00 MSK D+1. Он становится доступным модели только D+2 к 00:00 MSK, то есть с дополнительными как минимум 21 часом. Stress-вариант — D+5. Это консервативная модель доступности торговых данных через API, не доказанная дата публикации monthly ZIP. Используются восстановленные в 2026 снимки, а не архивированные historical vintages. Сам target по-прежнему CBR effective-date proxy, а не доказанный момент публикации или банковская quote-time цель.

As-of join только назад, tolerance не более 2 дополнительных календарных дней после available-date. Перед return/rolling расчётом raw series переведены на ежедневную сетку с NaN на пропусках; разрыв в торгах не считается одним торговым днём. Не используются backfill, склейка Binance с EXMO и перенос последней Binance RUB цены после прекращения рынка. Для cross-venue/triangular признаков обязательно совпадение исходной даты баров.

Premium — log(реальная crypto/fiat цена / известный официальный USD/fiat). В знаменателе уже сдвинутые PIT-поля замороженного panel: CBR USD/RUB и NBK USD/KZT. Поэтому полное покрытие KZT premium отсутствует в 2020 и составляет 66.5% в 2021: raw крипторынок существует, но старый NBK reference ещё отсутствует. Это ограничение контрольного источника, не отсутствие USDT/KZT. Полный EXMO joint наблюдается примерно на 99% development строк и только 20.5% дат 2026. Для всех конфигураций рассчитаны отдельные `source_available_matched_metrics.csv` и CI, где кандидат и baseline оцениваются на одинаковых строках с наблюдаемыми признаками; baseline указан отдельно для каждого subset.

## Что дали признаки

Каждая группа проверена отдельно: BTC returns/volatility/relative volume; USDC-USDT depeg/range; EXMO RUB и KZT premium/returns; range, relative volume, Amihud-like impact proxy; EXMO кросс KZT/RUB; Binance RUB premium/liquidity/triangular residual; межбиржевое RUB отклонение; совместные группы и официальные FX-контроли. Range/impact proxy не названы исполнимым bid-ask spread. EXMO кросс использует две фактические рыночные пары, но не является отдельной наблюдаемой P2P-ценой.

| Модель, development 2023–2025 | Brier ↓ | Δ к старой | Δ к V3 long | Candidate lift | Сигналов | Reference delta, bps |
|---|---:|---:|---:|---:|---:|---:|
| Старая CNY basis, train 24m | .190617 | 0 | +.005766 | 1.4352 | 663 | 47.69 |
| V3 CNY basis, train 120m | .184850 | −.005766 | 0 | 1.3710 | 723 | 40.08 |
| Short + official FX controls | .187808 | −.002808 | +.002958 | 1.3350 | 701 | 40.23 |
| Short + EXMO RUB premium, 3 признака | .188617 | −.001999 | +.003767 | 1.3223 | 726 | 37.86 |
| Short + EXMO RUB+KZT, 12 признаков | .192901 | +.002285 | +.008051 | 1.3434 | 696 | 37.73 |
| Short + official controls + EXMO | .188508 | −.002109 | +.003657 | 1.3759 | 720 | 41.88 |
| Short + BTC global | .195156 | +.004539 | +.010305 | 1.3607 | 703 | 40.68 |
| Short + USDC depeg | .194513 | +.003897 | +.009663 | 1.3769 | 719 | 38.54 |
| Short + all crypto | .196868 | +.006251 | +.012017 | 1.3586 | 703 | 43.21 |
| Long + EXMO RUB premium | .184805 | −.005811 | −.000045 | 1.3632 | 733 | 39.80 |
| Long + EXMO RUB premium, lag5 | .186627 | −.003990 | +.001776 | 1.4158 | 733 | 47.60 |
| Long + EXMO KZT/RUB cross, 1 признак | .184528 | −.006089 | −.000323 | 1.3695 | 705 | 41.15 |
| Long + official controls + EXMO | .186548 | −.004069 | +.001697 | 1.3818 | 736 | 39.84 |

Candidate lift стандартизован по year×corridor baseline frequency. Reference delta стандартизован по тем же ячейкам; это изменение официального reference rate после сигнала, не фактическая экономия перевода с комиссиями и банковским spread. Вероятностный Brier и utility/cadence не взаимозаменяемы.

Short RUB premium выигрывает у старой модели 0.001999 Brier, но paired CI [−.005251,+.001358]. Long RUB premium даёт только −.000045 относительно V3 long, CI [−.001760,+.001587], улучшение 1/3 лет и 8/15 ячеек. Самый низкий crypto Brier у последующей одномерной cross-проверки: incremental Δ=−.000323, CI [−.002357,+.001857], 1/3 лет, 6/15 ячеек. Общий выигрыш −.006089 против старой модели в основном объясняется train120m. Добавление joint crypto к official-only controls ухудшает development Brier и на short (+.000699), и на long (+.000692).

Binance RUB требует отдельного чтения: full-period Brier .188440 включает годы, когда источник уже отсутствует. Единственный полный development год с реальным действующим рынком — 2023: paired Δ к старому same-period baseline −.002351, CI [−.006769,+.002031]. Это не доказательство работоспособности Binance RUB фактора в 2024–2026. Детальный полный и source-available ledger сохранён, а исчезновение источника помечено.

Диагностика 2026 неоднородна: old Brier .206627, V3 long .208406, long RUB premium .208637, long EXMO cross .200173. Но cross candidate lift падает до 1.2056 против long 1.3369, а данные KZT доступны лишь на части года. Этот результат не превращён в promotion claim. Все CI — 10 000 bootstrap повторов по месячным блокам, стратифицированным по году; все пять коридоров одного дня остаются вместе. Нет поправки на множественное тестирование и последующий выбор признаков.

## Источники, доступ и права

Публичный доступ не приравнен к свободной коммерческой перепубликации. Весь анализ выполнен как локальное исследование, без аккаунтов, ордеров, обхода блокировок или защищённых архивов.

- Binance public-data README явно предназначен для скачивания и маркирован MIT. Это сильнее простого наличия URL, но MIT в репозитории не объявлен нами универсальной лицензией на любую коммерческую перераздачу exchange/API data. [Текущие Binance Terms, 17 July 2026, clause 27](https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/bf4879710c904b991848972ec4818ba2cf9e4ce314c09adae84fa2750d3477f7.pdf) описывают использование IP для личного и внутреннего бизнеса; deployment/redistribution clearance отдельно от research.
- [EXMO first-party API documentation](https://www.postman.com/exmo-finance/exmo-finance-s-public-workspace/collection/uhjz2oi/exmo-me-api) прямо описывает применение исторических свечей для программ и тестирования стратегий. Research не объявлен запрещённым. Открытая лицензия на коммерческую перераздачу всей истории не установлена; [соглашение EXMO.me](https://exmo.me/blog/user-agreement) сохранено.
- [Bybit P2P API](https://bybit-exchange.github.io/docs/p2p/guide) требует advertiser status и API key. Получение собственного order history не равно публичному архиву исторических объявлений. Публичные spot RUB/KZT запросы возвращают `Not supported symbols`; никаких фиктивных P2P рядов вместо них не создано.
- HTX metadata содержит старые RUB symbols, однако current historical klines возвращают `invalid symbol` и по USDT/RUB, и по BTC/RUB. KZT также недоступен. Наличие symbol в metadata не признано историческим coverage.
- Binance Kazakhstan domain API возвращает HTTP567; доступ не обходился. История USDTKZT получена через документированный общий market-data-only API.
- ATAIX public overview вернул HTTP502; доступный Swagger описывает current ticker/orderbook, не многолетний архив. Старый Currency.com endpoint завершился DNS/network failure. Для них не рассчитаны вымышленные метрики.

`source_receipts.csv` содержит URL, конечный URL, UTC retrieval time, HTTP status, raw path и SHA256 всех успешных и неуспешных попыток. `failed_endpoints.csv` дополнительно учитывает application errors при HTTP200. Первичные документы и raw ответы лежат в `raw/`; решения по источникам — `source_access_decisions.csv`.

## Воспроизводимость и передача в общий V4

Из корня AlphaTransfer:

```sh
/opt/homebrew/bin/python3.11 research_v4/crypto/fetch.py
/opt/homebrew/bin/python3.11 research_v4/crypto/source_audit.py --download
/opt/homebrew/bin/python3.11 research_v4/crypto/experiment.py --stage all
/opt/homebrew/bin/python3.11 research_v4/crypto/verify.py
```

Python 3.11, pandas3.0.3, sklearn1.9.0, pyarrow; один native CPU thread для моделей. Все downloads кешируются вместе с receipts. Скрипты при import ничего не скачивают и не запускают обучение. `experiment.feature_panel(extended=True)` возвращает frozen V3 panel с crypto daily features; группы доступны в `output/feature_groups.json`, suffix `_l2`/`_l5`. Root передана компактная группа `erub_premium_l2` для отдельной bounded комбинации с KASE; результат этой комбинации относится к root liquidity ledger.

Baseline parity: `output/baseline_verification.json`. Все конфигурации, вероятности, fold metrics, paired CI, matched subsets и input hashes: `output/`. V3 и final_solution не изменялись. Основной результат этого направления — проверенный реальный USDT/KZT архив, отделение spot от P2P и отрицательная ablation, предотвращающая ошибочное объяснение улучшений за счёт множества похожих внешних факторов.
