# Важно перед передачей артефактов третьим лицам

Статус: **обязательная legal/data-owner проверка**. Это не юридическое
заключение.

В `raw/`, `normalized/` и `results*/` присутствуют MOEX-derived данные и
метрики. Официальная Market Data Policy MOEX относит non-display processing и
создание/использование derived data к договорным сценариям; публикация данных
на сайте сама по себе не даёт право на любое programmatic или third-party use:
[MOEX Market Data Policy](https://www.moex.com/files/4a1jy8j83qc25vv9p286tzsmc1).

KASE отдельно указывает, что investment analysis, derived/non-display use и
delayed API требуют стандартного соглашения:
[KASE Non-Display/Derived Information](https://kase.kz/en/information/non-display).

До письменного подтверждения прав:

- не публиковать и не отправлять наружу raw/normalized MOEX/KASE data;
- не включать contract-required численные результаты в публичный датасет;
- в презентации отделять public-only mechanism result от
  `contract_required_market_data` exploratory track;
- при необходимости пересобрать public-only bundle из CBR/NBK/BNS/Fed/
  Treasury/EIA/World Bank/ECB с сохранённой атрибуцией и проверенными terms.

NBK и Kazakhstan BNS официально допускают свободное повторное использование
официальной статистики, включая изменение и software-use, при ссылке на
источник: [NBK license](https://nationalbank.kz/en/page/data-usage-terms),
[BNS terms](https://stat.gov.kz/ru/description/). Это не передаёт права на
исходные биржевые KASE данные, если NBK лишь перепубликовывает агрегат.

FRED намеренно не включён: действующие Terms запрещают использование FRED
Content для разработки/обучения ML и отдельно запрещают storage/cache/archive:
[FRED Terms](https://fred.stlouisfed.org/legal/).

Отдельно: chat-derived материалы из `main` не входят в этот data bundle.
Нельзя передавать raw messages, usernames/handles или иные персональные данные;
даже обезличенные агрегаты используются только после проверки consent,
provenance и разрешённого контура.
