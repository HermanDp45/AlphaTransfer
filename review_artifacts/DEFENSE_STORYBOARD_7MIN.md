# AlphaTransfer: storyboard защиты на 7 минут

Цель защиты — показать не model zoo, а зрелое решение, которое нашло механизм,
отбросило ложные метрики и знает условия безопасного запуска.

## Слайд 1 — Решение, 30 секунд

Заголовок: **«Выгодный момент без ложных обещаний: structural FX engine +
abstention»**.

- Клиенту важна исполнимая сумма получателя, банку — incremental margin после
  конкуренции за CRM-слот.
- Текущий ответ: GO на engineering; conditional GO на frozen prospective
  shadow после P0 data contracts; NO-GO на production push.
- Обещание защиты: покажем, что реально измерено, где модель молчит и как
  доказать бизнес-эффект.

## Слайд 2 — Почему старый headline нельзя защищать, 45 секунд

- `NOW lift=10.57`: не Q&A truth.
- `CLOSING hit=100%`: trigger и truth образуют почти тавтологию.
- Всего 17 policy events; 5/7 folds пусты; большая часть недель без сигнала.
- Вывод: не «улучшили число», а заменили evaluator и сохранили forensic trail.

Визуал: красная карточка `INVALID`, рядом четыре причины без мелкого кода.

## Слайд 3 — Главное открытие: target совместим со структурной формулой, 60 секунд

```text
RUB per 1 LCY ≈ (RUB per USD) / (LCY per USD)
RUB per nominal LCY ≈ (RUB per USD) × nominal / (LCY per USD)
```

- Ручная one-day all-five arithmetic compatibility: 5/5; KZT на выбранном
  strict-prior alignment: 412/413; фактический denominator/source не установлен.
- Это совместимость с contemporaneous формулой, не доказанная reconstruction,
  prediction или alpha.
- Official-reference NOW-h требует будущего пути обоих cross-компонентов;
  Alpha residual отдельно нужен для исполнимой котировки/business utility.

Визуал: два входа → deterministic cross → официальный reference → residual до
Alpha executable quote. Обязательно подпись: «15:30 — input cutoff, не
publication time; h=5 — пять CBR proxy rows».

## Слайд 4 — Данные: широкий сбор, один сильный диагностический блок, 45 секунд

- 31 source family / 76 hashed artifacts: CBR, NBK, BNS, MOEX, Fed, Treasury,
  EIA, World Bank, ECB и др.
- Public macro/KZ stack не улучшил baseline; широкий stack оказался хуже.
- Единственная сильная гипотеза — CNY spot-minus-fixing displacement.

Не называть признак cash-and-carry basis. До MOEX/KASE legal sign-off показать
два слоя: public-safe mechanism и contract-required exploratory appendix.

## Слайд 5 — Измеримый результат без маркетинга, 60 секунд

На 2023–2025, h=5 CBR proxy rows:

- Brier `0.2187 → 0.1906`, улучшение 12.85%;
- paired delta CI95 `[−0.0381; −0.0178]`, Holm `p=0.0037`;
- 3/3 years, 15/15 fold×corridor cells;
- 727 decision-date clusters, а не 3 635 независимых «наблюдений»; временная
  зависимость внутри последовательности всё равно сохраняется.

Подпись крупно: **retrospective hypothesis under assumed MOEX T+1**.

## Слайд 6 — Почему мы всё ещё не запускаемся, 75 секунд

Два графика рядом:

1. Timing cliff: lag 1 → 2 дня даёт Brier `0.1906 → 0.2255`, candidate lift
   `1.435 → 1.065`.
2. Quality–cadence frontier: при lift ≥1.3 weekly fulfillment лишь 0.76–0.84;
   при fulfillment ≥0.90 lift около 1.10–1.15.

Готовый график: [`../final_solution/research/artifacts/cadence_quality_frontier.svg`](../final_solution/research/artifacts/cadence_quality_frontier.svg).

Вывод: все 5 коридоров проходят retrospective quality intervals, но 5/5
проваливают cadence gate. Мы не force-fill тихие недели.

## Слайд 7 — Целевая архитектура с fail-closed gates, 60 секунд

```text
timestamped inputs → deterministic cross → joint nowcast/residual
→ calibrated uncertainty → utility/abstention → CRM auction → ledger
```

- Целевая policy должна давать каждому контакту reason и replayable source
  vintage; сейчас этот gate не закрыт.
- `candidate`, `delivered`, `suppressed` — разные сущности.
- Decision оптимизирует ожидаемую net utility, а не CTR и не одну accuracy.
- Первый prospective benchmark — simple GAM/logit; shallow HGB только
  challenger, champion заранее не объявляется.

## Слайд 8 — Как банк узнает, что это принесло деньги, 60 секунд

- Сначала immutable frozen shadow.
- Затем persistent client randomization: BAU scheduler vs scheduler+FX; для
  ценности ML — третий arm с простой deterministic policy.
- Primary: net contribution margin/randomized client через 90 дней после
  последнего контакта; CTR только diagnostic.
- Stop по client information + ≥50 predeclared market-episode blocks + outcome
  maturity, не по удобной календарной дате.
- Strong GO только если lower CI выше business MES и guardrails проходят
  simultaneous non-inferiority.

Финальная фраза:

> Решение будет сильным, если не станет всегда советовать: будет учитывать
> устройство target, видеть границы данных и молчать, когда ожидаемая цена
> ошибки выше пользы.

## Резервные ответы жюри

**Почему не Transformer/CatBoost?** Эффективный N — даты/эпизоды, targets
коррелированы, а широкий stack уже ухудшил OOT Brier. Сложность добавляется
только после победы над structural baseline.

**Почему 47.7 bps не экономия клиента?** Это delta официального reference rate;
нет Alpha bid/ask, spread, fee, route, expiry и поведения клиента.

**Почему не отправлять слабый сигнал в тихую неделю?** CRM-слот и доверие имеют
стоимость; Q&A разрешает команде молчать. Force-fill разрушает truth.

**Почему 2026 не holdout?** Он неполон и уже просмотрен при развитии protocol.
Новый holdout начинается только после frozen spec.

**Зачем нефть/инфляция, если не победили?** Для regime stratification и stress
tests. В fast predictor они возвращаются только по одному через PIT ablation.

**Нужен ли Bloomberg?** Не для воспроизводимой защиты. В production это
лицензированный redundant feed/calendar, если экономический эффект оправдает
стоимость.
