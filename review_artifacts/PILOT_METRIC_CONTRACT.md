# AlphaTransfer: контракт пилота и продуктовых метрик

Дата: **2026-09-04**. Статус: proposal для согласования до рандомизации.

## 1. Что пилот должен доказать

Offline-модель отвечает: «насколько выбранная дата лучше случайной по
reference rate». Бизнес-вопрос другой: «создаёт ли отправленный клиенту сигнал
инкрементальную маржу и ценность после конкуренции за CRM-слот, spread/fee и
каннибализации будущих переводов». Поэтому primary estimand — ITT, а не
конверсия среди открывших push.

## 2. Дизайн

- Unit randomization: **клиент**, persistent 50/50 treatment/control.
- Не рандомизировать `client × corridor`: один клиент может иметь несколько
  получателей, а общий CRM-лимит создаёт interference.
- В обеих группах теневой eligibility engine фиксирует одни и те же market
  episodes. В treatment FX-кандидат допускается в общий CRM auction; в control
  он исключён, но **обычный next-best CRM-кандидат остаётся**. Сравнение с
  принудительной тишиной оценило бы другой продукт и занизило opportunity cost.
- CRM caps/budgets изолируются между arms либо эксперимент доказывает, что они
  не binding. Иначе treatment одного клиента меняет delivery другого и ломает
  stable-unit assumption. Canonical unit — клиент, не account/device; при
  общих получателях/домохозяйствах нужен разрешённый внутренний cluster-id.
- Assignment делается до первого eligible episode и остаётся неизменным.
  Primary ITT-population и правила eligibility замораживаются заранее; нельзя
  после результата оставить только открывших push или удобные trigger dates.
- Продолжительность задаётся числом predeclared non-overlapping market episode
  blocks, а не только client N. При текущем диагностическом темпе около одного portfolio episode в
  неделю 50 эпизодов означают ориентир **около 50 активных недель**; даже при
  более широком определении corridor episode — порядка **30 недель**. Затем
  нужны 90 дней для дозревания единственного primary outcome. Прежний ориентир
  8–10 недель недостаточен для
  market-regime inference и годится только как operational ramp/A/A check.
- Стратификация до randomization: pre-period contribution margin/volume,
  основной corridor, recency/frequency, зарплатный цикл, CRM reachability.
- Анализ: ITT, CUPED по pre-period, стандартные ошибки с кластеризацией по
  клиенту и market episode/date; один frozen final look. Для client-ITT на
  реализовавшемся market path — одна строка на клиента и
  stratified-randomization inference/ANCOVA с HC2/HC3. Для переноса на будущие
  regimes — two-way clustering; при малом числе episodes wild-cluster bootstrap
  или randomization inference.
- Если нужно доказать именно ценность ML, предпочтителен трёхрукавный дизайн:
  `A=BAU`, `B=простая deterministic/rules policy` с сопоставимой cadence,
  `C=frozen ML policy`. Сначала проверяется `C>A` (полезен ли продукт), затем
  `C>B` (добавляет ли ML ценность). При нехватке мощности — два последовательных
  эксперимента.

## 3. Иерархия метрик

### Primary

Один primary horizon: cumulative `incremental net contribution margin per
randomized client` от старта до **90 дней после последнего разрешённого
контакта**. Формально:

```text
tau_ITT_CM(H) = E[Y_CM_i(policy_with_FX, 0:H) | Z_i=1]
              − E[Y_CM_i(BAU_without_FX, 0:H) | Z_i=0]
```

В denominator остаются все ex-ante targetable randomized clients, включая
недоставленные и не открывшие push. `Y_CM_i` — фактическая total relevant
contribution margin:

```text
fees + realized FX/spread revenue
− provider/liquidity/hedging cost
− incentive/price-lock cost
− refunds/chargebacks/risk loss
− variable messaging/service cost
+ margin of other products/campaigns potentially displaced
```

Не размечать отдельные расходы как «attributable to treatment» post hoc:
причинную атрибуцию даёт randomized arm difference. Opportunity cost нельзя
одновременно получить из BAU-control и второй раз вычесть модельным shadow
price.

### Key secondary / mechanism

- net transferred volume/client и completed transfers/client;
- recipient amount and effective all-in RUB/LCY versus matched executable quote;
- conversion within 1/3/7 days after eligible episode;
- cumulative effects at 7/30/60 days как secondary trajectory; 90-day —
  единственный primary readout;
- new/reactivated transfer users separately from active users.

### Cannibalization

- pull-forward loss: `max(0, −ΔV[8,90]) = max(0, ΔV[0,7]−ΔV[0,90])`;
  60-day cut остаётся только secondary trajectory;
- displacement from another corridor/recipient;
- displacement from an organic transfer that would have happened anyway;
- substitution between Alpha transfer rails/products.

### Guardrails

- global push opt-out and complaint rate;
- app uninstall / notification-disable proxy;
- failed/expired quote and changed-rate-after-click rate;
- transfer failure, fraud/AML/manual-review rate;
- CRM cannibalization of higher-value campaigns;
- corridor/provider concentration and operational incidents.

CTR/open rate are diagnostics, never success criteria.

Для объёма публикуются `ΔV[0,7]`, `ΔV[8,30]`, `ΔV[31,90]`, `ΔV[0,90]` и:

```text
pull_forward_share = max(0, ΔV[0,7] − ΔV[0,90]) / ΔV[0,7]
```

если `ΔV[0,7] > 0`. Клиентскую курсовую ценность нельзя считать только среди
совершивших перевод: это post-treatment selection. Она суммируется на каждого
randomized клиента, с нулём при отсутствии перевода.

```text
customer_value_i = Σ transfers RUB_amount ×
  (effective_LCY_per_RUB / matched_reference_LCY_per_RUB − 1)
```

Conversion 1/3/7d привязывается к одинаковому ghost-eligibility timestamp в
обоих arms; один перевод нельзя засчитывать нескольким overlapping episodes.

## 4. Utility и минимально значимый эффект

На уровне eligibility:

```text
expected_utility =
    P(incremental transfer | contact, context) × expected contribution margin
  − contact cost
  − opportunity cost of occupied CRM slot
  − expected fatigue / opt-out cost
  − quote-expiry and market-move risk
```

Перевод bps в деньги для суммы `A`:

```text
value_RUB = A_RUB × bps / 10 000
```

Для чека 50 000 RUB: 25/50/100 bps = 125/250/500 RUB gross market-timing
value до spread, fee и behavioural response. MES пилота должен быть задан в
**net RUB/client**, а не выбран после результата.

## 5. Ориентиры мощности

Двусторонний тест, `α=0.05`, power 80%, равные arms, до design effects:

| Standardized effect d | Всего клиентов | С CUPED R²=0.40 |
|---:|---:|---:|
| 0.10 | 3 140 | 1 884 |
| 0.05 | 12 558 | 7 535 |
| 0.03 | 34 884 | 20 930 |
| 0.02 | 78 489 | 47 093 |

Ориентиры бинарной конверсии: baseline 5% +1 pp — около 16 315 клиентов;
10% +2 pp — около 7 682; 20% +2 pp — около 13 019. Это planning approximations,
не финальный расчёт: нужны реальная дисперсия, ICC по episode, CUPED R² и доля
доставляемых treatment contacts.

Формула `N × 1/0.6² = 2.78` при 60% delivery — только иллюстрация для
экзогенной постоянной reach и эффекта, заданного *per delivered contact*. Для
policy-level ITT MES дополнительное деление не применяется; при
treatment-dependent suppression оно неверно. Финальный sizing симулируется на
банковском pre-period с zero-inflated/heavy-tailed margin, strata, реальной
delivery, CRM displacement, общими episodes и out-of-sample CUPED `R²`.

## 6. Event sufficiency

- Не менее 50 заранее определённых non-overlapping market episode blocks до
  итогового вывода о механизме; несколько коридоров в один день — не отдельные
  эпизоды. Их фактическую зависимость проверяют ACF/block sensitivity, а не
  объявляют независимостью. Календарный срок
  вычисляется из frozen signal policy; клиентский sample size не заменяет
  число market-regime clusters.
- Перекрывающиеся horizons и один общий RUB shock требуют episode/date
  clustering.
- Низкая cadence — причина продлить shadow/pilot, а не lowering threshold после
  просмотра treatment outcome.
- Sequential monitoring guardrails допустим; primary efficacy — только frozen
  final look либо заранее заданный alpha-spending design.

`Market episode` фиксируется до outcomes: перекрывающиеся интервалы
`[signal_ts, signal_ts+h]` по коридорам объединяются; общий RUB shock одной даты
не размножается на пять независимых событий. Stop collection наступает, когда
одновременно достигнуты planned client information target, episode/block target
и maturity 90-day outcome.

Offline cadence считается на **всех** exposure weeks, когда engine работал,
включая zero-signal, boundary и unresolved-label недели. Online hard cap:
`max_t contacts_i[t−6d:t] ≤ 2`; дополнительно публикуются `P(0/1/2/>2)`,
mean/p95/max contacts, suppression mix и доля вытесненных BAU-контактов.

## 7. Decision states

- **STRONG GO:** lower 95% bound `tau_CM > MES_CM`, 90-day net volume
  положителен, customer-value gate и все simultaneous non-inferiority
  guardrails пройдены.
- **LIMITED / CONDITIONAL GO:** lower bound `>0`, point estimate `≥MES`, но
  lower bound `≤MES`; ограниченный rollout с постоянным holdout.
- **INCONCLUSIVE:** CI включает 0 и допускает MES; сбор продолжается только по
  заранее записанному information rule.
- **NO-GO / FUTILITY:** upper 95% bound `<MES` на final look либо нарушен
  safety/quote-integrity boundary.
- **CORRIDOR RESTRICTION:** pooled effect не разрешает rollout там, где
  corridor harm-bound неприемлем; случайный subgroup noise не валит весь тест.

До старта simulated power должна быть `≥80%` на MES. Post-hoc power не является
GO-gate; после старта проверяются запланированные N/blocks и ширина CI.
Guardrail `g` проходит только если one-sided simultaneous upper 95% CI harm
ниже заранее заданного margin `M_g`; «статистически не ухудшился» недостаточно.
CTR, retrospective lift и красивые bps не могут переопределить primary ITT.

## 8. Обязательный лог пилота

На каждый eligibility decision: `decision_ts`, source vintages/available_at,
features/hash, score/probability, explanation, corridor candidate,
suppression reason, CRM contenders, assignment, delivery, open/click,
executable quote/spread/fee/expiry, transfer outcome и последующие 90-дневные
outcomes. PII хранится только в разрешённом контуре; analytical table получает
стабильный обезличенный experiment id.
