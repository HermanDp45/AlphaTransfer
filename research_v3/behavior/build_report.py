#!/usr/bin/env python3
"""Generate human-readable and compact machine-readable results from replay CSVs."""
from pathlib import Path
import json
import pandas as pd
from simulate import summarize
HERE=Path(__file__).resolve().parent
R=HERE/'results'
a=pd.read_csv(R/'client_policy_results.csv')
a['period']=a.year.map(lambda y:'2023-2025 OOT' if y<2026 else '2026 diagnostic')
pool=summarize(a,['scenario','period','policy'])
pool.to_csv(R/'pooled_policy_summary.csv',index=False)
seg=summarize(a,['scenario','period','segment','policy'])
seg.to_csv(R/'pooled_segment_summary.csv',index=False)


def table(df,cols,labels):
    out=['| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in df.iterrows():
        vals=[]
        for c in cols:
            v=r[c]
            vals.append(f'{v:.2f}' if isinstance(v,(float,int)) else str(v))
        out.append('| '+' | '.join(vals)+' |')
    return '\n'.join(out)

b=pool[(pool.scenario=='base')&(pool.period=='2023-2025 OOT')].copy()
b['relevance_pct']=100*b.relevance_rate
b=b.set_index('policy')
m=b.loc['market_only'];u=b.loc['user_aware'];rm=b.loc['matched_random_market'];cal=b.loc['matched_calendar']
contact_drop=100*(1-u.contacts/m.contacts)
value_keep=100*u.gross_timing_value_rub/m.gross_timing_value_rub
breakeven=(m.gross_value_rub_per_client-u.gross_value_rub_per_client)/(m.contacts_per_client-u.contacts_per_client)
headline=dict(status='SCENARIO_RESEARCH_ONLY',period='2023-2025 previously inspected OOT',population='equal thirds frequency segments',
    contact_reduction_pct=float(contact_drop),gross_timing_value_retained_pct=float(value_keep),
    market_only_contacts_per_client_period=float(m.contacts_per_client),user_aware_contacts_per_client_period=float(u.contacts_per_client),
    market_only_relevance_pct=float(m.relevance_pct),user_aware_relevance_pct=float(u.relevance_pct),
    market_only_gross_rub_per_client_period=float(m.gross_value_rub_per_client),user_aware_gross_rub_per_client_period=float(u.gross_value_rub_per_client),
    matched_random_market_gross_rub_per_client_period=float(rm.gross_value_rub_per_client),
    matched_calendar_gross_rub_per_client_period=float(cal.gross_value_rub_per_client),
    contact_cost_joint_value_break_even_rub=float(breakeven),
    fx_model_brier_improvement=0,causal_incremental_revenue=None,empirical_customer_behavior_validation=False,
    incremental_volume_rub_by_construction=0)
(R/'headline_metrics.json').write_text(json.dumps(headline,ensure_ascii=False,indent=2)+'\n')

rows=[]
for (sc,period),g in pool.groupby(['scenario','period']):
    g=g.set_index('policy');m0=g.loc['market_only'];u0=g.loc['user_aware'];r0=g.loc['matched_random_market']
    rows.append(dict(scenario=sc,period=period,market_net=m0.net_scenario_value_rub_per_client,
        aware_net=u0.net_scenario_value_rub_per_client,delta_net=u0.net_scenario_value_rub_per_client-m0.net_scenario_value_rub_per_client,
        aware_gross=u0.gross_value_rub_per_client,random_gross=r0.gross_value_rub_per_client,
        contact_reduction_pct=100*(1-u0.contacts/m0.contacts),relevance_pct=100*u0.relevance_rate))
stress=pd.DataFrame(rows);stress.to_csv(R/'scenario_comparison.csv',index=False)

segment_rows=[]
for (period,segment),g in seg[seg.scenario.eq('base')].groupby(['period','segment']):
    g=g.set_index('policy');x=g.loc['market_only'];y=g.loc['user_aware'];r=g.loc['matched_random_market']
    segment_rows.append(dict(period=period,segment=segment,market_contacts=x.contacts_per_client,aware_contacts=y.contacts_per_client,
        market_gross=x.gross_value_rub_per_client,aware_gross=y.gross_value_rub_per_client,random_gross=r.gross_value_rub_per_client,
        relevance_market=100*x.relevance_rate,relevance_aware=100*y.relevance_rate,
        gross_retention=100*y.gross_timing_value_rub/x.gross_timing_value_rub if x.gross_timing_value_rub else None))
segments=pd.DataFrame(segment_rows);segments.to_csv(R/'segment_comparison.csv',index=False)
mc=summarize(a,['scenario','period','seed','policy'])
mc_base=mc[(mc.scenario=='base')&(mc.period=='2023-2025 OOT')]
mc_rows=[]
for p,g in mc_base.groupby('policy'):
    mc_rows.append(dict(policy=p,min_gross=g.gross_value_rub_per_client.min(),max_gross=g.gross_value_rub_per_client.max(),
                        min_contacts=g.contacts_per_client.min(),max_contacts=g.contacts_per_client.max()))
mc_table=table(pd.DataFrame(mc_rows),['policy','min_gross','max_gross','min_contacts','max_contacts'],['Политика','MC min RUB','MC max RUB','MC min contacts','MC max contacts'])
base_table=table(b.reset_index(),['policy','contacts_per_client','relevance_pct','gross_value_rub_per_client','net_scenario_value_rub_per_client','timing_bps_all_planned_volume'],
    ['Политика','Контактов / клиент-период','Релевантны, %','Gross proxy RUB / клиент','Net scenario proxy RUB / клиент','bps на весь planned volume'])
seg_table=table(segments[segments.period.eq('2023-2025 OOT')],['segment','market_contacts','aware_contacts','market_gross','aware_gross','random_gross','relevance_aware'],
    ['Сегмент','Market contacts','Aware contacts','Market gross RUB','Aware gross RUB','Matched random gross RUB','Aware relevance, %'])
s_table=table(stress[stress.period.eq('2023-2025 OOT')],['scenario','market_net','aware_net','delta_net','contact_reduction_pct','relevance_pct'],
    ['Сценарий','Market net proxy RUB','Aware net proxy RUB','Δ aware − market RUB','Контактов меньше, %','Aware relevance, %'])
years=pd.read_csv(R/'policy_summary.csv');years=years[years.scenario.eq('base')].copy()
year_table=table(years[years.policy.isin(['market_only','user_aware','matched_random_market','weekly_fixed'])],['year','policy','contacts_per_client','gross_value_rub_per_client','timing_bps_all_planned_volume'],
    ['Год','Политика','Контактов / клиент','Gross proxy RUB / клиент','bps на planned volume'])
styl=pd.read_csv(R/'synthetic_stylized_facts.csv').groupby('segment').agg(
    transfers_year=('transfers_per_year','mean'),ready=('readiness_day_fraction','mean'),
    urgent=('urgent_fraction','mean'),median_window=('median_advance_window','median'),intent_tpr=('intent_tpr','mean'),intent_fpr=('intent_fpr','mean')).reset_index()
styl_table=table(styl,['segment','transfers_year','ready','urgent','median_window','intent_tpr','intent_fpr'],
    ['Сегмент','Переводов / год','Доля ready days','Срочная доля','Медианное окно, дни','Intent TPR','Intent FPR'])

report=f'''# Персонализация AlphaTransfer: меньше контактов при сохранении большей части timing value

Статус: **SCENARIO_RESEARCH_ONLY**, 2026-09-04. Синтетические поведенческие данные + неизменные исторические OOT market predictions. Причинный бизнес-эффект и реальная экономия клиентов не измерены.

## Что привнесло исследование

При сбалансированной смеси трёх частотных сегментов в 2023–2025 пользовательский gate сократил число контактов на **{contact_drop:.1f}%**, сохранив **{value_keep:.1f}%** gross timing value исходной market-only политики. Релевантность контакта в пределах синтетического окна готовности выросла с **{m.relevance_pct:.1f}% до {u.relevance_pct:.1f}%**. Однако суммарная gross value **снизилась с {m.gross_value_rub_per_client:.1f} до {u.gross_value_rub_per_client:.1f} RUB на клиент-период**. Это полезный trade-off, а не победа по всем метрикам.

При **одинаковом количестве контактов у каждого клиента** user-aware дал {u.gross_value_rub_per_client:.1f} RUB против {rm.gross_value_rub_per_client:.1f} RUB у случайного прореживания рыночных кандидатов и {cal.gross_value_rub_per_client:.1f} RUB у календарного schedule. Этот разрыв показывает потенциальную ценность качественных банковских данных о готовности, **условно на предположениях генератора**. Дешёвая календарная частота сама по себе заменяет intent недостаточно хорошо.

**Изменение уже существующих ML-метрик: 0.** FX probability, Brier и исходная h=5 сигнализация сохранены. Синтетика не добавляет независимой информации о будущих курсах. Исходная headline proxy-выгода 47.69 bps считалась на market candidates; приведённые ниже bps делятся на весь planned client volume и относятся к исходной дате личного перевода. Это разные estimands, их нельзя сравнивать вычитанием.

## Дизайн и границы вывода

Запущены 3 seeds × 450 синтетических клиентов × 4 исторических периода × 12 сценариев × 6 политик = **388 800 client-policy-period observations**. По 150 клиентов на frequent/monthly/occasional и равное распределение пяти коридоров внутри сегмента. Это **не 388 800 независимых рыночных наблюдений**: у всех один общий FX path. Все годы уже исследовались при построении исходной модели; 2026 — только diagnostic. Увеличение synthetic N уменьшает Monte Carlo noise, но не uncertainty внешней применимости.

Для пользователя фиксируются латентные transfer needs, исходные даты, суммы и моменты доступности денег. История до начала периода определяет оценку интервала; свежий intent и доступные деньги наблюдаются с ошибками. Решение не видит будущий due date, будущий FX, target или forward bps. После контакта с заданной вероятностью разрешено выполнить существующий перевод раньше, внутри доступного окна. Недостаток денег, срочность и дедлайн нельзя обойти. Все суммы и число переводов во всех arms сохраняются; эффект на volume и revenue **не идентифицирован**, хотя simulated volume delta по конструкции нулевой.

В базовом сценарии response=0.35, urgent share=0.20, intent TPR/FPR=0.70/0.03, стоимость контакта=1 RUB. Эти значения **не эмпирические оценки**. Полная карта и источники — [EVIDENCE.md](EVIDENCE.md). Модель fatigue тоже допущение, поэтому выполнен вариант без неё. Seed и random uniforms общие между сравниваемыми политиками. Каждый клиент имеет один коридор; portfolio selection пяти направлений не используется как клиентское поведение.

Матчинг exact contact budget выполняется по реализовавшемуся user-aware quota; matched benchmarks пригодны для описательного сравнения расхода, но не являются online стратегиями или причинным экспериментом. В будущем пилоте quota и allocation rules надо заморозить заранее. `weekly_fixed` имеет свой явно показанный бюджет.

## Основное сравнение: 2023–2025, equal-weight сегменты

Знаменатель «клиент-период» — один клиент в одном OOT годовом окне; границы окон совпадают с исходными predictions. Результаты 2026 не годовые и не приведены к году. `Net scenario` = gross customer timing proxy − допущенный расход контактов − дополнительные execution costs. Это условная совместная ценность; **она не равна банковской contribution margin**.

{base_table}

Timing value на каждый исходный перевод с RUB amount A:

`gross_RUB = A × (official_RUB_per_LCY(due) / official_RUB_per_LCY(execution) − 1)`.

Для несдвинутого перевода ноль. На весь клиентский planned volume — `10000 × sum(gross_RUB)/sum(A)`. В отличие от оценки только среди responders, denominator включает все запланированные события и клиентов. Официальный курс — proxy, исполнимые банковские котировки отсутствуют. Дополнительный 25/50 bps drag применяется к различию условий сдвинутой сделки против исходной даты; одинаковые baseline fees не вычитаются повторно.

При базовых amount/response предположениях user-aware выигрывает у full-budget market-only по условной joint value, если один дополнительный контакт стоит больше **{breakeven:.2f} RUB**. Это break-even сценария, **не установленная цена CRM-слота** и не расчёт банковского ROI.

## Частотные сегменты

{seg_table}

Для frequent окно потребности появляется часто, но recent-transfer suppression может выбрасывать часть полезных рыночных дат. Для monthly и occasional контакт каждую неделю обычно расходует слоты вне потребности. Не следует требовать 1–2 FX push/неделю от месячного пользователя: кейсодатель уточнил, что cadence 1–2 — самопроверка рыночного фильтра на коридор. Клиентская cadence ограничивается его потребностью и общим CRM cap.

Состав аудитории дополнительно менялся в `population_mix_sensitivity.csv`: balanced, условный active bank 15/50/35, frequent-heavy 50/35/15, исторический stress 0/34/66. Ни одна смесь не выдаётся за оценку состава банка.

## Проверка устойчивости предположений

{s_table}

Нулевой response даёт ровно нулевую gross timing value у всех политик. Это обязательный negative control: хороший FX signal сам по себе не создаёт перевод. Малое окно гибкости, преобладание срочных потребностей и слабые наблюдаемые признаки снижают ценность персонализации. Execution drag может съесть значительную часть выгоды. Поэтому высокий signal lift недостаточен для GO.

## Зависимость от реального исторического market path

{year_table}

На разных годах меняются как уровень, так и сравнительная ценность weekly baseline. Персонализация не должна автоматически объявляться превосходящей market-only по общей timing value. При этом экономия контактов сохраняется в базовом сценарии. 2026 остаётся уже просмотренным diagnostic и не подтверждает prospective качество.

Monte Carlo диапазоны по трём seeds для 2023–2025:

{mc_table}

Это диапазоны **синтетической случайности при фиксированном рынке**, не confidence intervals для population effect. Bootstrap клиентов здесь не доказывал бы устойчивость к будущим FX regimes. Без real bank outcomes не публикуем p-values или confidence bands для бизнес-uplift.

## Достигнутые свойства генератора

{styl_table}

Это тест того, что код генерирует заданные частоты/готовность/ошибки измерения. Не внешний empirical holdout. Все проверки отсутствия задержек, исполнения до денег, сдвига срочных переводов, создания count/volume и превышения 2/7d прошли. Бюджет двух matched политик совпадает с user-aware **для каждого клиента**, а не только в среднем. `validate.py` отдельно меняет будущие FX/targets и убеждается, что набор контактов не меняется; flat FX даёт нулевую gross value.

## Почему synthetic user features не усиливают FX prediction

Если U синтезирован независимо от будущего Y при фиксированном рыночном X и календаре, то `P(Y | X, U) = P(Y | X)`. При этом U может улучшить выбор **кому и когда показать уже существующий market signal**, потому что меняется utility. Размножение одного date/corridor на тысячи synthetic users не создаёт новые FX labels. Случайный train/test split по таким строкам дал бы утечку общей даты; нужны date blocks и customer holdout. Здесь FX модель вообще не переобучалась.

## Что интегрировать и как получить реальные доказательства

1. Оставить рыночную модель отдельной: probability/quality/cadence на date×corridor. После неё добавить readiness/affordability gate и общий CRM auction. Чистый прототип [preview_gate.py](preview_gate.py) возвращает suppression reasons и не меняет probability. Интегрируемый stdlib API [build_behavior_preview](../../final_solution/alphatransfer_final/behavior.py) добавлен в `final_solution`: explicit simulation, отказ при будущих наблюдениях, отсутствие данных, consent, urgency, CRM cap и cold-start. Семь [unittest](../../final_solution/tests/test_behavior.py) прошли. `production_eligible=False` во всех состояниях; production TTL/timezone/live quote и остальные гейты обязательны.
2. Собрать 12–24 месяца обезличенной истории: timestamp каждого перевода и получателя, amount/all-in quote, recency/frequency, зачисление дохода и доступный баланс, последние quote-view/form-start/draft, preferred deadline/urgent flag, клиентский consent, delivery/open, competitor CRM contacts. Нужны `observed_at/available_at`, история изменений, feature lineage. Этничность/гражданство не требуются для частотной сегментации.
3. Первый пользовательский ML — **discrete-time hazard органического перевода** на 1/3/7 дней по RFM, регулярности, денежному поступлению, получателю и app intent. Использовать train-time history, temporal/customer holdout, calibration/Brier/PR-AUC, не путать propensity to transfer с uplift от push. Неполные окна требуют censoring-aware разметки. Гибкость лучше спросить прямо («Планируете перевод до…») и проверять observed preference.
4. Следующий шаг — RCT: persistent client assignment; A=BAU, B=market-only, C=market+readiness, равные ex-ante правила бюджета. При недостаточной мощности сначала B vs C с одинаковым CRM auction. Заморозить eligibility до назначения; в control логировать ghost events. Primary — **90-day incremental net contribution margin per randomized client**, с BAU displacement, guardrails и cannibalization; текущий [PILOT_METRIC_CONTRACT](../../review_artifacts/PILOT_METRIC_CONTRACT.md) уже описывает основу. Customer-value считать на всех assigned users. Нужен и client information target, и достаточно общих market episodes.
5. Только после рандомизации учить uplift/CATE и expected incremental margin для выбора CRM action. Cross-fitting и doubly robust policy evaluation полезны при известных propensity и overlap; синтетические response labels не заменяют experiment. Реальный запуск должен уметь отказаться от FX push, если слот другой кампании ценнее.
6. Согласовать клиентский payoff с **его сроком**, а не всегда h=5: полезность перевода сегодня против его органической даты/дедлайна. Market horizons 1/3/5/10/20 сохраняются для кейса; user utility считается отдельным слоем. UI говорит о текущей исполнимой сумме и историческом сравнении, затем перепроверяет live quote при открытии. Сдвиг уже запланированного перевода нельзя выдавать за инкрементальный оборот.

## Решение

**Включить как research preview и обоснование ценности собственных банковских данных. Не повышать production статус.** Доказан работоспособный симуляционный контур и точный trade-off: много меньше контактов, большая часть timing value сохранена, better relevance и higher value при равном бюджете в заданном генераторе. Не доказаны внешний реализм синтетики, causal uplift, net bank revenue или дополнительное качество FX модели. Именно эти различия следует сохранить на защите и в продуктовых артефактах.
'''
(HERE/'REPORT.md').write_text(report)
# Compact SVG is a standalone research figure and requires no visualization runtime.
points=b.reset_index();W,H=920,440;margin=70
maxx=55;maxy=max(600,float(points.gross_value_rub_per_client.max())*1.15);miny=min(-100,float(points.gross_value_rub_per_client.min())*1.1)
colors={'user_aware':'#d51e37','market_only':'#1f3d69','frequency_gate':'#668eae','matched_random_market':'#aa78a6','matched_calendar':'#9b9b9b','weekly_fixed':'#708967'}
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',
'<style>text{font-family:Arial,sans-serif;fill:#223047;font-size:13px}</style>',
'<text x="70" y="27" font-size="19" font-weight="bold">Synthetic readiness: contacts vs official-rate timing proxy</text>',
'<text x="70" y="47" fill="#667">2023–2025, balanced segments; assumed response 0.35; not causal uplift</text>']
def xy(x,y):return margin+x/maxx*(W-2*margin),H-margin-(y-miny)/(maxy-miny)*(H-2*margin)
for yy in range(-100,int(maxy)+1,100):
    x,y=xy(0,yy);svg.append(f'<line x1="{margin}" y1="{y}" x2="{W-margin}" y2="{y}" stroke="#e6e8ed"/><text x="{margin-45}" y="{y+4}">{yy}</text>')
for xx in range(0,56,10):
    x,y=xy(xx,miny);svg.append(f'<text x="{x-8}" y="{H-margin+22}">{xx}</text>')
for _,row in points.iterrows():
    x,y=xy(row.contacts_per_client,row.gross_value_rub_per_client);p=row.policy
    dy={'matched_random_market':0,'matched_calendar':17,'user_aware':-10,'frequency_gate':14,'market_only':-11,'weekly_fixed':16}[p]
    svg.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{colors[p]}"/><text x="{x+10}" y="{y+dy}" font-weight="bold">{p}</text>')
svg += ['<text x="350" y="428">Contacts per client-period</text>','<text transform="translate(16 340) rotate(-90)">Gross timing proxy RUB per client-period</text>','</svg>']
(R/'contacts_value_frontier.svg').write_text('\n'.join(svg))
print(json.dumps(headline,ensure_ascii=False,indent=2))
