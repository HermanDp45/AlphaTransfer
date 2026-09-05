#!/usr/bin/env python3
"""Build audit tables and report from completed experiment artifacts."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research_v4.segments import experiment as e
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def table(df,cols,renames=None):
 d=df[cols].copy()
 for c in d:
  if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(lambda x:f'{x:.3f}')
 if renames:d=d.rename(columns=renames)
 return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.to_numpy())

def main():
 pooled=pd.read_csv(OUT/'weighted_development.csv');years=pd.read_csv(OUT/'weighted_by_year.csv');segments=pd.read_csv(OUT/'by_segment_development.csv');raw=pd.read_csv(OUT/'client_outcomes.csv.gz');contacts=pd.read_csv(OUT/'base_contacts.csv.gz',parse_dates=['date']);receipts=json.loads((OUT/'selected_policy_receipts.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
 primary=pooled[(pooled.scenario=='base')&(pooled.mix==e.MAIN)].copy();primary.to_csv(OUT/'headline.csv',index=False)
 frequent_controls=segments[(segments.segment=='frequent')&segments.policy.isin(['universal','group_aware'])].copy();frequent_controls.to_csv(OUT/'frequent_response_cost_sensitivity.csv',index=False)
 frequent_dates=contacts[(contacts.year<2026)&contacts.segment.eq('frequent')].groupby('policy').agg(unique_market_dates=('date','nunique'),synthetic_contacts=('date','size')).reset_index();frequent_dates.to_csv(OUT/'frequent_market_support.csv',index=False)
 diagnostic=[]
 for (scenario,year,corridor,policy),f in raw.groupby(['scenario','year','corridor','policy']):diagnostic.append(dict(scenario=scenario,year=year,corridor=corridor,policy=policy,**e.weighted(f,e.MIXES[e.MAIN])))
 pd.DataFrame(diagnostic).to_csv(OUT/'corridor_year_diagnostics.csv',index=False)
 seedrows=[]
 for (scenario,seed,policy),f in raw[raw.year<2026].groupby(['scenario','seed','policy']):seedrows.append(dict(scenario=scenario,seed=seed,policy=policy,**e.weighted(f,e.MIXES[e.MAIN])))
 pd.DataFrame(seedrows).to_csv(OUT/'seed_diagnostics.csv',index=False)
 # Week-block resampling preserves all corridors and synthetic contacts sharing each week.
 # Conditional selected-policy diagnostic only; no model/policy refit and no causal inference.
 c=contacts[contacts.year<2026].copy();c['week']=c.date.dt.to_period('W-SUN').dt.start_time
 w=dict(zip(e.SEGMENTS,e.MIXES[e.MAIN]));c['weight']=c.segment.map(w);c['hits']=c.h5_hit*c.weight
 weekly=c.groupby(['week','policy'])[['weight','hits']].sum();weeks=pd.date_range(c.week.min(),c.week.max(),freq='W-MON');policies=['universal','group_aware','universal_expected_budget','group_expected_budget']
 arrays={p:weekly.xs(p,level='policy').reindex(weeks,fill_value=0).to_numpy() for p in policies};rng=np.random.default_rng(9931);bootstrap=[]
 for b in range(1000):
  starts=rng.integers(0,len(weeks),size=int(np.ceil(len(weeks)/8)));ix=np.concatenate([(np.arange(8)+j)%len(weeks) for j in starts])[:len(weeks)]
  quality={p:arrays[p][ix,1].sum()/arrays[p][ix,0].sum() for p in policies}
  bootstrap.append((quality['group_aware']-quality['universal'],quality['group_expected_budget']-quality['universal_expected_budget']))
 bs=np.array(bootstrap);intervals={'method':'paired circular 8-calendar-week blocks; all synthetic rows and corridors retained within week; conditional fixed-policy descriptive interval, no refit','replicates':1000,'unique_development_market_dates_with_contacts':int(c.date.nunique()),'group_minus_universal_h5_quality_pp':(np.quantile(bs[:,0],[.025,.5,.975])*100).tolist(),'group_minus_universal_expected_budget_h5_quality_pp':(np.quantile(bs[:,1],[.025,.5,.975])*100).tolist()}
 (OUT/'market_block_diagnostic.json').write_text(json.dumps(intervals,indent=2)+'\n')
 sensitivity=pooled[pooled.scenario=='base'].pivot(index='mix',columns='policy',values='net_scenario_value_rub');sensitivity['group_minus_universal']=sensitivity.group_aware-sensitivity.universal;sensitivity.to_csv(OUT/'mixture_sensitivity.csv')
 controls=pooled[(pooled.mix==e.MAIN)&pooled.policy.isin(['universal','group_aware','universal_expected_budget','group_expected_budget'])]
 selected=[]
 for r in receipts:
  for seg in e.SEGMENTS:selected.append(dict(test_year=r['test_year'],segment=seg,universal=r['universal'][seg],group_aware=r['group_aware'][seg],unconstrained=r['group_unconstrained_exploratory'][seg]))
 pd.DataFrame(selected).to_csv(OUT/'policy_choices.csv',index=False)
 m=primary.set_index('policy');u=m.loc['universal'];g=m.loc['group_aware'];fre=segments[(segments.scenario=='base')&(segments.segment=='frequent')].set_index('policy');fu=fre.loc['universal'];fg=fre.loc['group_aware']
 bands=pd.read_csv(HERE/'data/l2kgz_regularity_bands.csv');trans=pd.read_csv(HERE/'data/l2kgz_transitions.csv');sources=json.loads((HERE/'data/l2kgz_aggregate_manifest.json').read_text())
 text=f'''# V4: частота клиента и политика выбора рыночной возможности

Статус: **retrospective scenario research, не причинный uplift и не production policy**. Выполнено 2026-09-05. Основной результат — разделение полезного frequent tradeoff и провалившейся общей гипотезы. Все показатели ниже воспроизводятся из CSV, отрицательные результаты сохранены.

## Что реально добавлено к V3

V3 давал readiness gate после одинакового рыночного сигнала. Здесь каждой группе разрешено выбирать собственные **горизонт h1/3/5/10/20, порог вероятности и минимальный промежуток между рыночными возможностями**. Настройка использует только прошлые OOT годы. Реально скачаны микроданные World Bank L2KGZ и получены агрегаты регулярности вместо ссылок на каталог; сохранена сопоставимость рынков, пользователей, расходов и объёма между политиками.

**Частым пользователям** group policy дала {100*(fg.contacts/fu.contacts-1):.1f}% больше контактов ({fu.contacts:.3f}→{fg.contacts:.3f}) и {100*(fg.net_scenario_value_rub/fu.net_scenario_value_rub-1):.1f}% больше условной net timing value ({fu.net_scenario_value_rub:.2f}→{fg.net_scenario_value_rub:.2f} RUB/client-period), при падении common-h5 quality на {100*(fu.h5_quality-fg.h5_quality):.2f} п.п. Это прозрачная цена частоты, а не улучшение accuracy FX модели.

**В фиксированной основной смеси** результат хуже universal: {u.net_scenario_value_rub:.2f}→{g.net_scenario_value_rub:.2f} RUB/client-period ({100*(g.net_scenario_value_rub/u.net_scenario_value_rub-1):.1f}%), контакты {u.contacts:.3f}→{g.contacts:.3f}, h5 quality {u.h5_quality:.2%}→{g.h5_quality:.2%}, relevance {u.relevance:.2%}→{g.relevance:.2%}. Улучшение relevance само по себе не доказывает денежную полезность. Политика без ограничений cadence также не превзошла universal; проблема включает нестабильность выбора параметров по короткой истории отдельных групп.

## Сегментация из чатов — другой признак

Источник: `Researches/chat_research_kz/report.md`, §3; `Presentation Artifacts/Презентация_Д4.md`, слайд6; `product_artifacts/CLIENT_JOURNEY.md`.

| Гипотеза из чатов | Продуктовый смысл | Что доступно этому эксперименту |
|---|---|---|
| S1 регулярный отправитель с гибким окном | Предлагать дату в уже известном допустимом окне | Частота симулируется; гибкость и intent — отдельные сценарные переменные |
| S2 срочная потребность | Сразу текущая сумма и надёжный маршрут | Срочные потребности никогда не сдвигаются; preview подавляет push |
| S3 ищет работающий маршрут | Доступность, total cost, fallback | Нельзя лечить проблему маршрута прогнозом курса |
| S4 сравнивает банк/P2P/наличные | Сравнение исполнимого net результата | Предпочтение внешнего канала не выводится из банковской истории |
| S5 суперчастый участник чата | Роль не установлена | Количество сообщений не означает число переводов |

S1/S2 могут быть состояниями одного клиента в разные дни. Три группы frequent/monthly/occasional — ось частоты внутри потенциальной аудитории, а не доли пяти JTBD. Ни raw чаты, ни идентификаторы авторов не экспортированы. Чатовые counts и qualitative samples не использованы как population weights.

## Первичные данные и веса

[World Bank RPP baseline report](https://documents1.worldbank.org/curated/en/552541540823620142/pdf/131455-RPPbaselinesurveyFINAL.pdf): полевой опрос 2016, Kyrgyzstan819 и Tajikistan1053 респондента, figure5.1.2.3, PDFpage20/printed19. Скачан PDF, числа проверены визуально. KG: ≥monthly40%,4–6/year32%,2–3/year21%,yearly6%; TJ41/34/17/7. Округлённая сумма99% нормализуется. Это получатели переводов, преимущественно из РФ, не банк-клиенты-отправители. Несколько отправителей у одной семьи дополнительно мешают прямому переносу частоты.

Основная fixed mixture: frequent4/99,monthly36/99,occasional59/99. Наблюдаемы только первые две группы вместе40/99; распределение 10% monthly-bin в frequent — **плановое допущение**, не оценка. Sensitivity меняет frequent долю этого bin от0 до50%, держит одинаковые веса для всех политик. Частота 7/14 дней внутри frequent остаётся сценарной, не доказанной российской payroll статистикой.

[World Bank Listening to Kyrgyz Republic 2021–2025](https://microdata.worldbank.org/catalog/6523), Ref KGZ_2021-2025_L2KGZ_v02_M, [DOI](https://doi.org/10.48529/swmc-aq82). Скачан официальный CSV ZIP {sources['zip_bytes']:,} bytes: {sources['household_month_rows']:,} household-month и {sources['individual_month_rows']:,} individual-month строк. Используется `mig_living_remittance` — yes/no отправки домохозяйству за прошлый месяц. Есть {sources['observed_migrant_months']:,} наблюдаемых migrant-month и {sources['observed_migrants']:,} мигрантов; пропуски не превращены в no. Исходные rows находятся только во временном исследовательском каталоге, в repository только агрегаты. Автор данных: World Bank; скачано 2026-09-05. Публичный v02 релиз содержит поздние наблюдения: даже подвыборка до2024 не объявляется доступной в2023.

{table(trans,['period','previous_sent','consecutive_pairs','unweighted_p_next_sent','derived_household_weighted_p_next_sent'])}

Последовательные пары — один и тот же мигрант в соседних календарных месяцах; апрель2024 отсутствует и не склеивается. Используется population weight `popw`; для мигрантских распределений дополнительно показан производный household weight `popw/hhsize`, усреднённый по наблюдениям. Это descriptive adjustment, **не предоставленные longitudinal migrant weights**. Нет design-based population CI. Условие ≥6/12 наблюдений и ≥1 перевода создаёт отбор; attrition и возвращение мигрантов важны.

{table(bands[(bands.period=='full2021_2025')&(bands.min_observed_months==6)],['band','migrants','eligible_migrants','unweighted_share','derived_household_weighted_share'])}

Новая empirical sensitivity относит >75% observed-month senders к monthly-or-frequent bin, а остальные к occasional. Внутри bin10% условно frequent. Это перевод survey definition в stress scenario, **не идентификация частоты внутри месяца**. Новые данные подтверждают persistence, но показывают существенные месяцы без переводов: регулярный календарь V3 не прошёл полноценную внешнюю валидацию. Значения intent TPR/FPR, response35%, окно2/5/10дней, urgent20%, чек18/45/90тысRUB и fatigue остаются гипотезами.

[IOM Kyrgyz return-migrant survey July2024](https://kyrgyzstan.iom.int/sites/g/files/tmzbdl1321/files/documents/2024-11/kg_return-migrant-survey_r3_eng.pdf): официальный поисковый индекс содержит9% >monthly,57% monthly,18% quarterly,15% circumstances,1% nonresponse среди отправлявших. PDF вернул403; поэтому это отдельно помеченный sensitivity, не основной downloaded источник. Распределение не переносится на Alfa и не используется как известное в2023. Категория circumstances условно объединена с occasional; это ещё одно допущение.

## Протокол и защита от ложного улучшения

- Заморожен `baseline_reproduction`, один model family, по всем пяти h. Никакого дополнительного обучения FX модели на синтетике.
- Universal выбирает одну из60 комбинаций h×quantile(.25,.50,.75)×cadence(3,7,14,28 calendar days). Пороги вычисляются по прошлым score distributions, не test.
- Group-aware: frequent cadence3/7,monthly7/14,occasional14/28; каждая группа выбирает свою комбинацию. Цель: gross timing RUB минус1RUB/contact; минимум2 контакта/client-period в обучающей смеси. Unconstrained group использует все60 комбинаций для каждой группы: добавлен **после просмотра первого результата**, явный exploratory followup.
- Train2023→test2024; train2023–24→test2025; train2023–25→diagnostic2026. Все старые h20 label horizons завершены до следующего года; тест проверяет это по исходным CBR session positions. Сами V3 scores и рынки ранее были просмотрены: это nested prior-year retrospective selection, не нетронутый prospective holdout.
- Контакты разрешены только на **пересечении** доступных дат всех h. Так h1 не получает дополнительные tail dates. Common market rows by year: {manifest['common_market_rows_by_year']}. Всего {manifest['unique_market_dates']} уникальных дат, а не множество независимых synthetic examples.
- Все политики получают одинаковые cohorts, random uniforms, потребности, суммы и FX paths. Training seed7101 отличается от evaluation8101/8102/8103; evaluation180 клиентов/segment, 540 total. Синтетическая population composition не меняется между политиками.
- Reuse `research_v3.behavior.simulate`: готовность `(phase OR observed intent) AND balance`, recent-transfer gate, максимум2 контакта в rolling7 calendar days. У пользователя один primary corridor; cap общий для предложений симуляции. Внешний CRM traffic и multicorridor portfolio клиента не симулированы, их лимиты нужны в боевом gate.
- Cadence применяется к хронологическому потоку **market candidates до readiness**, поэтому она causal, но может пропустить редкое удачное окно готовности. Это конкретная проверенная политика, а не доказательство бесполезности любой personalization.
- Organic перевод происходит в срок; можно только продвинуть его внутри funded window. `A*(FX_due/FX_exec−1)` — conditional customer timing-value proxy. Net вычитает contact cost и execution drag. Count и RUB volume неизменны во всех arms; bank incremental conversion, revenue, profit и causal uplift **не оценены**.
- CBR daily — reference rate/publication proxy, не исполнимый bank quote; одинаковые real FX paths недостаточны для финансового P&L доказательства.

## Сопоставимые метрики, 2024–2025

Client-period — от1января до последней общей score date года, не полный год. Основной pooling усредняет client-periods, сценарий и mixture фиксированы; `contacts_per_week` нормирует экспозицию. Gross/net здесь RUB на клиента за период. **h5_quality** всегда оценивает одну и ту же h5 цель. Own-h quality сохранена в CSV для диагностики, но не используется для заявления роста качества: более короткая цель проще. Relevance — доля контактов, попавших в латентное окно потребности, известное только симулятору.

{table(primary,['policy','contacts','contacts_per_week','h5_quality','relevance','gross_timing_value_rub','net_scenario_value_rub','timing_bps_all_volume'])}

`v3_readiness` — контроль V3 gate на одинаковом новом common grid и frozen baseline h5 candidates. Его цифры не являются прямой заменой sealed V3 report с другим покрытием дат/модельным артефактом/смесью.

## Частота против качества по сегментам

{table(segments[(segments.scenario=='base')&segments.policy.isin(['universal','group_aware','group_unconstrained_exploratory'])],['segment','policy','contacts','h5_quality','relevance','net_scenario_value_rub'])}

Frequent действительно чаще получает предложения и выигрывает по timing proxy с потерей h5 quality. Но у редкого клиента есть мало потребностей: отбрасывание рыночных дат по длинному cooldown лишает его значительной части ценности. «Редкие переводы» не означают «редко проверять рынок»; правильнее редко отправлять пуши и внимательно проверять короткое окно реального intent.

### Frequent: чувствительность response/cost и поддержка рынками

{table(frequent_controls,['scenario','policy','contacts','h5_quality','gross_timing_value_rub','net_scenario_value_rub'])}

{table(frequent_dates,['policy','unique_market_dates','synthetic_contacts'])}

Контакты искусственных пользователей повторно используют одни и те же рыночные даты. Всего common development grid455 уникальных дат (228в2024,227в2025); они также serially dependent и коррелированы между коридорами. Число эффективно независимых рыночных наблюдений меньше и не известно. +14% — модельная разность при response35%, не измеренный treatment uplift; таблица показывает смену величины/знака при других предположениях.

## Expected budget controls и стабильность

Target — минимум prior weighted contact means universal/group. Каждая arm получает Bernoulli thinning, калиброванный на предыдущих users/years; current/test quota не читается. Random sender выбирает дни без FX score, weekly — calendar gap7; readiness одинаковый. Prior calibration receipts содержат target, achieved mean и thinning probability. Это **matched expected budget**, не точное ex-post равенство: на новых годах budget drift виден в таблице и нельзя приписать всю разность только selection quality. В первой основной паре universal_expected даже меньше контактов, чем group_expected, но больше value; это не даёт формального causal superiority.

{table(years[(years.scenario=='base')&(years.mix==e.MAIN)&years.policy.isin(['universal','group_aware','group_unconstrained_exploratory'])],['year','policy','contacts','h5_quality','net_scenario_value_rub'])}

2025 и2024 расходятся по знаку общего эффекта. 2026 неполный и diagnostic. В `corridor_year_diagnostics.csv` сохранены все пять corridor×year cells; в `seed_diagnostics.csv` Monte Carlo разброс, который нельзя трактовать как неопределённость новых рыночных режимов. Paired8-week block diagnostic h5-quality difference group−universal95% interval: {intervals['group_minus_universal_h5_quality_pp']}; единица resampling — календарная неделя со всеми коридорами/клиентами внутри. Это условный интервал фиксированных selected policies, без refit, **не confirmatory CI** и не CI причинного эффекта.

## Нулевые и слабые поведенческие сигналы

{table(controls,['scenario','policy','contacts','h5_quality','relevance','gross_timing_value_rub','net_scenario_value_rub'])}

Zero-response обязан давать gross0 и net=−contact cost; это проверено. Weak intent использует TPR.30/FPR.15, balance sensitivity.75, phase noise7days вместо базовых. Policy choices остаются замороженными. Узкое окно, contact cost10 и execution drag25bps — stress tests. Если стоимость контакта трактуется как CRM бизнес-cost, складывать её с customer timing value можно только как joint objective, **не bank contribution margin**.

## Sensitivity смеси без изменения cohort composition

{table(sensitivity.reset_index(),['mix','universal','group_aware','group_minus_universal'])}

Для каждого mix одни и те же веса применены ко всем arms **до вычисления ratios**. Policy fit и бюджет main-mixture не перенастраиваются под новое распределение: sensitivity показывает переносимость одной политики, а не выигрыш от смены состава выборки. Formal population weights банка отсутствуют. Нет признака-указания национальности/этничности клиента: в production сегмент определяется только собственным consented поведением.

## Выборы и воспроизводимость

{table(pd.DataFrame(selected),['test_year','segment','universal','group_aware','unconstrained'])}

`selected_policy_receipts.json` хранит fit years, max prior date и все абсолютные thresholds. `manifest.json` — SHA256 пяти input files, rates и V3 simulator; исходники не мутировали. Seven meaningful tests: poison future scores, chronological schedule prefix invariance, label maturity h20, zero response with unchanged worlds, future/invalid preview refusal, urgency and cap persistence.

Команды: см. README. По умолчанию mode=universal. `preview.build_segment_policy_preview` композиционно использует V3 behavior API и selected receipt; оставляет probability как есть, проверяет доступность receipt/scores/context и добавляет причины threshold/cadence. Даже bank_observed context возвращает production_eligible=False. Новая ветка не меняет sealed V3/final_solution.

## Продуктовое решение

1. Оставить universal market policy + readiness базовой исследовательской arm; сегменты S1–S5 определяют потребность/экран, а не автоматически вероятность выгоды.
2. Frequent cadence можно показывать как отдельный явно обозначенный выбор «больше возможностей / ниже доля h5 удачных дат». Здесь оценён tradeoff, а не универсальный выигрыш.
3. Для monthly/occasional искать **событие готовности**, затем оценивать рынок достаточно часто в этом окне; долгий календарный cooldown до readiness не принимать без проверки.
4. Чтобы заявить причинную пользу, нужен prospective user-level randomized holdout с неизменным общим budget, real executable quote, delivered/opened/intent/purchase, deadline и net amount. Randomize profile policy, проверять incremental completed-transfer volume и bank CM отдельно от переноса уже запланированных переводов. Срочные случаи не включать в wait-treatment.
5. Самый сильный новый evidence asset — реально доступная панель Кыргызстана: она позволяет проверять missing months и persistence и проектировать irregular recurrent simulator. Но 7/14-дневная frequent frequency и push response по этим данным не идентифицируются.
'''
 (HERE/'REPORT.md').write_text(text)
 hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name not in ('artifact_manifest.json',)}
 (HERE/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(primary[['policy','contacts','h5_quality','net_scenario_value_rub']].to_string(index=False));print(intervals)
if __name__=='__main__':main()
