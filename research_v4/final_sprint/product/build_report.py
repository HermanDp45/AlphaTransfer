"""Publish the two-scenario research report and optional standalone annotation config."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np
import pandas as pd
from research_v4.final_sprint.product import closing_experiment as c
from research_v4.final_sprint.product import scenario_adapter as adapter
HERE=c.HERE;OUT=c.OUT

def table(f,cols):
 q=f[cols].copy()
 for col in cols:
  if pd.api.types.is_float_dtype(q[col]):q[col]=q[col].map(lambda x:f'{x:.6f}')
 return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in q.to_numpy())

def intervals(data):
 rows=[]
 for (cutoff,mode),g in data.groupby(['cutoff','mode']):
  for scenario,label in [('NOW','now_target'),('CLOSING','closing_target')]:
   selected=g.signal.astype(bool) if scenario=='NOW' else g.closing_annotation.astype(bool)
   cells=pd.DataFrame({'month':g.date.dt.to_period('M'),'signal':selected.astype(int),'hits':selected*g[label],'base_hits':g[label],'n':1}).groupby('month').sum()
   rng=np.random.default_rng(62913);draws=rng.integers(0,len(cells),size=(2500,len(cells)));agg=cells.to_numpy()[draws].sum(axis=1);s,h,b,n=agg.T;valid=(s>0)&(b>0);lift=(h[valid]/s[valid])/(b[valid]/n[valid]);lo,hi=np.quantile(lift,[.025,.975])
   rows.append({'cutoff':cutoff,'mode':mode,'scenario':scenario,'market_dates':len(g),'calendar_months':len(cells),'contacts_or_annotations':int(selected.sum()),'hits':int((selected*g[label]).sum()),'lift':float(g.loc[selected,label].mean()/g[label].mean()),'lift_ci_low':float(lo),'lift_ci_high':float(hi),'bootstrap_valid_replicates':int(valid.sum()),'conditional_fixed_model_month_bootstrap':True})
 return pd.DataFrame(rows)

def example(data,policies):
 g=data[data.cutoff.eq('2026-01-01')&data['mode'].eq('normal')&data.closing_annotation];row=g.iloc[-1];policy=next(x for x in policies if x['cutoff']=='2026-01-01');day=str(row.date.date());facts={'known_at':day,**{k:float(row[k]) for k in ('ret1','pr60','rub_per_unit','recent_low_rank5','change_from_recent_low_bps')}}
 candidates=[]
 for scenario,prob,model in [('NOW',row.now_probability,row.now_model_id),('CLOSING',row.closing_probability,row.model_id)]:
  candidates.append({'scenario':scenario,'target_contract':adapter.CONTRACTS[scenario],'as_of':day,'known_at':day,'corridor':'KZT','probability':float(prob),'threshold':policy['thresholds'][scenario],'past_baseline_rate':policy['past_baseline_rates'][scenario],'model_id':model,'model_cutoff':'2026-01-01','calibration_end':'2025-12-31','factual_context':facts})
 preview=adapter.resolve_scenarios(candidates,day);assert preview['selected']['scenario']=='NOW' and len(preview['annotations'])==1 and len(preview['next_state']['contacts'])==1
 c.save(OUT/'example_dual_input.json',{'as_of':day,'candidates':candidates,'context':{'routing_mode':'dual_annotations'},'state':{'contacts':[]}});c.save(OUT/'example_dual_preview.json',preview)

def config_payload(annotations,policies,receipts,jmanifest):
 policy=next(x for x in policies if x['cutoff']=='2026-01-01');r=next(x for x in receipts if x['cutoff']=='2026-01-01' and x['model_id']=='closing_treasury_halyk_shrink120m')
 return {'schema_version':1,'profile':'dual_annotations','threshold':float(policy['thresholds']['CLOSING']),'scenario':'CLOSING','primary_scenario':'NOW','target_contract':c.CONTRACT,'horizon_effective_cbr_rows':5,'materiality_bps':0,'model_id':r['model_id'],'model_cutoff':'2026-01-01','calibration_end':r['validation_max'],'latest_calibration_label':r['validation_last_label'],'policy_name':policy['policy'],'threshold_fit_uses_2026':False,'now_model_selection_uses_2026':jmanifest['selection_uses_2026'],'now_model_id':policy['now_config'],'rule':'existing NOW contact AND own calibrated CLOSING probability >= threshold AND observed ret1 > 0','own_closing_probability_required':True,'now_contacts_and_truth_unchanged':True,'extra_contacts':0,'nonexclusive_tags':True,'metric_denominators':{'NOW':'all original NOW contacts','CLOSING':'only CLOSING-annotated subset; evaluate endpoint label'},'bank_freshness_assumption':'bank inputs are fresh per user product assumption; historical delay remains diagnostic, not the selection gate','research_only':True,'annotation_rule_post_readout':True,'checkpoint_source':str((HERE/r['checkpoint']).relative_to(ROOT)),'checkpoint_source_sha256':r['checkpoint_sha256'],'joint_policy_sha256':c.sha(OUT/'joint/policies.json'),'results_by_cell':annotations.to_dict('records'),'caveats':['Retrospective model selection inspected2026; no prospective holdout claim.','CLOSING annotations were defined after joint/routing readout; thresholds remained past-only.','Secondary sample16 in2025 and21 in2026 normal; correlated dates and conditional bootstrap uncertainty.','Delayed scenarios do not both meet all targets; do not claim delay robustness.','CBR reference target, not executed transfer savings; Halyk BANK SELL RUB is a feature, not customer RUB-sell execution.','No pooled NOW/CLOSING lift and no double-counting tags as notifications.']}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--export-adapter-config',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
 pred=pd.read_csv(OUT/'joint/predictions.csv.gz',parse_dates=['date']);dual=pred[pred.variant.eq('dual_annotations')].copy();metrics=pd.read_csv(OUT/'joint/metrics.csv');ann=pd.read_csv(OUT/'joint/annotations_metrics.csv');closing=pd.read_csv(OUT/'closing_metrics.csv');policies=json.loads((OUT/'joint/policies.json').read_text());jmanifest=json.loads((OUT/'joint/manifest.json').read_text());receipts=json.loads((OUT/'model_receipts.json').read_text());verify=json.loads((OUT/'verification.json').read_text())
 ci=intervals(dual);ci.to_csv(OUT/'joint/annotation_month_intervals.csv',index=False);example(dual,policies);config=config_payload(ann,policies,receipts,jmanifest);c.save(OUT/'closing_annotation.json',config)
 exported=None
 if args.export_adapter_config:
  exported=ROOT/'final_solution/final_sprint/closing_annotation.json';exported.parent.mkdir(exist_ok=True);c.save(exported,config)
 fresh=ann[ann['mode'].eq('normal')];all_accept=bool(fresh.joint_acceptance.all());summary_columns=['cutoff','mode','combined_contacts','weeks_1_to_2_share','NOW_contacts','NOW_lift','CLOSING_contacts','CLOSING_lift']
 report=f'''# NOW и CLOSING: две головы и один поток уведомлений

**Оба сценария сохранены с разными метками и вероятностями.** Финальный исследовательский профиль `dual_annotations` оставляет каждое исходное NOW-уведомление и добавляет CLOSING как вторую аннотацию на части тех же дат. Он не увеличивает число сообщений, не меняет даты NOW и не удаляет сильные наблюдения из его метрики.

Для окончательно выбранной NOW-модели **{jmanifest['now_selection']['champion']['config_id']}**, policy **{jmanifest['now_selection']['champion']['policy']}**, точечные условия на свежих банковских данных выполнены в обоих проверенных годах: **{all_accept}**. Это ретроспективный результат после просмотра2026 и определения annotation-profile; он не равен независимому подтверждению будущей работы.

{table(ann,summary_columns)}

На2026 normal: **40 контактов,29 из33 недель с1–2 контактами (87.88%), NOW lift1.505263**. На21 из этих40 дат добавлена CLOSING-аннотация:16/21 подтверждений, **CLOSING lift1.398319**. Число контактов остаётся40, а не40+21. На2025 normal:60 контактов,45/51 недель (88.24%), NOW lift1.5125; CLOSING10/16 подтверждений, lift1.468447.

Bank-delay сохранён как диагностика. Пользователь зафиксировал продуктовую предпосылку свежих банковских данных, поэтому задержка не участвует в выборе финальной модели. Нельзя из этого заключать, что защита от задержки достигнута: delayed2025 CLOSING lift1.243861, delayed2026 NOW lift1.263158 — ниже1.3.

## Семантика кейса и границы доказательства

Кейс прямо требует разные правила: NOW — текущий курс не хуже каждого следующего значения, CLOSING — курс действительно вырос к концу горизонта. Здесь **R = RUB за единицу валюты получателя**, меньший R выгоднее при отправке рублей. Основной горизонт — **5 эффективных наблюдений CBR**, не календарные дни.

| Сценарий | Метка | Что она не утверждает |
|---|---|---|
| NOW | `R[t] <= min(R[t+1]…R[t+5])` | Не обещает реального исполнения по reference-курсу |
| CLOSING | `R[t+5] > R[t]`, порог0 | Не означает, что внутри окна не было более дешёвых дней |

Сценарии не взаимоисключающие. На пути100→99→101→102→103→105 верен CLOSING, но неверен NOW. На полностью плоском пути верен NOW, но неверен строго положительный CLOSING. Метрики и score одной головы нельзя переименовать в другую.

Текущий legacy `final_solution/alphatransfer_final/product.py` по-прежнему имеет `offline_target=NOW`; его текстовые метки описывают исторические факты. Новый дополнительный standalone профиль несёт оба model targets. Старый product.py этой задачей не менялся.

## Отдельная обученная CLOSING-голова

Сделаны **четыре fits**: Treasury-only и Treasury+Halyk; cutoffs2025 и2026. Каждый использует120 месяцев pooled training, прошлые12 месяцев KZT calibration, прежний HGB120/depth2 и40 слабых residual-деревьев KZT. Shrink weight выбирается по прошлой validation, Platt-калибровка и пороги тоже используют только созревшие прошлые labels.

Halyk здесь внешний признак. Его BANK SELL RUB side не является целевой исполнимой ценой RUB→KZT. Treasury lag7 и Halyk начинаются в2020: более ранние feature values остаются missing. При stress к тому же checkpoint после cutoff подаются Halyk L2 вместо L1; обучения или подстройки по тесту нет.

{table(closing,['model_id','cutoff','mode','brier','prior_prevalence_brier','candidate_count','closing_lift','weeks_1_to_2_share'])}

Самостоятельный CLOSING Treasury+Halyk на2026 normal: Brier0.247636 и lift1.133564. Он достигает частоты, но не целевого lift1.3. Поэтому более слабые CLOSING-сигналы не используются для заполнения пропусков в NOW-потоке.

`closing_history.csv.gz` содержит все прошлые12 месяцев scores, включая незрелый хвост только как probability/state history. Outcomes этого хвоста (`target`, `closing_target`, `now_target`, forward/endpoint bps) явно blank. Конец label проверяется по фактической пятой следующей CBR-сессии до cutoff. Test2025 —242 KZT-даты, test2026 —156 дат13 января—25 августа и33 наблюдаемые календарные недели.

## Что дали альтернативные способы объединения

Все варианты фиксированы до соответствующего дополнительного вычисления; их происхождение после просмотра прежних результатов явно сохранено в `JOINT_PROTOCOL.md` и `ROUTING_ADDENDUM.md`. Пороги на2026 не подбирались заново.

{table(metrics[metrics['mode'].eq('normal')],['cutoff','variant','combined_contacts','weeks_1_to_2_share','NOW_contacts','NOW_lift','CLOSING_contacts','CLOSING_lift'])}

Агрессивный joint ранжирует прошедшие отдельные пороги головы по probability/past-event-prevalence и выбирает один тип под общим cap. В2026 он добавляет слабые CLOSING-контакты, не улучшая недельное покрытие. Гейт наблюдаемого роста `ret1>0` и более строгий NOW-гейт по низкому рангу/снижению не дают устойчивого решения. Нельзя обязательным pr60-гейтом подменять саму обученную NOW-метку.

Эксклюзивный routing сохраняет даты NOW, но переименовывает часть сообщений в CLOSING. При этом метрика оставшегося NOW-подмножества ухудшается: на2026 normal lift1.296399. Поэтому результат нельзя выдавать за неизменный lift всего NOW-потока.

**Dual annotations** разрешают именно семантическую проблему: основной NOW-смысл остаётся у всех уведомлений, а CLOSING — дополнительное независимо проверяемое утверждение. Это не новая вероятность NOW, не дополнительные market samples и не скрытое улучшение за счёт удаления плохих дат. Параллельно сохранён прежний CatBoost joint-control со своими input и engine snapshots в `results/joint_catboost_control/`.

## Неопределённость аннотаций

Парный calendar-month bootstrap:2500 повторов; сохраняются целые месяцы, фиксированные models/thresholds и отдельные сценарные denominators. При каждом повторе заново считается random-day baseline на пересэмплированных датах. Интервалы условны на уже выбранные модели и правило; поправки на все292 model-policy comparisons и всю предшествующую историю поиска здесь нет. Малые subset16/21 annotations и временная корреляция ограничивают силу вывода.

{table(ci,['cutoff','mode','scenario','contacts_or_annotations','hits','lift','lift_ci_low','lift_ci_high'])}

Итоговые точечные gates не следует читать как доказанное стабильное превосходство. Для вторичной CLOSING-аннотации отдельный prospective контроль особенно нужен: правило сформулировано после просмотра joint/routing результатов.

## Работающий adapter и локальная интеграция

`scenario_adapter.py` — stdlib API `resolve_scenarios(...)`. Default `dual_annotations` возвращает основной NOW и отдельный список CLOSING annotations с собственными probability/model_id/target_contract. Нет реальных сообщений и мутации переданного state. Неизвестные outcome fields, future timestamps, mismatched target contract и противоречащие same-day facts отклоняются. Доступны schema `scenario_schema.json` и готовый пример `results/example_dual_preview.json`.

UI различает **модельный вердикт** и **проверенный факт**. Основная карточка: «Модельный сигнал: выгодно сейчас», дополнительная отметка: «Есть модельные признаки закрытия окна». В теле — только опубликованный курс, наблюдаемое изменение или исторический ранг; никаких гарантированных утверждений будущего курса. Если сильного исторического факта нет, остаётся нейтральная текущая reference-цена. Явный `require_strong_fact` — отдельный более строгий режим.

Клиентский adapter дополнительно проверяет общий rolling-seven-day CRM-cap и календарный cooldown, включая другие сообщения. Это отдельный слой поверх рыночного backtest с cap2/calendar-week и CBR-session cooldown. Рыночные87.88% недель не являются обещанием такой же частоты после клиентских ограничений.

Параметры аннотации готовы в `results/closing_annotation.json`{'; экспортированы в `final_solution/final_sprint/closing_annotation.json`' if exported else ''}. Root standalone scorer применяет `existing_NOW_contact AND closing_probability>=threshold AND ret1>0`. Порог CLOSING2026: **{config['threshold']:.12f}**; он выбран по созревшей validation2025. Ни контакты, ни NOW probability/label не заменяются.

## Проверки и воспроизведение

Verifier: **{verify['status']}**, {len(verify['checks'])} блоков; stdlib adapter suite —11 тестов. Независимо проверены CLOSING endpoint и maturity через actual session positions; ручная float64 сборка checkpoint + residual + Platt; одинаковые test keys; masked history tail; неизменность fit после отравления будущих features/незрелых labels; prefix-invariance сырых Halyk/Treasury; отсутствие влияния будущих outcomes на scheduler; отдельные denominators; сохранение NOW-дней и одной отправки при dual tags.

Runtime: `/private/tmp/alphatransfer-tabm-venv/bin/python`, один поток. Команды и контракты — в `README.md` и `SCENARIO_CONTRACT.md`. Старые V3/V4 snapshots не менялись; в `final_solution` записан только явно согласованный дополнительный JSON-конфиг. Полной промышленной готовности, публикационных vintage timestamps и экономии по исполненным переводам исследование не доказывает.
'''
 (HERE/'REPORT.md').write_text(report)
 if args.seal:
  files={str(p.relative_to(HERE)):c.sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in str(p)};c.save(HERE/'artifact_manifest.json',{'files':files,'external_config':{'path':str(exported.relative_to(ROOT)),'sha256':c.sha(exported)} if exported else None,'status':'complete','scenario_semantics':'NOW primary plus optional nonexclusive CLOSING annotations','not_prospective':True})
 print('CLOSING threshold',config['threshold']);print(ci[['cutoff','mode','scenario','lift','lift_ci_low','lift_ci_high']].round(6).to_string(index=False));print('REPORT READY; seal',args.seal)
if __name__=='__main__':main()
