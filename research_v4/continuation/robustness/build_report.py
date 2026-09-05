"""Generate paired diagnostics and Russian report without changing frozen inputs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json,re
import numpy as np
import pandas as pd
from research_v4.continuation.robustness import experiment as e
HERE=e.HERE;OUT=e.OUT

def table(frame,cols):
 f=frame[cols].copy()
 for c in cols:
  if pd.api.types.is_float_dtype(f[c]):f[c]=f[c].map(lambda x:f'{x:.6f}')
 return '| '+' | '.join(f.columns)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in f.to_numpy())

def paired_intervals(pred):
 comparisons=[('halyk_l1',0,'v3_long_globalcal',0),('halyk_l1',1,'v3_long_globalcal',0),('minimax_shrink_v3',0,'halyk_l1',0),('minimax_shrink_v3',1,'halyk_l1',1),('minimax_shrink_v3',1,'v3_long_globalcal',0),('v3_long_localcal',0,'minimax_shrink_v3',1)]
 samples={'development':pred[pred.date<'2026-01-01']}
 for cutoff in ('2026-01-01','2026-03-01'):samples['common_march5_'+cutoff]=pred[pred.cutoff.eq(cutoff)&(pred.date>='2026-03-05')]
 result=[]
 for scope,data in samples.items():
  for a,ad,b,bd in comparisons:
   x=data[data.rule.eq(a)&data.delay.eq(ad)].merge(data[data.rule.eq(b)&data.delay.eq(bd)],on=['date','corridor','cutoff'],suffixes=('_a','_b'),validate='one_to_one').sort_values('date')
   assert np.array_equal(x.target_a,x.target_b);delta=((x.probability_a-x.target_a)**2-(x.probability_b-x.target_b)**2).to_numpy();n=len(delta)
   for block in (10,20,40):
    rng=np.random.default_rng(6631);boot=[]
    for _ in range(1500):
     starts=rng.integers(0,n,size=int(np.ceil(n/block)));indices=((starts[:,None]+np.arange(block))%n).ravel()[:n];boot.append(delta[indices].mean())
    lo,hi=np.quantile(boot,[.025,.975]);result.append(dict(scope=scope,rule_a=a,delay_a=ad,rule_b=b,delay_b=bd,dates=n,block_cbr_dates=block,delta_brier=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),interpretation='negative favors A; paired circular date-block diagnostic, selected policies held fixed, no search correction'))
 return pd.DataFrame(result)

def main():
 base=pd.read_csv(OUT/'all_predictions.csv.gz',parse_dates=['date']);age=pd.read_csv(OUT/'age_predictions.csv.gz',parse_dates=['date']);dev=pd.concat([pd.read_csv(OUT/'development_summary.csv'),pd.read_csv(OUT/'age_development_summary.csv')],ignore_index=True);yearly=pd.concat([pd.read_csv(OUT/'metrics_by_cutoff.csv'),pd.read_csv(OUT/'age_metrics_by_cutoff.csv')],ignore_index=True);common=pd.concat([pd.read_csv(OUT/'common_march5_metrics.csv'),pd.read_csv(OUT/'age_common_march5_metrics.csv')],ignore_index=True)
 dev.to_csv(OUT/'combined_development.csv',index=False);common.to_csv(OUT/'combined_common_march5.csv',index=False);interval=paired_intervals(base);interval.to_csv(OUT/'paired_brier_intervals.csv',index=False)
 age_exposure=age.groupby(['cutoff','rule','delay']).agg(dates=('date','size'),fallback_share=('fallback','mean'),age_cutoff=('age_cutoff','first'),median_quote_age=('age_calendar_days','median')).reset_index();age_exposure.to_csv(OUT/'age_exposure.csv',index=False)
 selection=json.loads((OUT/'selection.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());receipts=json.loads((OUT/'model_receipts.json').read_text());verification=json.loads((OUT/'verification.json').read_text())
 choose=dev.pivot(index='rule',columns='delay',values='brier').rename(columns={0:'normal_brier',1:'delayed_brier'});choose['worst_brier']=choose.max(axis=1);choose['delay_penalty']=choose.delayed_brier-choose.normal_brier;choose.to_csv(OUT/'headline.csv')
 timelines=pd.DataFrame([{k:r[k] for k in ['cutoff','train_start','train_end','train_max_label_available','calibration_start','calibration_end','calibration_max_label_available','history_end','train_rows','kzt_calibration_dates','blend_alpha']} for r in receipts])
 main_rules=['v3_long_globalcal','v3_long_localcal','halyk_l1','minimax_shrink_v3'];summary_common=common[common.rule.isin(main_rules)];metriccols=['rule','delay','brier','candidate_count','candidate_quality','candidate_lift_standardized','forward_delta_bps'];h=choose.loc['halyk_l1'];r=choose.loc['minimax_shrink_v3'];b=choose.loc['v3_long_globalcal']
 report=f'''# KZT Halyk: устойчивость к задержке и две даты фиксации модели

**Главная поправка к прежней интерпретации V4:** train-lag1 → test-lag2 даёт Brier **0.181404** на development 2023–2025. Значение **0.186250** было получено при отдельном переобучении на lag2. Это разные эксперименты. Настоящая задержка входных данных исходной lag1-модели ухудшает Brier с0.176324 до0.181404, но он остаётся ниже long V3 0.182705. Из старого retrain-lag2 результата нельзя было заключать, что тот же deployed lag1 checkpoint обязательно теряет преимущество.

**Улучшения без потери качества не найдено.** Предобъявленный minimax выбор из четырёх новых защитных методов — `minimax_shrink_v3`. Он снижает worst-view Brier с{h.worst_brier:.6f} до{r.worst_brier:.6f} (на{h.worst_brier-r.worst_brier:.6f}), но повышает normal Brier на{r.normal_brier-h.normal_brier:.6f}. Ни один новый метод не выполнил одновременно normal≤original lag1 и delay≤long V3. Исходный Halyk lag1 проходит этот более мягкий delay-vs-V3 критерий сам; это не означает отсутствие чувствительности к задержке.

У простого контроля **long V3 + KZT-only calibration** Brier0.180435 в обоих режимах. По worst-view он даже немного лучше новой minimax blend0.180753. Поэтому выбор внутри robust-family не означает превосходства над всеми простыми controls. Приоритет зависит от того, важнее ли normal качество или гарантируемая в пределах проверенных сценариев нечувствительность к Halyk.

Все результаты — **ретроспективное исследование уже просмотренной истории**, не prospective holdout, не causal uplift клиента и не доказательство заработка на исполнимой банковской котировке. Старые данные и алгоритмы V3/V4 неизменны. Независимый verifier проверяет прежний V4 seal из `continuation/previous_v4_manifest.json`: только две обновляемые root-страницы README/REPORT сверяются по byte-identical backups, все остальные старые файлы — по текущим оригиналам. V3 проверяется полностью по прежнему manifest.

## Что предобъявлено

`PROTOCOL.md` записан до вычисления continuation результатов. Один target h5 KZT, один long120m HGB с прежними параметрами. Восемь правил:

| Правило | Fit / calibration | При normal и delayed deployment |
|---|---|---|
| V3 global calibration | Pooled HGB, предыдущая validation всех5коридоров | Halyk не используется |
| V3 local calibration | Тот же HGB, предыдущая KZT validation | Halyk не используется |
| Halyk lag1 | Исходный V4 residual_shrink fitL1/calL1 | Тот же checkpoint получает L1 либо L2 |
| Retrain lag2 reference | Отдельный fitL2/calL2 | Получает L2 либо L3; не равен mismatch предыдущей строки |
| Lag augmentation | По детерминированному date hash training дата выбирает L1 либо L2 | Одна строка на market-date/corridor, нет фиктивного удвоения markets |
| Feature dropout | На25% training dates все Halyk features заменены missing | Обычные доступные L1 либо L2 features на inference |
| Lag ensemble | Среднее двух отдельно обученных L1/L2 моделей | Views(L1,L2) либо(L2,L3) |
| Minimax shrink to V3 | Alpha∈{{0,.25,.5,.75,1}} по worst предыдущей validation | Смешиваются calibrated HalykL1 и global-calibrated V3 |

Pooled HGB: depth2,120 iterations,learning rate.05,leaf40,L2=2,early_stopping=False. KZT continuation:40 Newton logloss stumps,depth1,leaf60,learning rate.025, frozen **float64** pooled probabilities. Для augmentation/dropout stage-2 weight выбирается по worst validation L1/L2. Robust Platt calibrator использует две возможные views одних validation дат; это loss augmentation, **не новые независимые labels**. Пороги candidates выбираются только на прошлой нормальной validation, затем сохраняются.

Primary robust-family rule выбирается по минимальному max(normal,delay) Brier development2023–2025; selection сохранена **до нового вычисления2026**. Никакие настройки не подбирались по новым2026test outcomes. Уже известный ранее2026 всё равно не становится нетронутым holdout.

## Сравнение на всех727development KZT датах

{table(choose.reset_index(),['rule','normal_brier','delayed_brier','worst_brier','delay_penalty'])}

Возрастные fallback arms — отдельно предобъявленный после основного результата follow-up, подробно ниже. Они не входят в первоначальный robust-family selection.

## Качество прогноза и качество выбранных дат

{table(dev,metriccols)}

Brier оценивается на **всех** датах. Candidate quality — только среди отправленных моделью сигналов. Это разные метрики; рост hit rate при меньшем числе сигналов не обязательно улучшает прогноз.

`candidate_lift_standardized` полностью совместим с V3: число hits делится на сумму annual-cell baseline hit probabilities по выбранным датам. Каждая year×corridor cell получает именно её candidate-count вес. `forward_delta_bps` — средняя разность forward return кандидата и baseline mean той же cell. В CSV сохранены `candidate_lift_unstandardized` и `forward_bps_absolute`; они не подменяют стандартизированные сравнения. На single-cutoff KZT различие lift исчезает, но при pooling2023–2025 оно существенно. Отдельный численный тест воспроизводит V3 standardized lift и forward delta.

Candidate policy сохраняет прежний cooldown: не чаще одной возможности через более3effective CBR sessions. Порог выбирается на prior validation по прежней cadence/loss функции. Это один KZT candidate stream; общий bank CRM/user cap здесь не моделируется.

## January freeze и March freeze

{table(timelines,['cutoff','train_start','train_end','train_max_label_available','calibration_start','calibration_end','calibration_max_label_available','history_end','blend_alpha'])}

- **January2026 freeze:** training заканчивается2024-12-24, последние train labels созревают до2025. Calibration берёт2025 outcomes, завершившиеся к2025-12-31. Никаких2026labels в fit.
- **March1,2026 freeze:** rolling calibration `[2025-03-01,2026-03-01)`, отдельно training10years до2025-03-01. Calibration может использовать January/February2026 observations только с известными к cutoff labels: последний calibration observation2026-02-20, его h5 заканчивается2026-02-28. Это другая freeze scheme, не January-модель с незаметно обновлённым порогом.
- Evaluator **не вызывает `core.run_fold`**, потому что его `validation_history` привязана к январю. History cooldown явно равна `[cutoff−12months,cutoff)`, включая незрелый label tail только для past scores/state, без использования его outcomes.
- One-day delay начинается на cutoff. Normal и delayed world получают одну и ту же исходную cooldown state, рассчитанную на нормальной предшествующей истории. Это mismatch stress после фиксации модели, а не переобучение в другом мире.
- Самостоятельный V3 baseline реально переобучен для каждой freeze scheme. January и March сравниваются на общей сетке **2026-03-05—2026-08-25,120KZTдат**. March1 policy также имеет122даты начиная2026-03-03; две доMarch5 учитываются в его carry state. January policy содержит156зрелых2026дат начиная2026-01-13.
- Frozen CBR source заканчивается **2026-09-01**, последний зрелый h5 target — **2026-08-25**. Это не полные шесть месяцев до конца сентября; отсутствующий хвост не заполняется.

### Одинаковые120дат послеMarch5

{table(summary_common,['cutoff',*metriccols])}

Предварительно выбранный minimax blend на January freeze:0.234606normal/0.238117delay против0.238843V3. На March freeze:0.234444/0.237466 против0.238962same-cutoffV3. Небольшой положительный результат delay protection в2026 описывается **как retrospective confirmation**, с неопределённостью ниже. Halyk lag1 на January freeze без delay заметно лучшеblend, а при delay хужеV3. Это реальный tradeoff.

Полные2026таблицы всех arms сохранены в `metrics_by_cutoff.csv` и `combined_common_march5.csv`. Не выбираем удачный dropout или другой метод по тому, что он лучше выглядит на уже увиденном2026.

## Age-aware fallback: полезная гипотеза, отрицательный итог

После основного результата добавлены только два правила (`AGE_FOLLOWUP_PROTOCOL.md`): при stale/missing Halyk использовать V3 либо уже выбранный minimax. Age — максимальный calendar age **доступной observation date** для personalRUB,legalRUB,personalUSD; отсутствие любого необходимого feed даёт unavailable. Сам gate получает только ages и probabilities, **не normal/delayed flag**. Порог age>2или>3calendar days выбирается по worst purged prior-validation Brier. Никакой дополнительной Platt-калибровки после переключения нет: на свежих датах probability точно равна исходной HalykL1.

Development fallback→V3:0.176848normal/0.181512delay; fallback→minimax:0.176764/0.181889. Оба варианта немного хуже исходногоHalykL1 в обоих режимах. Это не доказательство бесполезности настоящего feed-health gate; age из исторического chart snapshot лишь proxy, а публикационные timestamps и delivery logs отсутствуют. **Расширять поиск до положительного результата не стали.**

{table(age_exposure,['cutoff','rule','delay','age_cutoff','fallback_share','median_quote_age'])}

Эта идея появилась после просмотра новых2026результатов и прямо помечена как post-inspection follow-up. Выбор age threshold использует только предыдущую validation;2026test labels не участвуют, но историческая исследовательская зависимость сохраняется.

## Неопределённость на реальных рыночных датах

Paired circular block bootstrap,1500replicates, block10/20/40effective CBR dates. В каждой паре одни и те же даты и labels; отрицательная delta означает меньший Brier уA. Fixed policies held constant, **без refit и без поправки на весь накопленный поиск**; интервалы диагностические, не confirmatory significance.

{table(interval[interval.block_cbr_dates.eq(20)],['scope','rule_a','delay_a','rule_b','delay_b','dates','delta_brier','ci_low','ci_high'])}

Все приведённые 95% интервалы при block=20 включают ноль. В частности, development преимущество minimax при delay против исходного HalykL1 составляет −0.000652, но интервал равен [−0.008412; 0.007977]. Для delayed minimax против V3 на общих датах 2026 года интервалы также включают ноль: January freeze [−0.006371; 0.003766], March freeze [−0.006294; 0.002365]. Знак среднего здесь не является статистическим подтверждением превосходства.

727development и120common2026дат — реальные исходные рыночные observations, зависимые во времени. Training lag augmentation не увеличивает их число. Нельзя считать две lag views или несколько models независимыми новыми рынками.

## Независимые алгоритмические проверки

Статус: **{verification['status']}**, {len(verification['checks'])}блоков. `verify.py` проверяет:

1. Все прежние sealed V3/V4 данные, алгоритмы и source fingerprints неизменны; две root-страницы V4 проверены в byte-identical backups, прежний V4 manifest сохранён отдельно от обновляемого root seal.
2. Сохранённые checkpoints воспроизводят probabilities через **независимую** сборку HGB + Newton corrections + Platt coefficients; candidate stream и cutoff-specific history state воспроизводятся отдельным циклом.
3. Annual2023–2026 controls совпадают с frozenV3/V4 на883KZTдатах: probability error≤1.12e−16,0candidate mismatches. Это включает originalHalykL1 и отдельноretrainedL2, поэтому их сравнение не меняет даты/семантику модели.
4. Все train/calibration labels завершаются до следующего cutoff; end dates проверяются независимо через session positions, а не доверием к полюlabel_available_date.
5. После cutoff все market features×100, все будущие и ещё не созревшие labels инвертируются; повторный fit сохраняет модели, calibrated predictions, shrink weights, candidate thresholds и age-gate parameters. ПровереныJanuary иMarch отдельно.
6. Изменение будущих rawHalyk values не меняет прошлые признаки и observation-age metadata дляlag1/lag2/lag3. Future test rows/outcomes не меняют prefix candidates.
7. Неравные годовые base rates в ручномfixture дают standardizedlift1.6 вместоnaive2.5; реальныйV3 lift/forward delta воспроизводится.

`results/model_receipts.json` содержит training/calibration timestamps, чекпойнты, SHA256 и feature fingerprints; `results/selection.json` отделяет development selection от2026. `results/verification.json` содержит детали проверок. Snapshot as-of joining и poison tests не доказывают действительную историческую доступность опубликованных данных: archive/vintage publication time остаётся ограничением.

## Практическое решение

- Исправить прежнюю формулировку lag sensitivity: **retrainedL2 и trainL1/testL2 mismatch — разные величины**.
- Если нужна лучшая normal Brier в проверенной истории, исходный HalykL1 остаётся сильнее новых защит. Нельзя обещать отсутствие просадки при задержке.
- Если важнее полная независимость от Halyk feed, простой longV3 с KZT calibration — обязательный control; он не уступает новой minimax worst-score наdevelopment.
- Minimax blend — прозрачный компромисс: меньшая delay penalty ценой normal качества. Он пригоден для следующей prospective проверки с надёжными timestamps, но не получает production promotion из этого backtest.
- Age fallback сохранили как отрицательный эксперимент. Новых порогов после этого не подбирали.
- Пользовательский conversion, деньги банка и реальные execution prices не оценены. All-in live quotes, feed monitoring и prospective holdout нужны отдельно.
'''
 # Separate Russian prose from adjoining dates/model tokens without altering IDs.
 report=re.sub(r'(?<=[А-Яа-яЁё])(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])(?=[А-Яа-яЁё])',' ',report)
 for before,after in [('Brier0','Brier 0'),('blend0','blend 0'),('development2023','development 2023'),('2026test','2026 test'),('2026labels','2026 labels'),('120KZT','120 KZT'),('727development','727 development'),('120common2026','120 common 2026'),('1500replicates','1500 replicates'),('10/20/40effective','10/20/40 effective'),('40 Newton','40 Newton'),('120 iterations,learning','120 iterations, learning'),('all5','all 5'),('всех5','всех 5'),('normal/','normal / '),('0.234606normal','0.234606 normal'),('0.238117delay','0.238117 delay'),('0.176848normal','0.176848 normal'),('0.181512delay','0.181512 delay'),('20effective','20 effective')]:
  report=report.replace(before,after)
 (HERE/'REPORT.md').write_text(report)
 hashes={str(p.relative_to(HERE)):e.digest(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in str(p)}
 (HERE/'artifact_manifest.json').write_text(json.dumps({'nature':'Retrospective continuation; old V3/V4 immutable','files':hashes},indent=2)+'\n')
 print(choose.round(6).to_string())
if __name__=='__main__':main()
