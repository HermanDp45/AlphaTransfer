"""Paired month uncertainty and a candid report for the bank-quote tasks."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from research_v4.oxr2010_bank.bank_target import experiment as e
HERE=e.HERE;OUT=e.OUT

def table(frame,columns):
 q=frame[columns].copy()
 for c in columns:
  if pd.api.types.is_float_dtype(q[c]):q[c]=q[c].map(lambda x:f'{x:.6f}')
 return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in q.to_numpy())

def paired_intervals(pred,task):
 pairs=[('bank_cbr_oxr','bank_cbr'),('bank_cbr','persistence'),('bank_history','persistence'),('cbr_history','persistence'),('bank_cbr_oxr','persistence')]
 scopes={'development':pred[pred.date<'2026-01-01']}
 for c in ('2026-01-01','2026-03-01'):scopes['common_march5_'+c]=pred[pred.cutoff.eq(c)&pred.date.ge('2026-03-05')]
 output=[]
 for scope,frame in scopes.items():
  for left,right in pairs:
   a=frame[frame.arm.eq(left)];b=frame[frame.arm.eq(right)];q=a.merge(b,on=['date','quote_date','cutoff'],suffixes=('_a','_b'),validate='one_to_one').sort_values('date');assert np.array_equal(q.target_a,q.target_b)
   effects={'brier':(q.probability_a-q.target_a)**2-(q.probability_b-q.target_b)**2}
   if task=='future_mean':effects['mae_change_bps']=(q.predicted_change_bps_a-q.change_bps_a).abs()-(q.predicted_change_bps_b-q.change_bps_b).abs()
   for metric,d in effects.items():
    cells=pd.DataFrame({'month':q.date.dt.to_period('M'),'delta':d}).groupby('month').delta.agg(['sum','size']);sums=cells['sum'].to_numpy();sizes=cells['size'].to_numpy();rng=np.random.default_rng(7142);draws=rng.integers(0,len(cells),size=(2500,len(cells)));boot=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1);lo,hi=np.quantile(boot,[.025,.975]);output.append({'task':task,'scope':scope,'arm_a':left,'arm_b':right,'metric':metric,'dates':len(q),'calendar_months':len(cells),'delta':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi),'interpretation':'negative favors A; paired calendar-month blocks,2500 replicates,frozen fits,no accumulated-search correction'})
 return pd.DataFrame(output)

def main():
 pred=pd.read_csv(OUT/'all_predictions.csv.gz',parse_dates=['date','quote_date']);n=pd.read_csv(OUT/'now/predictions.csv.gz',parse_dates=['date','quote_date'])
 intervals=pd.concat([paired_intervals(pred,'future_mean'),paired_intervals(n,'bank_now')],ignore_index=True);intervals.to_csv(OUT/'paired_month_intervals.csv',index=False)
 dev=pd.read_csv(OUT/'development_summary.csv');common=pd.read_csv(OUT/'common_march5_summary.csv');ndev=pd.read_csv(OUT/'now/development_summary.csv');ncommon=pd.read_csv(OUT/'now/common_march5_summary.csv')
 audit=json.loads((OUT/'source_audit.json').read_text());check=json.loads((OUT/'verification.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());receipts=json.loads((OUT/'model_receipts.json').read_text())
 rows=[]
 for r in receipts:
  if r['arm']=='persistence':rows.append({k:r[k] for k in ('cutoff','train_rows','train_max','train_latest_label','validation_rows','validation_max','validation_latest_label','test_rows','test_min','test_max')})
 timeline=pd.DataFrame(rows);timeline.to_csv(OUT/'timeline.csv',index=False)
 panel=pd.read_pickle(OUT/'panel.pkl');coverage=panel.groupby(panel.date.dt.year).agg(quotes=('date','size'),mature_labels=('target','count'),cbr_coverage=('cbr_available','mean'),oxr_coverage=('oxr_available','mean'));coverage.to_csv(OUT/'coverage_by_year.csv')
 selected=intervals[intervals.arm_a.eq('bank_cbr_oxr')&intervals.arm_b.eq('bank_cbr')]
 now_vsbase=intervals[intervals.task.eq('bank_now')&intervals.arm_a.eq('bank_cbr')&intervals.arm_b.eq('persistence')]
 report=f'''# Halyk: прогноз реальной банковской котировки отдельно от CBR

**Проверили actual bank quote target и получили преимущественно отрицательный результат.** На 708 одинаковых зрелых наблюдениях development 2023–2025 прогноз средней будущей котировки хуже простых прошлых базовых прогнозов. Добавление OXR к bank+CBR ухудшает Brier и MAE. Дополнительный NOW-event даёт небольшой development выигрыш bank+CBR, который не подтверждается на общих датах 2026 года; добавление OXR также ухудшает его.

Это существенно ограничивает банковский PoC: улучшение прежнего **CBR-target** не означает доказанного улучшения предсказания котировки конкретного банка. Две задачи используют разные outcomes, prevalence и даты; их абсолютные Brier нельзя сопоставлять как «до/после» одного эксперимента.

## Что именно измеряется

Источник — архив Halyk personal **BANK SELL RUB**. Банк продаёт клиенту RUB, получая KZT. Цена **q = KZT за 1 RUB**; клиент, покупающий RUB, предпочитает меньшую q. Для нашего исходного направления, где клиент отдаёт RUB и получает KZT, нужна другая сторона — BANK BUY RUB и all-in условия перевода. Их этот архив не даёт.

У каждого фактического наблюдения с effective date s есть одна исследовательская точка решения в **12:00 Asia/Almaty на s+1 календарный день**. Это момент, когда предполагается доступной котировка q_s. Публикационный timestamp не установлен, а исполнимость вчерашней цены в момент решения не доказана. Поэтому далее речь о прогнозе **архивного банковского ценового ряда**, а не о доходе от исполнения операции.

Первичная задача:

- Горизонт — следующие **пять фактических чистых наблюдений Halyk**, без календарной интерполяции. Их средняя арифметическая цена — `future_mean_kzt_per_rub`.
- Регрессия прогнозирует `10000 × (future_mean/q_s − 1)` в bps относительного изменения. Persistence прогнозирует нулевое изменение и будущую среднюю, равную q_s. Оцениваются также ошибки в KZT/RUB.
- Классификация прогнозирует `I(future_mean > q_s)`: подорожает ли покупка RUB относительно последней известной архивной цены. Точное равенство считается 0. Это **не** исходный CBR NOW-event.
- Метка становится доступной не раньше следующего календарного дня после пятой будущей котировки. По этому моменту очищаются train/validation границы.

Все RUB quotes находятся на сетке 0.01 KZT. Независимая проверка Decimal выявила три flat-окна, где наивная floating-point арифметика ошибочно давала положительную метку. В итоговой версии сравнения используют целые ценовые тики, без подбираемого epsilon. Исправлены метки 2021-02-25, 2023-04-22 и 2024-09-26; ниже приведены только пересчитанные результаты.

## Достаточность и чистота данных

Raw API содержит **{audit['raw_rows']}** записей; ISO `date_at` и epoch milliseconds совпали во всех случаях. UTC переводится в историческую зону Asia/Almaty, включая смену UTC+6 на UTC+5 в 2024 году; неоднозначного day/month parser нет. Даты с конфликтующими значениями исключаются целиком: **{audit['conflicting_dates_excluded']}** дат. После удаления одинаковых дубликатов остаются **{audit['clean_quotes']}** реальных котировок за {audit['first_quote']}—{audit['last_quote']}.

Медианный горизонт пяти quote updates — **{audit['median_horizon_calendar_days']:.0f} календарных дней**. Три окна длиннее 14 дней исключены; пять последних anchors имеют ещё неизвестный target. Фильтр длины будущего окна — **условие ретроспективной оценки по плотности архива**, неизвестное на момент решения. Его нельзя представлять как готовый online gate. Конфликтные и отсутствующие котировки могут быть связаны с состоянием рынка; игнорируемую выборочную доступность данных этот PoC не устраняет.

{table(coverage.reset_index(),['date','quotes','mature_labels','cbr_coverage','oxr_coverage'])}

CBR и OXR присоединяются назад по known-at с максимумом возраста семь дней. CBR используется в KZT/RUB после консервативного лага effective date+1 день. OXR доступен после `max(published_at, completed UTC day)+24h`. Отсутствие внешнего источника оставляет NaN, а не удаляет неудобные test даты. Все методы имеют один и тот же bank-target cohort.

**OXR с 2010 и OXR с 2018 дали точно одинаковые признаки и отдельно обученные прогнозы для всех пяти cutoffs.** Банковских labels до 2020 года нет: добавленная ранняя OXR история не может превратить этот банковский PoC в десять лет supervised обучения. Это полезный отрицательный контроль источника дополнительной информации.

## Фиксированные модели и временные границы

Четыре HGB feature arms: только собственная история банка; только история CBR; банк+CBR; банк+CBR+OXR. Общие календарные признаки одинаковы. Истории включают известный уровень, изменения за 1/5/20 обновлений, volatility20 и rank60; совместные модели дополнительно видят известные расхождения котировок. Параметры classifier/regressor фиксированы: 120 деревьев, depth2, learning rate0.05, leaf40, L2=2, early stopping выключен. Не искали успешную конфигурацию на 2026.

Два простых контроля используют **только прошлую validation**: вероятность равна её prevalence; regression либо нулевое изменение, либо среднее прошлое изменение. Поэтому их Brier одинаков, а regression различается. Test prevalence не выдаётся за обученный baseline. HGB калибруется Platt только на прошлых двенадцати месяцах; regression обучается на train без подстройки к test.

{table(timeline,['cutoff','train_rows','train_max','train_latest_label','validation_rows','validation_max','validation_latest_label','test_rows','test_min','test_max'])}

January freeze исключает любые 2026 labels из обучения и калибровки. March freeze использует trailing validation с марта 2025, допуская только созревшие к 1 марта 2026 январские/февральские outcomes. January и March сравниваются на **одинаковых 118 банковских датах решения с 2026-03-05**, а не на 120 прежних CBR датах. Равенство пар `(date, quote_date)` проверено явно. История 2026 уже изучалась ранее; это ретроспективная проверка, не pristine holdout.

## Первичный прогноз средней будущей котировки

{table(dev,['arm','rows','brier','brier_skill_vs_past_prevalence','mae_change_bps','mae_skill_vs_persistence','mae_future_quote_kzt_per_rub'])}

Положительный skill означал бы улучшение относительно прошлого baseline. На development он отрицателен у всех четырёх HGB arms. Более ранняя внешняя рыночная история не компенсирует ограниченную историю банковской метки в этом фиксированном семействе моделей.

На одинаковых датах после 5 марта 2026:

{table(common,['cutoff','arm','rows','brier','mae_change_bps'])}

Небольшие случайные выигрыши отдельных arms в 2026 не используются для выбора победителя. Основной заранее заданный incremental contrast — добавление OXR к bank+CBR — остаётся отрицательным по точечным оценкам обеих freeze schemes.

## Дополнение: банковское событие NOW

После просмотра первичного результата отдельно зафиксирован `NOW_ADDENDUM.md`. Метка: **`I(q_s ≤ min(q_next1,…,q_next5))`**. Текущая архивная цена покупки RUB не хуже каждой из следующих пяти; равенство считается успехом. В направлении BANK SELL RUB это корректный аналог «не появится более дешёвой возможности». Он всё ещё не доказывает, что q_s можно было исполнить на s+1.

Те же шесть arms, признаки, даты и purged cutoffs; переобучена только классификация. Regression не переобучалась. Этот follow-up не подменяет отрицательный первичный результат.

{table(ndev,['arm','rows','positive_share','brier','brier_skill_vs_past_prevalence'])}

{table(ncommon,['cutoff','arm','rows','brier','brier_skill_vs_past_prevalence'])}

NOW bank+CBR немного улучшает development Brier против past prevalence, но на общих датах 2026 обе freeze schemes уступают своим past-prevalence controls. OXR не даёт incremental выигрыша ни на development, ни на этих 2026 датах.

## Парная неопределённость

Ресэмплируются **целые календарные месяцы**: 2500 bootstrap повторов, общие даты и outcomes в каждой паре. Размеры месяцев учитываются в знаменателе средней ошибки. Это сохраняет зависимость внутри месяца, но не является доказательством независимости соседних месяцев; на коротком 2026 окне доступны лишь шесть месяцев. Модели не переобучаются внутри bootstrap, поправки на весь накопленный research search нет. Отрицательная delta означает преимущество arm_a.

Incremental OXR против bank+CBR:

{table(selected,['task','scope','metric','dates','calendar_months','delta','ci_low','ci_high'])}

Дополнительный NOW bank+CBR против прошлого baseline:

{table(now_vsbase,['scope','dates','calendar_months','delta','ci_low','ci_high'])}

Development выигрыш NOW bank+CBR против прошлого baseline равен −0.001657 Brier, но 95% интервал [−0.006207; 0.002444] включает ноль. Наоборот, incremental OXR ухудшение первичного Brier имеет положительный интервал на development и в обеих freeze schemes 2026. Это аргумент против добавления OXR в данном фиксированном банковском семействе моделей; он не доказывает бесполезность источника для любых других задач или моделей.

Полные интервалы по остальным контролям находятся в `results/paired_month_intervals.csv`. Сигнал для банка нужно подтверждать на новых данных с известными timestamps и исполнимой стороной котировки; эти числа не являются пользовательским или денежным causal uplift.

## Проверки и практический вывод

Независимый verifier: **{check['status']}, {len(check['checks'])} блоков**. Проверены SHA источников; историческая временная зона; каждый target по отдельному Decimal/quote-window алгоритму; все train/cal labels относительно cutoff; as-of ограничения; точные cohort ключи; сохранённые checkpoint predictions и ручная Platt формула; неизменность fit при изменении будущих признаков и незрелых labels; source-prefix invariance; отдельные модели OXR2010/2018. Проверки включают оба target определения.

Полный первичный fit занял {manifest['seconds']:.2f} секунды на одном потоке. Старые V3/V4 файлы не изменялись. `results/model_receipts.json`, `results/now/model_receipts.json`, `results/verification.json` и локальный `artifact_manifest.json` обеспечивают воспроизведение.

По текущим результатам честная банковская формулировка: **pipeline может прогнозировать и проверять конкретную архивную котировку, но устойчивое превосходство для Halyk и пользу OXR на банковской метке мы пока не показали**. Для исходного продукта нужны actual BANK BUY RUB / all-in RUB→KZT quotes, publication/delivery timestamps, история реально исполнимых условий и затем новая prospective проверка. Перенос улучшенной CBR Brier на банковский результат недопустим.
'''
 (HERE/'REPORT.md').write_text(report)
 files={str(p.relative_to(HERE)):e.sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in str(p)}
 e.save(HERE/'artifact_manifest.json',{'nature':'Bank SELL RUB quoted-rate proxy; primary future mean and post-readout NOW control; no causal execution gain','files':files})
 print(selected[['task','scope','metric','delta','ci_low','ci_high']].round(6).to_string(index=False));print('SEALED',len(files))
if __name__=='__main__':main()
