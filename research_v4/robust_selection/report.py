from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def table(df,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows():
        vals=[]
        for c in cols:
            v=r[c];vals.append('—' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)
def main():
    y=pd.read_csv(HERE/'by_year.csv');s=pd.read_csv(HERE/'summary.csv');rank=pd.read_csv(HERE/'selection_ranking.csv');selections=json.loads((HERE/'selection.json').read_text());cis=pd.read_csv(HERE/'paired_intervals.csv')
    text='''# Финальный отбор по устойчивости: 2024–2026, H3 и H5

Выполнено13 новых финальных обучений (7 TabM и6 HGB) плюс7 внутренних обучений TabM для выбора эпох;5 прежних H5 checkpoints переиспользованы с точной проверкой. Проведены три ежегодных переноса обучения для двух типов моделей — V3 и TabM — с отдельными моделями H3/H5. Полные метрики всех сравнений, включая предыдущие отчёты, собраны в **REPORTING_ALL_METRICS.csv**. Источник и протокол каждой строки сохранены: таблица объединяет результаты для отчётности, а не превращает несовместимые эксперименты в один тест.

## Протокол ежегодного обучения

| Тест | Обучение (плановые границы) | Калибровка и выбор порога |
| --- | --- | --- |
| 2024 | 2013–2022 | 2023 |
| 2025 | 2014–2023 | 2024 |
| 2026, доступная часть | 2015–2024 | 2025 |

Границы фактических дат могут быть уже из-за наличия признаков и созревания меток. Train длится 120 месяцев до отдельного 12-месячного calibration; он не включает calibration. Никакие исходы тестового года не выбирают число эпох, калибратор или порог этого года. Все имеющиеся годы уже исследовались ранее: это ретроспективная проверка, не новый нетронутый holdout. 2026 неполный, год не экстраполируется: тестH5 —13января–25августа, H3 —13января–27августа. Для2024 тестначинается10января иоканчивается24декабря(H5)/26декабря(H3); для2025 теже календарные границы.

H3 и H5 обучены отдельно с собственным target, label_available_date и purge. Основная таблица оценивает родной горизонт модели. Отдельный cross_horizon.csv оценивает фиксированные сигналы обеих моделей по обоим исходам на общем наборе зрелых H5 дат; это диагностика, а не обучение другого горизонта. Brier/logloss/AUC для чужого горизонта не выдаются как качество вероятностного прогноза этого события.

V3 — настоящий исходный pooled BASE15 HGB, исходная общая Platt-калибровка. Строгий контроль всегда использует 0,50. TabM — расширенные FULL33 (OXR/Halyk/Treasury), фиксированный seed2, та же архитектура и настройка обучения. Для всех пяти коридоров используется pooled TabM, для локального KZT дополнительно обучен KZT-only TabM. KZT-only не имеет результата «все коридоры». Поэтому это сравнение готовых систем; чистое сравнение архитектур с одинаковыми признаками находится в предыдущем architecture_2023_2025.

## Запрошенное сравнение двух типов моделей

Для TabM здесь фиксировано прежнее правило cadence85; для V3 — строгий порог 0,50. В каждой строке видно, какая обучающая популяция используется. Все доли приведены в шкале 0–1.

'''
    cols=['year','config_id','train_horizon','evaluation_scope','hit_rate','lift','forward_delta_bps','regret_bps','signals','signals_per_corridor_week','week_coverage','brier']
    q=y[((y.config_id=='v3')&(y.policy=='strict05'))|((y.config_id.str.startswith('tabm'))&(y.policy=='cadence85'))]
    text+=table(q.sort_values(['train_horizon','year','evaluation_scope','config_id']),cols)
    text+='\n\n## Правила уведомлений\n\n'
    text+='''Строгая 0,50 остаётся контрольной политикой. cadence80/85/90 выбирают фиксированный порог на предшествующем году, стремясь обеспечить соответствующее покрытие. rank80/90 вместо постоянного порога используют относительное место текущего score среди последних 63 прошлых score того же коридора. Квантиль выбирается только на calibration; тестовые метки и будущие score не используются. rank90_shared переносит один квантиль между всеми коридорами. Пропуски недель не вызывают принудительную отправку.

У всех правил максимум два контакта на коридор за календарную неделю; cooldown=2 сессии, у strict05=3. Для рангов используется минимум20 прошлых score, calibration прогревается63 последними доступными датами до начала calibration (включая purged train tail), оценёнными тем же checkpoint; это обучающая справочная история, а не новые out-of-sample примеры. Для теста используются реальные предшествующие score, включая дни без ещё созревших исходов. Правило cadence90 не гарантирует 90% будущих недель: это цель при настройке на прошлом.

## Отбор по устойчивости

До расчёта зафиксированы gates в каждой ячейке год×коридор: coverage≥0,80, lift≥1,30 и forward_delta_bps>0. Предпочтение coverage≥0,90 во всех ячейках. Среди прошедших — лучший худший lift, затем худшая выгода и Brier. Среднее не скрывает провалы отдельных периодов или коридоров. Если ни один вариант не проходит, выдаётся явно помеченный fallback по нормированному дефициту gates; он не объявляется удовлетворяющим требованиям.

Первый отбор использует только 2024–2025, после чего проверяется на 2026. Второй отбор на всех трёх годах — отдельная ретроспективная рекомендация. Это не два независимых подтверждения.

'''
    text+=table(pd.DataFrame(selections),['period','evaluation_scope','config_id','train_horizon','policy','selection_status','min_cell_coverage','min_cell_lift','min_cell_forward_delta_bps','passed_cells','cells'])
    text+='\n\n### Годовые метрики выбранных правил\n\n'
    chosen=[]
    for sel in selections:
        z=y[(y.config_id==sel['config_id'])&(y.train_horizon==sel['train_horizon'])&(y.policy==sel['policy'])&(y.evaluation_scope==sel['evaluation_scope'])].copy();z['selection_period']=sel['period'];chosen.append(z)
    text+=table(pd.concat(chosen),['selection_period','year','config_id','train_horizon','policy','evaluation_scope','lift','hit_rate','week_coverage','signals_per_corridor_week','forward_delta_bps','regret_bps','brier'])
    text+='\n\nДля KZT рекомендован TabM H5 + rank90 как компромисс на трёх периодах: coverage82,35/96,08/93,94%, lift1,469/1,450/1,368. За2024–2026 вместе:173сигнала, hit rate37,57%, lift1,42745, coverage90,37%,1,2815сигнала/неделю, выгода33,68bps. Это среднее покрытие90,37%, ане90% вкаждомгоду. Веса те же, улучшение частоты даёт политика; в2025 выгода падает с44,96 до12,63bps. Абсолютное среднее будущего изменения на выбранных днях2025 равно−15,61bps: положительная разность к базе не означает фактическую положительную прибыль.\n\nОтбор только по2024–2025 выбрал другой KZT вариант: pooled TabM H3 + cadence85. В2026 его lift1,2914 не проходит порог1,3. Ретроспективную рекомендацию H5 нельзя выдавать за этот заранее замороженный выбор.\n\nУниверсальный вариант pooled TabM H3 + rank80 держит агрегированный lift1,455/1,462/1,402 при coverage82,35/90,59/90,30%, но проваливает KZT2026:lift1,2698. Универсальной модели, прошедшей все15 ячеек, среди кандидатов нет. Варианта с90% покрытия во всех трёх KZT годах и остальными gates также нет.\n\n### Все кандидаты, первые 10 в каждой задаче отбора\n\n'
    text+=table(rank[rank['rank']<=10],['period','evaluation_scope','rank','config_id','train_horizon','policy','qualified','min_cell_coverage','min_cell_lift','min_cell_forward_delta_bps','gate_shortfall'])
    text+='\n\n### Условные доверительные интервалы\n\n'
    text+=table(cis,['year','evaluation_scope','train_horizon','baseline','candidate','lift_delta','lift_ci_low','lift_ci_high','delta_brier','ci_low','ci_high','forward_delta_bps_delta','forward_delta_bps_ci_low','forward_delta_bps_ci_high'])
    text+='\n\n### Выгода относительно базового дня, интервалы после отбора\n\n'
    if (HERE/'selected_utility_intervals.csv').exists():
        text+=table(pd.read_csv(HERE/'selected_utility_intervals.csv'),['config_id','train_horizon','evaluation_scope','policy','year','forward_delta_bps_ci_low','forward_delta_bps_ci_high'])
        text+='\n\nДля финального KZT-варианта2025 интервал [−1,53;+24,83]bps включаетноль. Поэтому прохождение точечных эксплуатационных gates не означает, что доказано значимое преимущество выгоды вкаждомгоду. Полностью закрытым требование кейса о значимой выгоде по всемпериодам считать нельзя.\n'
    text+='''

Парные 95% интервалы: 10 000 месячных bootstrap-перевыборок, стратифицированных по году, общие блоки для обеих моделей. Базовая частота пересчитывается внутри каждого draw. Интервалы условны на зафиксированных обучениях и правилах, не включают поиск моделей/политик и не исправлены на множественные сравнения. Они не подтверждают автоматически статистическую значимость каждого положительного результата.

## Числа коллеги

H5: 60 сигналов V3strict на KZT за 2023–2025, 34 попадания, hit rate0,566667, lift2,045288, выгода94,112586bps, regret42,934610bps. H3: те же60 сигналов, 39 попаданий, hit rate0,65, lift1,921939, выгода81,910520bps. Пара воспроизводится при пересчёте H5-сигналов по H3-исходам. Это не показатели отдельно обученной H3-модели.

Для прежнего KZT2023–2025: 59 недель с1–2 сигналами из152 дают coverage0,388158, 60/152 дают частоту0,394737. Прежнее среднее покрытия по полным неделям0,389455782 воспроизводится и совпадает с коллегой. Его частота0,401146132 точно не воспроизведена: без исходного кода её знаменатель не установлен. Это не различие качества H3/H5. Различия недельных метрик зависят от знаменателей. Здесь используется целое число наблюдаемых календарных недель внутри каждого тестового годового окна, включая нулевые/частичные недели. Ранее signals_per_corridor_week мог использовать дробное число недель по календарному span, а mean_cell_week_coverage — невзвешенное среднее ячеек. Обе величины не следует смешивать. Для отчётности сохранены week_cells, weeks_with_1_2, week_coverage и mean_cell_week_coverage.

## Определения метрик и ограничения

Lift = попадания / ожидаемые попадания, где ожидание рассчитано из базового hit rate каждого года×коридора и числа сигналов модели в этой ячейке. forward_delta_bps — средняя разность будущего среднего курса относительно базового изменения в той же ячейке; это не реализованная прибыль. Regret — отклонение от лучшей будущей цены с возможностью покупки сейчас. Hit rate — доля сигналов, для которых текущее значение не хуже минимума следующих H сессий. Brier/logloss меньше — лучше. Абсолютные hit rate и Brier разных H относятся к разным событиям и напрямую не ранжируют горизонты.

Устойчивость параметров не означает неизменные веса: в ежегодных folds веса переобучаются, а рецепт, набор признаков, механизм калибровки и отбора переносятся. Между коридорами pooled веса общие; обычные пороги могут быть локальными, shared-rank использует общий квантиль. На уровне клиента отдельный общий лимит уведомлений и релевантность перевода остаются отдельной продуктовой задачей. Наличие NOW-сигнала не доказывает «окно закрывается»; этот вердикт требует отдельной проверенной метки/модели.

## Артефакты

- REPORTING_ALL_METRICS.csv — единый CSV, source_report и row_type обязательны при фильтрации.
- by_year.csv — все новые годовые показатели all5/KZT.
- by_year_corridor.csv — все отдельные коридоры; summary.csv — три года вместе.
- cross_horizon.csv — оба исхода для фиксированных сигналов на общих датах.
- selection_ranking.csv / selection.json — полный воспроизводимый отбор.
- predictions.csv.gz / policies.json / calibration.json — аудит решений.
- v3/ и tabm/ — модели, временные границы и сырые прогнозы.
- audit/ — независимая проверка.
'''
    (HERE/'REPORT.md').write_text(text)
    # Wide union; source-specific fields retained without pretending common definitions.
    paths=[(HERE/f,'robust_selection',typ) for f,typ in [('summary.csv','aggregate'),('by_year.csv','annual'),('by_year_corridor.csv','year_corridor'),('cross_horizon.csv','cross_horizon'),('selection_ranking.csv','selection'),('paired_intervals.csv','paired_interval'),('selected_utility_intervals.csv','selected_utility_interval'),('retrospective_paired_intervals.csv','post_selection_paired_interval')]]
    paths += [(ROOT/'research_v4/architecture_2023_2025'/f,'architecture_2023_2025',typ) for f,typ in [('summary.csv','aggregate'),('by_year.csv','annual'),('paired_intervals.csv','paired_interval')]]
    paths += [(ROOT/'analysis_notes/v3_lift_reconciliation'/f,'v3_lift_reconciliation',typ) for f,typ in [('bridge.csv','historical_bridge'),('trained_horizons_2026.csv','trained_horizons')]]
    paths += [(ROOT/'research_v4/final_sprint'/f,'final_sprint',typ) for f,typ in [('leaderboard.csv','leaderboard'),('horizon_execution_sensitivity.csv','cross_horizon_sensitivity'),('quarter_sensitivity.csv','quarter_sensitivity')]]
    paths += [(HERE/'v3'/f,'robust_selection_v3_diagnostics',typ) for f,typ in [('original_metrics.csv','v3_original_diagnostic'),('colleague_h3_reproduction.csv','colleague_reproduction'),('matched_horizon_rescore.csv','v3_horizon_rescore')]]
    frames=[];sources=[]
    for p,exp,typ in paths:
        if not p.exists():continue
        z=pd.read_csv(p);z.insert(0,'source_report',str(p.relative_to(ROOT)));z.insert(1,'experiment_protocol',exp);z.insert(2,'row_type',typ)
        if exp=='architecture_2023_2025':z['train_horizon']=5;z['evaluation_horizon']=5
        if exp=='final_sprint':
            z['train_horizon']=5;z['evaluation_horizon']=z['horizon'] if 'horizon' in z else 5;z['evaluation_scope']='KZT'
        if exp=='v3_lift_reconciliation':
            z['train_horizon']=z['h'] if 'h' in z else 5;z['evaluation_horizon']=z['train_horizon']
        if exp=='robust_selection_v3_diagnostics':
            if 'evaluate_horizon' in z:z['evaluation_horizon']=z['evaluate_horizon']
            elif 'train_horizon' in z:z['evaluation_horizon']=z['train_horizon']
        if 'evaluation_scope' not in z and 'scope' in z:z['evaluation_scope']=z['scope'].replace({'all':'all5'})
        frames.append(z);sources.append(dict(path=str(p.relative_to(ROOT)),rows=len(z),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    union=pd.concat(frames,ignore_index=True,sort=False);union.to_csv(HERE/'REPORTING_ALL_METRICS.csv',index=False)
    (HERE/'reporting_sources.json').write_text(json.dumps(dict(rows=len(union),columns=list(union.columns),sources=sources),indent=2))
    print('Report and union CSV',len(union),'rows',len(union.columns),'columns')
if __name__=='__main__':main()
