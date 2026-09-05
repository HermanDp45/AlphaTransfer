from pathlib import Path
import pandas as pd,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def table(d,cols):
 lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for _,r in d.iterrows():lines.append('| '+' | '.join('—' if pd.isna(r[c]) else f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
 return '\n'.join(lines)
def main():
 y=pd.read_csv(HERE/'by_year.csv');s=pd.read_csv(HERE/'summary.csv');sel=json.loads((HERE/'selection.json').read_text());ci=pd.read_csv(HERE/'paired_intervals.csv')
 text='''# TabM KZT H3: длинная история и финальный профиль

По явному выбору пользователя исследуется и интегрируется **TabM KZT H3**, без замены на H5 или pooled-модель. Финальный исторический рецепт: **обучение с 2010-01-01, FULL33, seed2, последние12месяцев отдельно для калибровки, причинное правило rank80**.

Длинная история принята по устойчивости: во всех трёх годах lift≥1,3, покрытие≥80%, средняя выгода>0. Это частичное улучшение: 2025 ухудшился, а 2026 улучшился существенно. Смешивания лучших моделей задним числом по отдельным годам нет. Порог rank80 каждый раз выбирается только на предыдущем году; квантиль не подгоняется к тесту.

## Годовое сравнение на одинаковом правиле rank80

'''
 text+=table(y[y.policy=='rank80'],['config_id','year','lift','hit_rate','signals','week_coverage','signals_per_corridor_week','forward_delta_bps','regret_bps','brier'])
 text+='\n\n### Три года вместе\n\n'+table(s[s.policy=='rank80'],['config_id','lift','hit_rate','week_coverage','signals','forward_delta_bps','regret_bps','brier'])
 text+='''

В2024 годовой lift вырос1,547→1,609, но coverage снизилось86,3%→80,4%. В2025 lift снизился1,589→1,452, а выгода48,2→40,3bps. В2026 lift вырос1,235→1,441, выгода36,5→73,8bps, regret снизился64,2→38,4bps, но coverage снизилось97,0%→84,8%. Поэтому улучшение состоит в прохождении всех годовых ограничений, а не в доминировании по каждой метрике.

## Доверительные интервалы разницы

'''+table(ci[ci.policy=='rank80'],['delta_brier','ci_low','ci_high','lift_delta','lift_ci_low','lift_ci_high','forward_delta_bps_delta','forward_delta_bps_ci_low','forward_delta_bps_ci_high'])+'''

Интервалы парные, месячные,10 000 перевыборок со стратификацией по году. Интервалы разностей lift/Brier/выгоды включаютноль: статистически уверенное общее превосходство не доказано. Все периоды ранее просматривались; это ретроспективное исследование. Выбор сделан по заранее записанному правилу устойчивости, а не по одному хорошему2026.

## Полный отбор

'''+table(pd.read_csv(HERE/'ranking.csv'),['rank','config_id','policy','qualified','preferred90','min_coverage','min_lift','min_utility','gate_shortfall'])+'''

## Признаки и триггер

Используются33признака:15базовых (доходности, положение в историческом диапазоне, волатильность, CNYbasis),6OXR (basis/изменения/z-score/покрытие/возраст),6Halyk и6Treasury. Численные пропуски обрабатываются median/quantile-преобразованием, обученным только на train; дополнительно подаются33индикатора пропуска. Ранние годы не имеют Halyk/Treasury: значения не заполняются из будущего.

NOW_H3: текущая цена RUB/единицу валюты не выше минимума следующих3эффективных CBR-сессий. rank80 сравнивает текущую оценку с63предшествующими оценками того же checkpoint; минимум20исторических оценок, cooldown2сессии, максимум2кандидата на календарную неделю. Порог квантиля выбран на калибровке для80% недель, но это не гарантия будущего покрытия. Новое наблюдение добавляется в окно после решения. Состояние связано с хешами модели, калибровки и правила.

## Финальное обучение и интеграция

Финальный refit имеет as-of2026-09-05. Все доступные данные используются в одной из двух ролей: веса обучаются на истории с2010 до начала последних12месяцев, последний год используется для независимой от обучения весов калибровки и настройки триггера. Метки должны фактически созреть до границы. Обучать веса и калибратор на одних и тех же ответах не стали.

Исторические метрики выше относятся к ежегодному recipe, а не к новому checkpoint после последнего refit: у него пока нет будущего теста. Последняя доступная feature date показывается отдельно от model as-of. Демо на дате из calibration помечается historical_smoke, не выдаётся за независимый тест или live-рекомендацию.

В final_solution/tabm_h3 находятся самодостаточные построение33признаков, официальный TabM inference, причинное правило и сохранение состояния. Исполняемый путь не импортирует research-модули. NOW_H3 и возможная отдельная CLOSING-голова имеют собственные горизонты/контракты; старые H5 вероятности не переименовываются в H3.

## Файлы

- by_year.csv / summary.csv — результаты обеих длин истории и всех правил.
- ranking.csv / selection.json — воспроизводимый выбор.
- paired_intervals.csv — сравнения при одинаковом правиле.
- long_history/ — три новых ежегодных TabM checkpoint и raw прогнозы.
- audit/ — источники, latest panel и независимые проверки.
- REPORTING_ALL_METRICS.csv — единый отчётный CSV, включая предыдущие сравнения.
'''
 if (HERE/'closing_by_year.csv').exists():
  text+='\n\n## Дополнительная CLOSING_H3\n\n'+table(pd.read_csv(HERE/'closing_by_year.csv'),['year','now_signals','closing_annotations','hit_rate','lift','endpoint_delta_bps'])
  text+='\n\nГолова обучена отдельно: R[t+3]>R[t]. Аннотация требует NOW, вероятность CLOSING не ниже 0,5 и положительный ret1. В 2025 lift 1,259 ниже 1,3, поэтому активная аннотация отключена; вероятность доступна в диагностике. Дополнительных уведомлений нет. Endpoint-событие тесно связано с NOW и не является независимым подтверждением: истинный NOW почти всегда влечёт рост endpoint при отсутствии равенств.\n'
 if (HERE/'deployment_receipt.json').exists():text+='\n\n### Фактический пакет\n\n```json\n'+(HERE/'deployment_receipt.json').read_text()+'\n```\n'
 (HERE/'REPORT.md').write_text(text)
 prior=ROOT/'research_v4/robust_selection/REPORTING_ALL_METRICS.csv';allrows=[pd.read_csv(prior,low_memory=False)];sources=[]
 for f,typ in [('by_year.csv','annual'),('summary.csv','aggregate'),('ranking.csv','selection'),('paired_intervals.csv','paired_interval'),('closing_by_year.csv','closing_annotation'),('closing_head_metrics.csv','closing_head')]:
  path=HERE/f
  if not path.exists():continue
  z=pd.read_csv(path);z['source_report']=str(path.relative_to(ROOT));z['experiment_protocol']='h3_finalization';z['row_type']=typ;z['train_horizon']=3;z['evaluation_horizon']=3;z['evaluation_scope']='KZT';allrows.append(z);sources.append(dict(path=str(path.relative_to(ROOT)),rows=len(z),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 union=pd.concat(allrows,ignore_index=True,sort=False);union.to_csv(HERE/'REPORTING_ALL_METRICS.csv',index=False)
 (HERE/'reporting_sources.json').write_text(json.dumps(dict(prior_union=str(prior.relative_to(ROOT)),prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),prior_rows=len(allrows[0]),added_sources=sources,total_rows=len(union)),indent=2))
 print('Report + CSV',len(union),'rows')
if __name__=='__main__':main()
