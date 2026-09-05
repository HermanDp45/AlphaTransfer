#!/usr/bin/env python3
"""Build the Russian factual readout after every prespecified run is complete."""
import json
from pathlib import Path
import pandas as pd
from run_heads import OUT,sha,save

NAMES={'baseline_reproduction':'V3, голова 2 года','basis_train_120m':'V3 long, 10 лет','base_control_head10y':'Базовый HGB, 10 лет','base_control_head2y':'Базовый HGB, 2 года','chronos2_small_ft_head10y':'Small, 300 шагов, 10 лет','chronos2_small_ft_head2y':'Small, 300 шагов, 2 года','chronos2_synth_ft_head10y':'Synth, 300 шагов, 10 лет','chronos2_synth_ft_head2y':'Synth, 300 шагов, 2 года','chronos2_synth_ft_kzt_head10y':'Synth + KZT, 10 лет','chronos2_synth_ft_kzt_head2y':'Synth + KZT, 2 года','chronos2_synth_zs_head10y':'Synth без дообучения, 10 лет','chronos2_synth_zs_head2y':'Synth без дообучения, 2 года','chronos2_small_ft900_head10y':'Small, 900 шагов, 10 лет','chronos2_small_ft_cpu300_head10y':'Small, 300 шагов, CPU'}
DEV='development_2023_2025';TEST='inspected_retrospective_2026'

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def get(df,cid,stage=DEV,scope='all'):
    return df[df.config_id.eq(cid)&df.stage.eq(stage)&df.scope.eq(scope)].iloc[0]
def interval(row,metric='delta_brier'):
    low=metric+'_ci95_low';high=metric+'_ci95_high'
    if metric!='delta_brier':low=metric+'_delta_ci95_low';high=metric+'_delta_ci95_high';metric+='_delta'
    return f"{row[metric]:+.6f} [{row[low]:+.6f}; {row[high]:+.6f}]"

def main():
    ext=pd.read_csv(OUT/'extended_contract/summary.csv');mixed=pd.read_csv(OUT/'summary.csv')
    paired=pd.read_csv(OUT/'paired_intervals.csv');ext_ci=pd.read_csv(OUT/'extended_contract/paired_intervals.csv')
    budget=pd.read_csv(OUT/'budget900/summary.csv');cpu=pd.read_csv(OUT/'budget900/cpu_matched_summary.csv');cpu_ci=pd.read_csv(OUT/'budget900/cpu_matched_paired_intervals.csv')
    forecasts=pd.read_csv(OUT/'budget900/forecast_summary.csv')
    march=pd.read_csv(OUT/'extended_contract/march/summary.csv')
    candidates=['baseline_reproduction','basis_train_120m','chronos2_small_ft_head10y','chronos2_synth_ft_head10y','chronos2_synth_zs_head10y','chronos2_small_ft900_head10y']
    rows=[]
    for cid in candidates:
        df=budget if 'ft900' in cid else ext;a,b=get(df,cid),get(df,cid,TEST)
        rows.append([NAMES[cid],f'{a.brier:.6f}',f'{a.lift:.3f}',f'{a.forward_delta_bps:.2f}',int(a.signals),f'{b.brier:.6f}',f'{b.forward_delta_bps:.2f}'])
    primary=table(['Модель','Brier 2023–25 ↓','Lift','Δ к случайному дню, б.п.','Сигналы','Brier 2026 ↓','Δ к случайному дню 2026, б.п.'],rows)
    rows=[]
    for cid in ['base_control_head10y','chronos2_small_ft_head10y','chronos2_synth_ft_head10y','chronos2_synth_zs_head10y']:
        ref=cid.replace('head10y','head2y');a=paired[paired.config_id.eq(cid)&paired.benchmark.eq(ref)&paired.stage.eq(DEV)&paired.scope.eq('all')].iloc[0]
        b=paired[paired.config_id.eq(cid)&paired.benchmark.eq(ref)&paired.stage.eq(TEST)&paired.scope.eq('all')].iloc[0]
        rows.append([NAMES[cid],interval(a),f'{b.delta_brier:+.6f}',f'{a.forward_delta_bps_delta:+.2f}'])
    windows=table(['Голова 10 против 2 лет','Δ Brier 2023–25 [95% ДИ]','Δ Brier 2026','Δ эффекта сигналов 2023–25, б.п.'],rows)
    kzt=[]
    for cid in ['basis_train_120m','chronos2_synth_ft_head10y','chronos2_synth_ft_kzt_head10y','chronos2_synth_zs_head10y']:
        a,b=get(ext,cid,scope='KZT'),get(ext,cid,TEST,'KZT')
        kzt.append([NAMES[cid],f'{a.brier:.6f}',f'{a.forward_delta_bps:.2f}',f'{b.brier:.6f}',f'{b.forward_delta_bps:.2f}'])
    kzt_table=table(['KZT','Brier 2023–25','Эффект сигналов, б.п.','Brier 2026','Эффект сигналов 2026, б.п.'],kzt)
    budget_rows=[]
    for stage,label in [(DEV,'2023–2025'),(TEST,'2026')]:
        row=cpu_ci[cpu_ci.stage.eq(stage)&cpu_ci.scope.eq('all')&cpu_ci.config_id.eq('chronos2_small_ft900_head10y')].iloc[0]
        a,b=get(cpu,'chronos2_small_ft_cpu300_head10y',stage),get(cpu,'chronos2_small_ft900_head10y',stage)
        budget_rows.append([label,f'{a.brier:.6f}',f'{b.brier:.6f}',interval(row),f'{row.forward_delta_bps_delta:+.2f}'])
    budget_table=table(['Период','300 шагов, CPU','900 шагов, CPU','Δ Brier [95% ДИ]','Δ эффекта сигналов, б.п.'],budget_rows)
    march_rows=[]
    for cid in ['base_control_head10y','chronos2_small_ft_head10y','chronos2_synth_ft_head10y','chronos2_synth_zs_head10y']:
        f=march[march.config_id.eq(cid)&march.scope.eq('all')].set_index('cutoff');a,b=f.loc['2026-01-01'],f.loc['2026-03-01']
        march_rows.append([NAMES[cid],f'{a.brier:.6f}',f'{b.brier:.6f}',f'{a.forward_delta_bps:.2f}',f'{b.forward_delta_bps:.2f}'])
    march_table=table(['Общие даты 03.03–25.08.2026','Brier, январь','Brier, март','Эффект, январь, б.п.','Эффект, март, б.п.'],march_rows)
    best=ext_ci[ext_ci.config_id.eq('chronos2_synth_zs_head10y')&ext_ci.benchmark.eq('basis_train_120m')&ext_ci.stage.eq(DEV)&ext_ci.scope.eq('all')].iloc[0]
    kci=ext_ci[ext_ci.config_id.eq('chronos2_synth_zs_head10y')&ext_ci.benchmark.eq('basis_train_120m')&ext_ci.stage.eq(DEV)&ext_ci.scope.eq('KZT')].iloc[0]
    forecast_rows=[]
    for cid in ['random_walk_gaussian60','chronos2_small_ft','chronos2_small_ft900']:
        a,b=get(forecasts,cid),get(forecasts,cid,TEST)
        forecast_rows.append([{'random_walk_gaussian60':'Случайное блуждание, волатильность 60 наблюдений','chronos2_small_ft':'Small, 300 шагов','chronos2_small_ft900':'Small, 900 шагов'}[cid],f'{a.pinball_bps:.3f}',f'{b.pinball_bps:.3f}',f'{b.coverage80:.1%}'])
    forecast_table=table(['Прогноз квантилей','Pinball 2023–25, б.п. ↓','Pinball 2026, б.п. ↓','Покрытие интервала 80% в 2026'],forecast_rows)
    fits=[json.loads(x.read_text()) for x in sorted((OUT/'budget900/checkpoints').glob('**/fit_receipt.json'))]
    runtime=sum(x['fit_seconds'] for x in fits)
    cpu_round=cpu_ci[cpu_ci.config_id.eq('chronos2_small_ft_cpu300_head10y')&cpu_ci.scope.eq('all')]
    rounding='; '.join(f"{'2023–2025' if r.stage==DEV else '2026'}: Δ Brier {r.delta_brier:+.8f}" for r in cpu_round.itertuples())
    text=f'''# Chronos: десятилетняя голова NOW, дата заморозки и бюджет дообучения

Проверено недостающее звено прежнего эксперимента: десять лет данных теперь использует и финальный классификатор NOW, а не только нейросеть. Выполнены 68 годовых обучений HGB, сохранены 15 мартовских моделей и четыре новых полных дообучения Small по 900 шагов. Все исходные V3/V4 данные, веса и прогнозы сохранены. Результат — проверяемое сравнение, включая отрицательные результаты, а не обещание прироста от более крупной модели.

**Постановка и границы доказательства.** Цель прежняя: текущая стоимость единицы валюты в рублях не выше минимума следующих пяти эффективных наблюдений ЦБ; равенство считается положительным исходом. Chronos выдаёт отдельные квантили по горизонтам, которые сами по себе не определяют вероятность этого события. HGB учится исходной метке по базовым признакам и 14 заранее заданным сводкам прогнозов: вероятности по отдельным горизонтам, медианы, ширины интервалов и границы Фреше. Мы не подменяли вероятность всей траектории одной маргинальной вероятностью.

В каждом годовом разбиении дообучение нейросети заканчивается до года калибровки. Для 2026 года это история 2015–2024 годов, калибровка — 2025 год. Для более ранних тестов границы сдвигаются ежегодно. Последние пять наблюдений каждой валюты удалены из обучения головы и калибровки; фактические даты созревания меток проверены. Калибровка Платта, пороги и состояние пауз между сигналами используют только прошлый год. У обеих голов одинаковые 120 итераций HGB, глубина 2 и остальные параметры; ранняя остановка отключена.

Ранние признаки для обучения головы построены фиксированной нейросетью, которая уже дообучалась на соответствующей ранней истории. Это признаки обучающей выборки, **не десять лет независимых вневыборочных прогнозов нейросети**. Калибровка и внешний тест следуют после дообучения backbone. Период 2023–2025 остаётся ретроспективной исследовательской выборкой. 2026 год уже просматривался в V3/V4; это отдельная ретроспективная проверка, но не нетронутый контрольный набор. [Протокол](protocol.json) сохранён до новых оценок.

**Сравнение с действующим V3 long.** Основная таблица использует в точности базовые признаки неизменённого `panel_extended.pkl`. Контроль без Chronos воспроизвёл V3 long на всех 4 415 строках: разница вероятностей и сигналов равна нулю. В 2023–2025 годах оцениваются 3 635 строк, в 2026 году — 780, одинаковые для всех моделей. Brier характеризует качество вероятностей; меньше — лучше. Эффект сигналов — разница с равномерным случайным днём внутри той же ячейки «год × валюта», с весами по фактическим сигналам. Это исследовательский показатель на курсах ЦБ, не исполнимая экономия клиента с учётом банковского спреда.

{primary}

Для Synth без дообучения разница Brier с V3 long на 2023–2025 годах составляет {interval(best)}. Разница эффекта сигналов — {best.forward_delta_bps_delta:+.2f} б.п., 95% ДИ [{best.forward_delta_bps_delta_ci95_low:+.2f}; {best.forward_delta_bps_delta_ci95_high:+.2f}]. Это практически равное качество, без подтверждённого преимущества. Все годовые и валютные ячейки доступны в [полной таблице](extended_contract/cells.csv), неопределённость — в [парных интервалах](extended_contract/paired_intervals.csv).

**Чистый эффект длины головы.** Дополнительно сохранён общий набор признаков, который в точности воспроизводит прежний прогрев признаков V4 с 2020 года, а до 2020 года дополняется историей. Каждый двухлетний контроль воспроизвёл прежние вероятности и сигналы точно. Поэтому следующая таблица изолирует именно переход 2 → 10 лет при одинаковых признаках и весах нейросети. Её десятилетний базовый HGB отличается от полного V3 long из-за прогрева признаков 2020 года: Brier 0.186246 против 0.184850. Эти две схемы подготовки данных нельзя смешивать.

{windows}

Увеличение истории улучшило исследовательские точечные оценки Brier, но все соответствующие 95% интервалы пересекают ноль. Улучшение вероятностей также не гарантирует улучшение моментов отправки сигналов: у Synth без дообучения средний эффект сигналов снизился примерно на 10 б.п. [Все результаты сопоставимых окон](summary.csv).

**Казахстан.** Дополнительное дообучение Synth на прошлой истории KZT сравнивается с объединённым дообучением по тем же датам. Оно не выбиралось по тесту. На точной панели V3 получено:

{kzt_table}

У Synth без дообучения исследовательское улучшение KZT Brier относительно V3 long равно {interval(kci)}. Интервал пересекает ноль; на 2026 год улучшение не переносится. Положительный результат отдельной валюты не является подтверждением общего выигрыша.

**900 против 300 шагов.** Small выбран по измеренной стоимости вычислений до чтения новых итоговых метрик: 20 шагов занимали 2.934 секунды CPU; Synth на MPS — 11.09 секунды, что означало бы около 33 минут на четыре обучения по 900 шагов. Реально выполнено 3 600 новых полных шагов Small; обучение четырёх моделей заняло {runtime:.1f} секунды. Начальные веса, десятилетняя история, контекст 256, скорость обучения и seed совпадают с прежней постановкой. Линейное снижение скорости обучения растянуто на 900 шагов: это сравнение бюджета вместе с длиной расписания, а не продолжение уже завершённого 300-шагового checkpoint. Вывод о сходимости не делается. [Отдельный протокол](budget900/protocol.json).

Оба варианта в следующей таблице полностью вычислены на CPU, включая дополнительный контроль неизменённых 300-шаговых весов. Это исключает смешение эффекта обучения с округлениями MPS.

{budget_table}

Дополнительные 600 шагов ухудшили Brier в обоих периодах. На данных 2023–2025 годов 95% интервал разницы целиком выше нуля; в 2026 году особенно заметно ухудшение эффекта сигналов. Этот вариант не даёт оснований заменять 300-шаговую модель или V3 long.

Сравнение с исходным 300-шаговым вариантом также сохранено в [результатах бюджета](budget900/summary.csv). Влияние повторного CPU-вычисления само по себе: {rounding}. Проверена и исходная задача прогнозирования квантилей, независимо от головы NOW:

{forecast_table}

**Январская и мартовская заморозки.** В обоих случаях используются одни веса нейросети, обученные до конца 2024 года. В мартовском варианте десятилетнее обучение головы заканчивается перед мартом 2025 года, калибровка использует март 2025 — февраль 2026 года с удалением незрелых меток. История пауз воспроизведена именно для этих границ; полное восстановление январского варианта проверено по вероятностям и сигналам. Ни одна мартовская целевая метка не использована при обучении или калибровке.

Общие даты — **3 марта — 25 августа 2026 года, 122 наблюдения на валюту**. Это неполные шесть календарных месяцев. Ни одна модель не выбиралась по этой таблице.

{march_table}

Все сравнения, включая KZT, и 10 000 парных пересчётов приведены в [мартовском срезе](extended_contract/march/paired_intervals.csv). Более свежая калибровка меняет и вероятности, и набор сигналов; это диагностический результат, а не гарантия устойчивости будущего периода.

**Проверяемость и ограничения.** MPS реально доступен вне sandbox. CPU/MPS отличаются максимум на 4.77·10⁻⁷ в логарифме курса, или 0.0048 б.п. Дополнительно вычислены 25 659 прогнозных групп «модель × год × дата», по пять валют в каждой, за 179.8 секунды. Прежние квантили переиспользованы побитово. `cross_learning=False` проверен реальным сравнением одиночной и смешанной группы: более поздние контексты не влияют на текущую группу сверх численной погрешности. Изменение всех будущих входных значений дало ровно нулевое изменение текущего прогноза. Сохранённые головы восстановлены из файлов; метки, границы дат, хэши источников и паузы проверены независимо. [Итоговый машинный статус](final_verification.json).

95% интервалы используют 10 000 парных выборок целых календарных месяцев, раздельно внутри каждого года; все валюты одной даты пересэмплируются вместе. Средние случайного дня и число сигналов пересчитываются внутри каждой выборки. Это условные интервалы для уже обученных моделей; они не учитывают весь поиск гипотез и неопределённость повторного обучения. Поэтому небольшое преимущество отдельной ячейки не объявляется доказанным приростом.

[Chronos-2](https://arxiv.org/html/2510.15821v1), [Small](https://huggingface.co/autogluon/chronos-2-small) и [Synth](https://huggingface.co/autogluon/chronos-2-synth) проверялись по первичным источникам. Для выбранных весов указана Apache-2.0; у Small в предобучении были реальные и синтетические ряды, у Synth издатель заявляет только синтетические. Веса опубликованы в конце 2025 года, поэтому тесты 2023–2025 не имитируют доступность моделей в те годы. Полностью исключить пересечение реального корпуса Small с историческими финансовыми данными невозможно; эта неопределённость сохранена в [аудите источников](../../foundation/SOURCES.md).

Код, предсказания, веса и квитанции перечислены в [README](README.md) и [манифесте](MANIFEST.json). Числа из этого отчёта не заменяют основной продуктовый эксперимент на реальных банковских котировках и поведении клиентов.
'''
    (OUT/'REPORT.md').write_text(text)
    print('REPORT words',len(text.split()))

if __name__=='__main__':main()
