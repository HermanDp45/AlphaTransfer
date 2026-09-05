"""Build the continuation readout from measured CSVs; preserve prior V4 reports."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent

def table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(int(v)) if c in ('signals','rows','dates','blocks') and pd.notna(v)
                                      else f'{v:.6f}' if isinstance(v,float) else str(v)
                                      for c,v in zip(columns,row))+' |')
    return '\n'.join(lines)

def main():
    oxr=pd.read_csv(HERE/'oxr/summary.csv')
    foundation=pd.read_csv(HERE/'foundation/extended_contract/summary.csv')
    cpu_budget_path=HERE/'foundation/budget900/cpu_matched_summary.csv'
    cpu_budget=cpu_budget_path.exists()
    budget=pd.read_csv(cpu_budget_path if cpu_budget else HERE/'foundation/budget900/summary.csv')
    robust=pd.read_csv(HERE/'robustness/results/development_summary.csv')
    age=pd.read_csv(HERE/'robustness/results/age_development_summary.csv')
    # The primary foundation table uses the exact same base features as V3 long.
    f=foundation[foundation.scope.eq('all')&foundation.config_id.isin(['basis_train_120m','chronos2_small_ft_head10y','chronos2_synth_ft_head10y','chronos2_synth_zs_head10y'])]
    f=f.pivot(index='config_id',columns='stage',values='brier').reset_index()
    f.columns=['Модель','Brier2023–2025','Brier2026']
    selected=['v3_long_globalcal','v3_long_localcal','halyk_l1','minimax_shrink_v3','halyk_l2_retrained']
    r=robust[robust.rule.isin(selected)].pivot(index='rule',columns='delay',values='brier').loc[selected].reset_index()
    r.columns=['Модель','Brier: штатный вход','Brier: вход задержан ещё на день']
    o=oxr[oxr.scope.eq('all')&oxr.config_id.isin(['v3_120m','oxr_basis_120m_delay24h'])&oxr.track.isin(['development_2023_2025','test2026_january_freeze'])]
    short_id='chronos2_small_ft_cpu300_head10y' if cpu_budget else 'chronos2_small_ft_head10y'
    b=budget[budget.scope.eq('all') & budget.config_id.isin(['chronos2_small_ft900_head10y',short_id])]
    def budget_brier(model,stage):
        return float(b.loc[b.config_id.eq(model)&b.stage.eq(stage),'brier'].iloc[0])
    old_dev=budget_brier(short_id,'development_2023_2025')
    old_test=budget_brier(short_id,'inspected_retrospective_2026')
    new_dev=budget_brier('chronos2_small_ft900_head10y','development_2023_2025')
    new_test=budget_brier('chronos2_small_ft900_head10y','inspected_retrospective_2026')
    budget_note=('В таблице обе версии используют прогнозы CPU, одинаковый контракт признаков и те же даты; '
                 'это сравнение бюджета при фиксированном устройстве расчёта.' if cpu_budget else
                 'Эта промежуточная таблица диагностическая: 900 шагов используют прогнозы CPU, '
                 'а часть ранних прогнозов 300 шагов рассчитана на MPS. Эффект округления здесь ещё не отделён от бюджета.')
    budget_intervals=''
    if cpu_budget:
        budget_ci=pd.read_csv(HERE/'foundation/budget900/cpu_matched_paired_intervals.csv')
        budget_ci=budget_ci[budget_ci.scope.eq('all')&budget_ci.config_id.eq('chronos2_small_ft900_head10y')&budget_ci.benchmark.eq(short_id)].set_index('stage')
        di=budget_ci.loc['development_2023_2025'];ti=budget_ci.loc['inspected_retrospective_2026']
        budget_intervals=(f" Парные интервалы разницы 900−300: {di.delta_brier:+.6f} "
                          f"[{di.delta_brier_ci95_low:+.6f}; {di.delta_brier_ci95_high:+.6f}] на разработке и "
                          f"{ti.delta_brier:+.6f} [{ti.delta_brier_ci95_low:+.6f}; {ti.delta_brier_ci95_high:+.6f}] в 2026.")
    # Keep experiment branches, source contracts and test views explicit in ledger.
    ledgers=[]
    sources=[('oxr','primary_and_depth_controls','oxr/summary.csv'),('foundation','V2_warmup_preserved','foundation/summary.csv'),('foundation','exact_V3_extended','foundation/extended_contract/summary.csv'),('foundation','budget_control_exact_extended','foundation/budget900/summary.csv'),('foundation','March_V2_warmup_preserved','foundation/march/summary.csv'),('foundation','March_exact_V3_extended','foundation/extended_contract/march/summary.csv')]
    if cpu_budget:
        sources.append(('foundation','budget_CPU_matched_exact_extended','foundation/budget900/cpu_matched_summary.csv'))
    for branch,contract,path in sources:
        q=pd.read_csv(HERE/path).rename(columns={'stage':'track'})
        if contract=='budget_CPU_matched_exact_extended':
            q=q[q.config_id.isin(['chronos2_small_ft900_head10y','chronos2_small_ft_cpu300_head10y'])]
        q['branch']=branch;q['source_contract']=contract;q['source_table']=path
        if 'track' not in q:q['track']='common_march_onward'
        ledgers.append(q)
    for path,track in [('development_summary.csv','development_2023_2025'),('age_development_summary.csv','development_2023_2025'),('metrics_by_cutoff.csv','annual_or_march_by_cutoff'),('age_metrics_by_cutoff.csv','annual_or_march_by_cutoff'),('common_march5_metrics.csv','common_march5_onward'),('age_common_march5_metrics.csv','common_march5_onward')]:
        q=pd.read_csv(HERE/'robustness/results'/path).rename(columns={'rule':'config_id','candidate_count':'signals','candidate_lift_standardized':'lift'})
        q['branch']='robustness';q['source_contract']='exact_V3_extended_KZT';q['scope']='KZT';q['track']=track;q['source_table']='robustness/results/'+path;ledgers.append(q)
    ledger=pd.concat(ledgers,ignore_index=True);ledger.to_csv(HERE/'COMPARISON.csv',index=False)
    text=f'''# V4: OXR, длинные NOW-модели и задержки источников

**Дополнение от 5 сентября 2026 года. Нового подтверждённого победителя над V3 не получено.** Выполнены все четыре направления: OXR на доступной истории, десять лет обучения головы NOW, устойчивость к задержке и отдельные временные тесты 2026. Дополнительно в три раза увеличен бюджет дообучения Chronos Small. Главное уточнение прежнего вывода: при смоделированной задержке реальных данных неизменённая Halyk-модель показывает меньший Brier, чем отдельно переобученная lag2-модель.

Brier оценивает качество вероятностей: меньше — лучше. Метка NOW означает, что текущая рублёвая цена валюты не выше каждой из следующих пяти котировок ЦБ. `forward_delta_bps` сравнивает выбранные даты со случайным днём того же года и коридора по будущему среднему курсу. Это показатели на справочных курсах, без банковских комиссий и исполнимого спреда.

## OXR: небольшой эффект на разработке не переносится на 2026

Использованы реальные данные OXR2018-06-17–2026-09-02. Основной набор — расхождение OXR с ЦБ и его динамика; доходности, полный набор, только наличие/возраст источника и более длинный лаг проверены отдельно. Всего 95 обученных HGB с checkpoints и фиксированными отсечками. Первичный кандидат выбран на 2023–2025 до нового расчёта 2026. Дневной снимок допускается в пакет 10:05 MSK только с D+2, стресс — D+3: доступность равна max(публикация, следующая UTC-полночь)+24/48 часов. Все сравнения OXR ниже объединяют пять коридоров.

{table(o,['track','config_id','brier','signals','forward_delta_bps'])}

На разработке относительное улучшение Brier — 0.27%, но 95%-ный парный месячный интервал разницы [−0.001644; +0.000558] включает ноль. В January 2026 выбранный вариант уже немного хуже V3 и по Brier, и по преимуществу сигналов. Признаки полного набора не дали общего boost.

Чтобы проверить ценность старой истории, OXR искусственно обрезали до 2020/2022 при неизменном 10-летнем train официального курса. Добавление 2018–2019 к basis-модели улучшило development Brier лишь на 0.000645, интервал [−0.001484; +0.000149]. В 2026 история с 2022 оказалась лучше полной с 2018. Поэтому **нет измеренных оснований обещать улучшение от расширения OXR до 2010**. Это не доказательство бесполезности отсутствующих данных: экстраполировать процент прироста на другой временной режим нельзя. Контроли глубины basis добавлены после первого просмотра новых 2026 результатов и обозначены дополнительной чувствительностью.

![Глубина OXR и неопределённость эффекта](oxr/history_depth.png)

[Полный OXR-отчёт](oxr/REPORT.md) · [124 проверки независимого аудита](oxr/audit/REPORT.md).

## NOW поверх нейросети: десять лет вместо двух

Досчитаны ранние прогнозы Chronos на исторических окнах. Обучены головы NOW на 2 и 10 годах для Small FT, Synth FT, Synth KZT FT и Synth ZS. Проверены два контракта базовых признаков: сохранение исходного прогрева V4 для точного двухлетнего контроля и **точный panel_extended V3** для сравнения с длинной V3. Это существенно: простой перенос окна при сохранённом прогреве даёт другой baseline. В основной таблице ниже используется только второй контракт, с нулевой разницей прогнозов контрольной модели и V3.

{table(f,list(f.columns))}

Длинные головы часто улучшают свои короткие версии на разработке, но это не даёт устойчивого выигрыша над длинной V3. Например, в согласованном контроле с прежним прогревом Small с 2 до 10 лет улучшает Brier 0.187990→0.186245 на 2023–2025, а на 2026 ухудшает 0.209438→0.216279. Synth ZS с точной базой V3 почти повторяет её; варианты с дообученным backbone устойчивого превосходства не показали.

Отдельный контроль бюджета: Small 300 против 900 шагов, одинаковые 10-летняя история, context 256, 10-летняя голова и предыдущий год калибровки. Выполнены четыре новых дообучения по 900 шагов; изменение весов подтверждено, checkpoints сохранены. Более длинный linear schedule является частью изменения бюджета; это не продолжение старого 300-step checkpoint и не доказательство сходимости. {budget_note}

{table(b,['stage','config_id','brier','signals','forward_delta_bps'])}

В этом сравнении Brier Small на разработке изменился {old_dev:.6f}→{new_dev:.6f}, на 2026 — {old_test:.6f}→{new_test:.6f}. Больше шагов с этим schedule не улучшило результат; отрицательный результат сохранён вместе с весами.{budget_intervals}

MPS доступен вне песочницы. На 32 задачах Synth дал 0.25 с против 0.68 с CPU, примерно 2.7× быстрее; Small — 0.40 с против 0.20 с CPU. Максимальное различие квантилей CPU/MPS — 0.0048 bps. Поэтому GPU использован для расчёта ранних прогнозов, а дополнительное обучение Small — на CPU. Дополнительный 300-step CPU-контроль отделяет изменение бюджета от чувствительности HGB к округлениям CPU/MPS: поэтому его Brier 0.186198/0.216394 отличается от 0.186555/0.216981 в первой таблице. Ускорение измерено на указанном пакете задач и не переносится автоматически на всё обучение.

Ранние признаки для головы построены нейросетью, дообученной на том же разрешённом train-периоде. Это обученные внутри train признаки, **не десять лет независимых cross-fitted прогнозов**. При нашем дообучении backbone калибровочные и тестовые метки не использовались. Сохраняются прежние ограничения предобученных корпусов и исторической доступности весов 2025 года. [Подробный отчёт и проверки](foundation/REPORT.md).

## Устойчивость Halyk: исправление интерпретации и новые варианты

Прежние 0.186250 — результат **отдельного обучения на lag2**, а не деградация фиксированного lag1-кандидата при сбое источника. Теперь сравнение выполнено непосредственно: обучение, калибратор и пороги сохраняются; после отсечки источник приходит ещё на день позже. История контактов до отсечки остаётся штатной. **Вся следующая таблица — только KZT, 727 дат 2023–2025.** Её baseline отличается от объединённого результата пяти коридоров выше именно из-за этой области оценки.

{table(r,list(r.columns))}

В строке отдельно обученного lag2 контрольный вход уже имеет lag2, а задержанный — lag3; его нельзя смешивать с первыми строками, где проверяется lag1→lag2.

Исходный Halyk-кандидат ухудшается 0.176324→0.181404 и по среднему Brier остаётся лучше длинной V3: 0.182705. Новый minimax shrink уменьшает максимальный Brier двух сценариев до 0.180753, но ухудшает штатное качество до 0.178228. Простая KZT-калибровка V3 даёт 0.180435 в обоих сценариях — меньшую точечную оценку того же критерия. Среди проверенных вариантов **ни один не улучшил оба сценария относительно исходного Halyk**. Все интервалы ключевых сравнений с блоками по 20 наблюдений ЦБ включают ноль: например, minimax минус Halyk при задержке — −0.000652 [−0.008412; +0.007977]. Сценарные средние не доказывают превосходство по устойчивости.

Проверены смешивание лагов при обучении, удаление банковских признаков на части train, ансамбль и ограниченное смешивание с V3. Ещё две политики переключались по наблюдаемому возрасту котировок: они не используют скрытый флаг сценария. Обе оказались хуже исходного Halyk-кандидата на разработке. Полный набор отрицательных результатов и парные интервалы сохранены. [Отчёт по задержке](robustness/REPORT.md).

## 2026: две временные отсечки

**January freeze:** до 2025 обучается основная модель;2025 используется для калибровки и настройки частоты с исключением незрелых h5-меток. Прогнозы 2026 не используют его labels при обучении. **March freeze:** отсечка 1 марта; предыдущие 12 месяцев используются для калибровки с тем же контролем зрелости, для показанных длинных моделей предшествующие 10 лет — для fit. Допустимы только метки, пятый будущий курс которых уже наблюдался до отсечки. Веса и пороги внутри каждого тестового периода не меняются.

Признаки могут обновляться по мере поступления прошлых курсов — это обычный последовательный прогноз, а не обучение на тестовой метке. Использованы одинаковые даты внутри каждой пары. Полный январский тест: **13 января–25 августа,156 дат**. Общий мартовский тест OXR/Chronos: **3 марта–25 августа,122 даты**; дополнительный более строгий общий срез Halyk — **5 марта–25 августа,120 дат**. Эти знаменатели явно разделены в таблицах.

Такой тест отвечает на вопрос «как замороженная до отсечки модель работала на следующих месяцах»: в новые fit и калибровку не входят метки после отсечки. Но 2026 уже просматривался при выборе прежних идей и моделей, поэтому не является новым независимым подтверждением превосходства. Сохраняются ограничения исторических пересмотров источников и предобучения Chronos. Месячные интервалы 2026 имеют лишь 6–8 блоков и условны на уже выбранных моделях; весь исторический перебор ими не учтён.

## Что меняется в решении

OXR и новые нейросетевые варианты не получили оснований для замены V3. Казахстанский Halyk-кандидат остаётся исследовательским кандидатом по качеству, а модель V3 с KZT-калибровкой — простым ориентиром устойчивости без банковского источника. Превосходство по всем метрикам или будущая экономия клиентов не объявляются.

Изменения размещены в этом дополнении V4. **Ни один вариант этого продолжения не перенесён в `final_solution`.** Оригинальные веса, прогнозы и код предыдущих V3/V4 сохранены; старые версии двух навигационных отчётов — в `prior_reports`. В [COMPARISON.csv](COMPARISON.csv) **{len(ledger)} строк** сопоставлений с явными веткой, контрактом признаков, отсечкой и сценарием задержки. Это строки результатов, не число независимых экспериментов.

[Воспроизведение](README.md) · [Проверка целостности и научных проверок](verification.json) · [Протокол](PROTOCOL.md).
'''
    (HERE/'REPORT.md').write_text(text)
    (HERE/'README.md').write_text('''# Продолжение V4

[Итоговый отчёт](REPORT.md) и [таблица сравнений](COMPARISON.csv).

- [OXR](oxr/REPORT.md): 95 обучений, временные отсечки и глубина источника; [код](oxr/experiment.py), [независимый аудит](oxr/audit/REPORT.md).
- [Chronos и NOW](foundation/REPORT.md): длинные головы, точная база V3, MPS/CPU, контроль300/900 шагов. Инструкции запуска внутри отчёта.
- [Задержка Halyk](robustness/REPORT.md): неизменённая модель при задержке, регуляризация, возраст источника и два frozen cutoffs.

Из каталога AlphaTransfer, окружениеPython3.11 с requirements-market.txt и foundation/requirements.txt исходногоV4:

```sh
python research_v4/continuation/oxr/experiment.py --phase development
python research_v4/continuation/oxr/experiment.py --phase test
python research_v4/continuation/oxr/history_sensitivity.py
OPENBLAS_NUM_THREADS=1 python research_v4/continuation/oxr/assess.py
python research_v4/continuation/oxr/build_report.py
python research_v4/continuation/build_report.py
python3 research_v4/continuation/verify.py
python3 research_v4/verify_package.py --seal
```

Модельные проверки в каждой ветке восстанавливают прогнозы и проверяют причинность; verify.py проверяет их результаты, исходныйV3 и предыдущийV4. После научных перезапусков пакет нужно запечатать заново, поскольку outputs/receipts изменяются. Сохранять старый протокол выбора как будто он создан до новых данных нельзя: новый запуск должен иметь собственный протокол/каталог.

`previous_v4_manifest.json` — точная копия старой печатиV4. `prior_reports` сохраняет исходные версии REPORT/README. Навигация родительскогоV4 обновляется отдельным шагом после завершения всех проверок; исходные модели и научные результаты не меняются.
''')
    print('Continuation report and ledger',len(ledger),flush=True)

if __name__=='__main__':main()
