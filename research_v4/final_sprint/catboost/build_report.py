"""Generate CatBoost findings from saved metrics, without fitting or selection."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    local=pd.read_csv(HERE/'candidates_summary.csv')
    unified=pd.read_csv(HERE.parent/'leaderboard.csv')
    name='catboost_treasury_halyk_24m_stable';policy='cadence85_cd2'
    selected=unified[unified.config_id.eq(name)&unified.policy.eq(policy)]
    predictions=pd.read_csv(HERE.parent/'policy_predictions.csv.gz',parse_dates=['date'],low_memory=False)
    predictions=predictions[predictions.config_id.eq(name)&predictions.policy.eq(policy)]
    intervals=[]
    for (year,mode),g in predictions.groupby(['fold_test_year','mode']):
        start=g.date.min().to_period('W-SUN').ordinal
        blocks=np.array([(d.to_period('W-SUN').ordinal-start)//4 for d in g.date])
        target=g.target.to_numpy(float);forward=g.forward_bps.to_numpy(float);signal=g.candidate_signal.to_numpy(bool)
        stats=np.array([[sum(blocks==b),target[blocks==b].sum(),sum((blocks==b)&signal),target[(blocks==b)&signal].sum(),forward[blocks==b].sum(),forward[(blocks==b)&signal].sum()] for b in sorted(set(blocks))])
        rng=np.random.default_rng(20260905);draw=stats[rng.integers(len(stats),size=(10000,len(stats)))].sum(axis=1)
        intervals.append(dict(config_id=name,policy=policy,year=int(year),mode=mode,blocks=len(stats),
              lift_ci95=np.quantile((draw[:,3]/draw[:,2])/(draw[:,1]/draw[:,0]),[.025,.975]).tolist(),
              forward_delta_bps_ci95=np.quantile(draw[:,5]/draw[:,2]-draw[:,4]/draw[:,0],[.025,.975]).tolist()))
    (HERE/'catboost24m_policy_intervals.json').write_text(json.dumps(dict(method='Independent fixed-prediction four-calendar-week cluster bootstrap;10000replicates;not selection-adjusted.',results=intervals),indent=2)+'\n')
    table=[]
    for _,r in selected.sort_values(['cutoff','mode']).iterrows():
        table.append(f"| {r.cutoff[:4]} | {r['mode']} | {r.brier:.6f} | {r.lift:.4f} | {r.weeks_1_2:.2%} | {int(r.signals)} | {r.forward_delta_bps:+.2f} |")
    matched=[]
    for months in (24,60,120):
        q=local[local.cutoff.eq('2026-01-01')&local['mode'].eq('normal')].set_index('config_id')
        base=q.loc[f'catboost_treasury_halyk_{months}m_full'];subset=q.loc[f'catboost_treasury_halyk_{months}m_stable_matched']
        matched.append(f"| {months} | {base.brier:.6f} | {subset.brier:.6f} | {subset.brier-base.brier:+.6f} |")
    sources=json.loads((HERE/'protocol.json').read_text())
    text=f'''# CatBoost и независимый аудит финального блока

Выполнено **18 обучений CatBoost**: три окна train — 24, 60 и 120 месяцев, два временных cutoff — январь 2025 и январь 2026. Первые 12 обучений сравнивают полный набор и регуляризованный вариант с отбором; ещё 6 контролей сохраняют все гиперпараметры полного варианта и меняют только признаки. Дополнительно выполнены 2 обучения HGB для проверки эффекта augmentation. Прежние результаты и прежний код `final_solution` не перезаписывались. Выбранный в общем исследовании исполняемый профиль упакован отдельно в `final_solution/final_sprint`; итоговый normal-only выбор описан в корневом отчёте.

CatBoost 1.2.10 обучен на пяти коридорах с фиксированным весом KZT=4, остальных=1; калибровка и уведомления оцениваются на KZT. Использованы 15 прежних признаков CBR/CNY, шесть Treasury с лагом 7 дней и шесть Halyk. OXR в эти модели не входит. До появления Treasury/Halyk история остаётся пропущенной, а не заполняется будущими значениями. Параметры: 500 деревьев, learning rate 0,04, depth 2–4, L2 20–80, подвыборка строк и признаков 0,8, один CPU-поток; ранней остановки по тесту нет.

Из 27 числовых признаков отбор оставляет 14: пять заранее заданных опорных признаков плюс девять по устойчивости ранга абсолютной корреляции Spearman в четырёх последовательных частях **только KZT train**. Это простой фильтр, а не доказательство причинного влияния признаков. Предыдущие 12 месяцев используются для Platt-калибровки и порогов; label maturity проверена перед каждой границей. Последние даты истории сохранены для состояния уведомлений, незрелые target/forward скрыты.

## Результат при общей политике уведомлений

`{name}` с `{policy}` использует depth=3, L2=50 и 14 выбранных признаков. Порог и cooldown=2 определены по предыдущему периоду калибровки. Пользователь разрешил ретроспективный выбор модели/политики на 2026 году; это **не нетронутая проверочная выборка**. Название cadence85 задаёт требование к калибровке, а не принудительную квоту будущей недели.

| Год | Доступность банка | Brier ↓ | Lift ↑ | Недель с 1–2 сигналами | Сигналов | Преимущество, б.п. |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

В 2026 году имеются 156 допустимых дат и 33 календарные недели, включая неполные граничные и недели без сигналов. В обеих версиях доступности нужную частоту имеют 31/33 недели. При обычной доступности среди 40 сигналов 21 успешный против 57 успешных дат из 156 во всём периоде: lift=1,43684. При дополнительной задержке Halyk на сутки lift=1,43358. Таким образом, этот ретроспективный кандидат выполняет три заданные точечные цели и на normal, и на bank_delayed. После уточнения пользователя основным критерием общего выбора является normal, а дополнительная задержка остаётся диагностикой; таблица здесь описывает конкретный CatBoost, а не объявляет его победителем среди всех семейств.

С прежней legacy-политикой тот же CatBoost даёт lift=1,16108 и 1,07815. Поэтому достижение цели относится к **совместной модели и политике**, а не к одному переобучению. Для Treasury+Halyk HGB с той же cadence85 политикой normal lift=1,17293; его преимущество составляет +37,81 б.п. против +42,46 б.п. CatBoost. Brier CatBoost, напротив, хуже: 0,234634 против 0,220019. Прогресс в целевой политике не означает улучшения всех вероятностных метрик.

[Четырёхнедельные интервалы](catboost24m_policy_intervals.json) по фиксированным прогнозам дают для normal lift примерно [1,143; 2,003], для задержки [1,124; 1,955]. Нижняя граница не достигает 1,3. Эти интервалы не повторяют весь ретроспективный отбор моделей и политик. Положительная точечная цель и её условный интервал не являются гарантией будущего результата.

## Что дал отбор признаков

В чистых контролях с одинаковыми гиперпараметрами отбор ухудшил Brier на 2026 normal во всех трёх окнах:

| Окно train, месяцев | Полный набор | Только отобранные | Δ Brier |
|---|---:|---:|---:|
{chr(10).join(matched)}

Уведомления также меняются немонотонно. В частности, matched24m достигает lift=1,5053 в задержанном режиме при cadence85, но только 1,1632 в обычном. Нельзя объявлять сам фильтр универсальным улучшением: удачный вариант включает конкретную регуляризацию и политику. При переносе cutoff с 2025 на 2026 совпадают 10/14, 10/14 и 11/14 выбранных признаков для окон 24/60/120 месяцев; Jaccard для девяти свободно выбираемых признаков составляет лишь 0,385/0,385/0,500. Внутреннее ранжирование по четырём train-блокам не обеспечивает неизменность набора во времени. Все промежуточные конфигурации сохранены в [таблице устойчивости](selection_stability_across_cutoffs.csv).

## Контроль удвоения train в HGB

Для HGB60m/c12 сравнили normal+banklag2 augmentation с **normal+normal**: одинаковое число обучающих строк, одинаковые параметры, реальные неизменённые normal/lag2 данные на тесте. Контроль исключён из выбора кандидата.

С cadence90/cooldown3 в 2026 normal lift у duplicate-normal равен 1,20743, у augmentation — 1,44892; в задержке — 1,12693 и 1,36842. Простой дубликат не воспроизводит прирост lift. Однако преимущество duplicate-normal выше: +46,99/+34,74 б.п. против +31,74/+30,59. Месячные парные интервалы Δlift включают ноль; normal ΔBrier augmentation хуже на +0,007338. Это подтверждает различие механизмов, но не доказывает превосходство augmentation по всем метрикам.

## Воспроизводимость и границы

[78 проверок](verification.json) прошли без ошибок: все 18 CatBoost checkpoint/calibrator/threshold и normal/lag2 прогнозы воспроизведены, проверены label maturity 118 ежегодных/ежемесячных HGB checkpoints, причинность политики и фактический перенос состояния во всех шести ежемесячных потоках. [Отдельный аудит общего evaluator](root_evaluation_verification.json) независимо пересчитывает таблицу метрик, все 72 CatBoost policy registry записи и четырёхнедельные интервалы. Его receipt фиксирует проверенную версию общей таблицы. Ещё [5 проверок augmentation](augmentation_controls/verification.json) и [12 проверок исполняемого профиля](runtime_audit/verification.json) подтвердили реальные delayed features, воспроизведение контактов, защиту курсора и отсутствие влияния исходов на inference. [Итог независимого аудита root-кода](ROOT_AUDIT.md) описывает найденные и исправленные проблемы.

Показатель «преимущество» равен среднему `forward_bps` на сигналах минус среднее на всех допустимых датах того же периода. Это результат на reference-rate CBR NOW h5, не чистая прибыль и не исполнимая банковская котировка. Halyk BANK SELL используется как внешний индикатор; для пользовательского RUB→KZT перевода нужна другая сторона котировки и фактические комиссии.

Основные файлы: [все кандидаты](candidates_predictions.csv.gz), [калибровка](candidates_calibration_predictions.csv.gz), [история состояния](candidates_history_predictions.csv.gz), [метрики](candidates_summary.csv), [чистые контроли отбора](matched/paired_feature_selection.csv), [augmentation-контроль](augmentation_controls/augmentation_vs_duplicate.csv), [исходный протокол](protocol.json), [код](experiment.py). Запуск: `experiment.py`, затем `matched_controls.py`; `augmentation_control.py` выполняет только два отдельных HGB-контроля. `verify.py` и `audit_evaluation.py` не обучают деревья.

Первый этап оценки остановился после всех 12 fits на сравнении одинаковых дат pandas разных разрешений ns/us. Сбой устранён приведением типа дат и повторной оценкой сохранённых CSV; модели и прогнозы не менялись. Это отражено в [completion receipt](completion.json) и исходном `fit_engine_snapshot.py`.
'''
    (HERE/'REPORT.md').write_text(text)

if __name__=='__main__':main()
