#!/usr/bin/env python3
"""Render the final report directly from persisted numerical artifacts."""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
s=pd.read_csv(OUT/'summary.csv')
f=pd.read_csv(OUT/'forecast_summary.csv')
ci=pd.read_csv(OUT/'paired_intervals.csv')

LABELS={'baseline_reproduction':'V3 HGB + basis, 2 года',
 'basis_train_120m':'V3 HGB + basis, 10 лет',
 'historical_path':'Прошлые эмпирические траектории',
 'random_walk_gaussian60':'Gaussian random walk, vol60',
 'chronos2_small_zs':'Small ZS → NOW adapter',
 'chronos2_small_ft':'Small FT → NOW adapter',
 'chronos2_small_zs_hgb_aug':'HGB + Small ZS признаки',
 'chronos2_small_ft_hgb_aug':'HGB + Small FT признаки',
 'chronos2_synth_zs':'Synth ZS → NOW adapter',
 'chronos2_synth_ft':'Synth FT → NOW adapter',
 'chronos2_synth_zs_hgb_aug':'HGB + Synth ZS признаки',
 'chronos2_synth_ft_hgb_aug':'HGB + Synth FT признаки',
 'chronos2_synth_ft_kzt':'Synth KZT FT → NOW adapter',
 'chronos2_synth_ft_kzt_hgb_aug':'HGB + Synth KZT FT признаки'}

def table(frame, columns):
    names=[c[1] for c in columns]
    out=['| '+' | '.join(names)+' |','| '+' | '.join(['---']*len(names))+' |']
    for _,r in frame.iterrows():
        row=[]
        for field,label,precision in columns:
            v=r[field]
            if field=='config_id':v=LABELS.get(v,v)
            row.append(str(v) if precision is None else f'{v:.{precision}f}')
        out.append('| '+' | '.join(row)+' |')
    return '\n'.join(out)

def select(stage,scope,names=None):
    x=s[s.stage.eq(stage)&s.scope.eq(scope)]
    if names is not None:
        x=x.set_index('config_id').loc[names].reset_index()
    return x

dev='development_2023_2025';diag='inspected_diagnostic_2026'
eventcols=[('config_id','Система',None),('brier','Brier ↓',6),('lift','Lift ↑',3),('forward_delta_bps','Forward Δ, bps ↑',2),('signals','Сигналы',0)]
forecastcols=[('config_id','Backbone / baseline',None),('pinball_bps','Pinball, log-bps ↓',3),('mae_bps','Median MAE, log-bps ↓',3),('coverage80','Покрытие 80% интервала',3)]
names=['baseline_reproduction','basis_train_120m','chronos2_small_zs','chronos2_small_ft','chronos2_small_zs_hgb_aug','chronos2_small_ft_hgb_aug','chronos2_synth_zs','chronos2_synth_ft','chronos2_synth_zs_hgb_aug','chronos2_synth_ft_hgb_aug','historical_path']
kzt=['baseline_reproduction','basis_train_120m','chronos2_synth_ft','chronos2_synth_ft_kzt','chronos2_synth_ft_hgb_aug','chronos2_synth_ft_kzt_hgb_aug']
fits=[json.loads(p.read_text()) for p in (OUT/'checkpoints').glob('*/*/fit_receipt.json')]
text=f'''# Предобученные временные модели: реальный benchmark V4

**Сильного общего улучшения V3 не найдено.** Проверены два актуальных pretrained backbone, настоящее дообучение всех весов и дополнительная специализация KZT. Лучшее уменьшение Brier относительно короткого V3 дал HGB с дообученными Small-признаками: **0.190617 → 0.187990, −1.38%**, но средняя forward-выгода его сигналов снизилась **47.69 → 34.96 bps**. Более длинный V3 уже имеет Brier **0.184850**, то есть остаётся сильнее и по этой метрике.

Наиболее аккуратный результат — HGB с **недообученными Synth-признаками**: Brier0.190519, lift1.436, forward50.31 bps. Это почти паритет с V3; доверительные интервалы не подтверждают улучшение. Самостоятельные NOW-модели поверх forecast-only признаков уступают действующему HGB. Эти измерения закрывают гипотезу «достаточно взять SOTA pretrained модель и дообучить на длинной FX-истории, чтобы получить сильный boost» в рамках данного фиксированного протокола.

## Что реально выполнено

- `autogluon/chronos-2-small`:27,934,624 параметра; zero-shot и4 независимых full-weight FT.
- `autogluon/chronos-2-synth`:118,985,888 параметров; zero-shot,4 pooled full-weight FT и4 дополнительных KZT-only FT.
- **12 обучений,2,800 optimizer steps**; совокупное измеренное время fits **{sum(x['fit_seconds'] for x in fits)/60:.2f} минуты CPU**, без повторного нейронного обучения для bootstrap.
- Имеются реальные safetensors, конфиги, hashes и receipts. Изменилось92.4–92.9% элементов весов Small,81.4–81.8% Synth и68.9–69.2% на этапе KZT. Все параметры были trainable; нулевые изменения отдельных весов возможны из-за нулевого градиента/округления. Это не linear probe.
- Сопоставлены40 систем×год decision-head результатов и4 исторических path-baseline результатов с обоими V3 ориентирами; полные таблицы сохраняют отрицательные результаты.

Обе модели имеют Apache-2.0 declaration, общий размер скачанных весов около588MB. **Synth по заявлению авторов обучен только на синтетике** и служит наиболее сильным контролем против запоминания реальных FX-рядов. Small создан2025-12-03, Synth2025-11-24: результаты2023–25 являются ретроспективным исследованием возможностей, а не исторической симуляцией доступности модели. Подробный аудит Chronos/Bolt, TimesFM2.5/3, Moirai2, Kronos и корпуса — в [SOURCES.md](./SOURCES.md), с первичными ссылками.

## Протокол и смысл NOW

В каждом outer году backbone видит только10 лет рядов ЦБ, заканчивающихся **до начала предыдущего calibration-года**. Например, для2023 обучение заканчивается2021-12-31, калибровка использует2022. Контекст256 наблюдений, horizon5;300 шагов full FT с lr1e−5. KZT stage2 использует только KZT того же разрешённого интервала:100 шагов, lr2e−6. Число шагов и lr не выбирались по outer результатам. Это ограниченное стохастическое обучение, а не доказательство сходимости или оптимальности всех гиперпараметров.

В каждый момент вход содержит только прошлые значения и текущий фиксинг. Пять валют одного cutoff образуют multivariate task; разные cutoffs имеют раздельные группы (`cross_learning=False`). Проверка perturbation показала, что замена более позднего соседнего task на значения+100 logunits меняет предыдущий прогноз лишь на4.77e−7 logunits — в пределах floating-point допуска.

**NOW сохранён без подмены:** все5 будущих фиксингов RUB/unit должны быть не ниже текущего. Меньшее RUB/unit выгоднее для покупки валюты; равенства считаются успехом. Marginal quantile forecasts сами по себе не задают вероятность этого joint path event. Мы используем14 forecast summaries: marginal survival estimates, медианы, ширины интервалов и Fréchet bounds. Только отдельный исторически обученный adapter оценивает NOW. **ZS означает zero-shot backbone; итоговая NOW probability — supervised+calibrated.** Произведение пяти marginal probabilities нигде не выдается за joint вероятность.

Adapter — logistic C=.1 либо исходный HGB с добавленными forecast features. Он обучается на тех же purged2 годах, что V3; Platt и frequency thresholds — на предыдущем purged1 годе. Cooldown воспроизводится по всей доступной score-истории calibration-года, включая хвост без labels. Backbone FT и adapter используют общий train-период; это стандартное supervised fitting, но training forecasts для head могут быть оптимистичны. Calibration/test labels не попадают в backbone/head fit. Между split label-окнами используется purge5.

Даты и labels в каждой паре совпадают: **3,635 currency-date rows /727 дат** в2023–25 и **780 rows /156 дат** в уже просмотренном2026. Эффективных независимых наблюдений существенно меньше числа строк: валюты коррелированы, а горизонты перекрываются. [README.md](./README.md) содержит полный воспроизводимый контракт.

## NOW и фактическая политика, development2023–25

{table(select(dev,'all',names),eventcols)}

Forward Δ — среднее преимущество выбранных дней над случайным днём **того же year×corridor**, с весом по числу сигналов. Это proxy на официальном курсе, не чистая прибыль реального перевода после spread/fees. Политики имеют один validation/cooldown алгоритм, но различаются порогами и числом событий; выигрыш одной метрики не означает доминирование при другом contact budget.

10,000 парных year-stratified month-block resamples сохраняют все валюты одного периода вместе. В каждом draw заново считаются cell random-day baselines и веса сигналов. Интервалы условны на обученных моделях и поиске; multiple-trial selection и model-refit uncertainty ими не устранены.

- **Small FT+HGB против V3:** ΔBrier−0.002626,95%CI[−0.007473,+0.001806]. ForwardΔ−12.72 bps,CI[−23.21,−1.94]. Улучшение proper score неубедительно; ухудшение событий заметно даже в этом exploratory interval. Число сигналов663→758, частота0.887→1.014 на corridor-week.
- **Synth ZS+HGB против V3:** ΔBrier−0.000098,CI[−0.001421,+0.001299]. ForwardΔ+2.63 bps,CI[−2.58,+7.91]. Число сигналов663→650. Нельзя заявлять подтверждённый boost.
- Small FT+HGB лучше V3 по Brier в12 из15 year×corridor cells, но в2025 часть эффекта меняется на отрицательный. Synth ZS+HGB лучше лишь в6 из15 cells. Полные stability/cadence/calibration разрезы — [cells.csv](./cells.csv), интервалы — [paired_intervals.csv](./paired_intervals.csv).

## Собственно прогноз распределения курса

{table(f[f.stage.eq(dev)&f.scope.eq('all')],forecastcols)}

Pinball и MAE указаны в **log-rate basis points**, поэтому численно не являются forward-выгодой перевода. Pinball усредняется по одинаковым13 quantiles и5 horizons. Квантили монотонно rearranged до оценки. Контроль GaussianRW использует нулевой drift и прошлую60-дневную волатильность; эмпирический контроль — только полностью завершившиеся5-step paths из trailing256 наблюдений.

GaussianRW лучший по среднему pinball. Synth ZS ближе всего: разница+0.653 log-bps,CI[−0.742,+1.967]; превосходства над RW нет. Median MAE Synth ZS хуже RW на5.540 log-bps,CI[2.259,8.645]. Fine tuning ухудшает средний forecast score у обеих моделей. Следовательно, небольшое улучшение HGB Brier от некоторых forecast-признаков не является доказательством лучшего самостоятельного вероятностного прогноза курса.

Сравнение Small и Synth не изолирует влияние synthetic pretraining: различается и размер28M против119M. Нельзя из этого опыта сделать вывод «синтетический pretraining сам по себе лучше реального».

## Специализация KZT

{table(select(dev,'KZT',kzt),eventcols)}

Stage2 дообучает именно веса Synth на KZT; inference task и алгоритм обучения decision-head остаются теми же. У HGB augmentation Brier улучшается **0.189971 → 0.184963,−2.64%** относительно pooled FT,Δ−0.005008,CI[−0.009718,−0.000152]. Это положительный, но пограничный exploratory результат одного эксперимента. ForwardΔ повышается24.18→30.24 bps; интервал прироста[−14.28,+25.40] bps широк. При этом исходный V3 даёт более сильную event-policy, а длинный V3 — лучший Brier.

Сам прогноз KZT после stage2 тоже немного лучше pooled FT: pinball−0.651 log-bps,CI[−1.670,+0.397]. GaussianRW всё ещё лучше специализированного Synth на2.357 log-bps,CI[0.812,3.957]. Улучшение специализации относительно слабого pooled предшественника нельзя выдавать за победу над реальным production-oriented baseline.

## Уже просмотренный partial2026

{table(select(diag,'all',['baseline_reproduction','basis_train_120m','chronos2_small_ft_hgb_aug','chronos2_small_zs_hgb_aug','chronos2_synth_zs_hgb_aug','chronos2_synth_ft_hgb_aug']),eventcols)}

Small FT+HGB теряет выигрыш Brier:0.209438 против0.206627 V3. Synth ZS+HGB сохраняет практически тот же Brier, но forward-выгода ниже. Для KZT stage2 HGB Brier0.239026 против0.242651 pooled FT;95%CI разницы[−0.012192,+0.004064] включает ноль. Forward39.24 против31.11 bps, всего33 против32 сигналов. Эти данные уже были доступны для инспекции и не являются новым независимым test.

Модельные80%-ные интервалы курса в2026 имеют покрытие только66.7–67.8%; GaussianRW около73.6%. Platt калибрует NOW probability, а не сами quantile forecasts. Следовательно, сырые foundation-интервалы нельзя использовать как калиброванные клиентские гарантии.

## Решение и следующий содержательный шаг

**Не заменять действующее решение на tested foundation pipeline.** Сохранить Synth ZS forecast-features как дешёвый challenger и KZT stage2 как доказанно выполненный вариант domain adaptation с ограниченным позитивным proper-score результатом. Для презентации корректная формулировка: «Мы реально проверили pretrained priors и дообучение, но экономический эффект не подтвердился; классические модели с правильными данными остаются сильнее».

Если продолжать это направление, основная новая гипотеза — fine tuning самого encoder под **joint survival / decision loss**, вместо оптимизации marginal quantiles и последующей компрессии в14 summaries. Нужен фиксированный inner temporal split, сравнение с одинаковой архитектурой from-scratch и predeclared contact/risk budgets. Другой самостоятельный путь — финансовый Kronos на настоящих OHLC MOEX/KASE, с проверкой corpus cutoff и без искусственной подстановки фиксинга в OHLC. Ни одно из этих продолжений в данном отчёте не объявлено выполненным.

Файлы: [benchmark.py](./benchmark.py), [assess.py](./assess.py), [summary.csv](./summary.csv), [forecast_summary.csv](./forecast_summary.csv), [forecast_paired_intervals.csv](./forecast_paired_intervals.csv), [verification_receipt.json](./verification_receipt.json). Все neural checkpoints и forecast caches сохранены; повторный bootstrap не требует refit.
'''
(OUT/'REPORT.md').write_text(text)
print(OUT/'REPORT.md')
