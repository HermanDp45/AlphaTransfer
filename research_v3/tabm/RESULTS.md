# TabM: вероятностный прирост не превратился в более полезные сигналы

Фактически выполнено 2026-09-05. Итог: **не заменять incumbent policy этим TabM/blend**. На development HGB+TabM улучшает Brier на 2.59%, но уменьшает lift и forward advantage. На частичном diagnostic 2026 вероятностный прирост отсутствует. Это полезный отрицательный результат о переносе probability scoring в decision quality.

Исходная работа и реализация: [TabM, ICLR 2025](https://arxiv.org/abs/2410.24210), [официальный код](https://github.com/yandex-research/tabm). Мы проверили одну компактную конфигурацию, а не полный HPO TabM и не все возможные feature embeddings. Научная статья не служит доказательством качества на данном FX-датасете.

## Что было запущено

Два набора признаков: base из 14 core+vol и base + один CNY spot-minus-fixing. Четыре годовых outer folds 2023–2026. В каждом — HGB incumbent, TabM и convex blend с долей TabM из 0/.25/.5/.75/1, выбранной на предшествующем validation году. Каждый алгоритм проходит тот же incumbent Platt/corridor-threshold/cooldown. Epoch selection TabM: последние 63 train dates после purge5; затем refit на train2y. Никакие outer labels не выбирают epochs, blend weight или порог.

TabM: 2×128 блоки, k=16, dropout=.1, BCE каждого ensemble member, среднее вероятностей на inference; train-only median imputation и quantile-to-normal. Python3.11.15, torch2.14.0, tabm0.0.3, sklearn1.9.0, numpy2.4.6, pandas3.0.5. Два CPU threads; основной этап после импортов занял 27.5 секунды. Это benchmark компактного supervised tabular DL, не обучение финансовой foundation model.

## Prediction-level parity

Для HGB base и HGB+basis совпали все 4,415 probabilities и candidate masks с существующими development+diagnostic артефактами: максимальная абсолютная ошибка **0.0**. Значит, сравнение не объясняется изменённым evaluator или версией sklearn. Evidence: `output/incumbent_parity.json`.

## Development 2023–2025: 727 уникальных дат

Все policy-показатели ниже относятся к corridor candidates, не к синтетическому all-five portfolio и не к delivered contacts. Bps — matched-cell deltas к random-day reference.

| Модель | Brier ↓ | Прирост Brier к своему HGB | Candidates | Lift ↑ | Forward bps ↑ | Symmetric bps ↑ | Min cell weekly coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGB base | .218714 | baseline | 742 | 1.0890 | +6.02 | −1.05 | 63.3% |
| TabM base | .215844 | +1.31% | 820 | .9957 | +1.60 | −6.89 | 63.3% |
| Blend base | .217442 | +.58% | 783 | 1.0331 | +2.07 | −4.23 | 63.3% |
| HGB + basis | .190617 | baseline | 663 | 1.4352 | +47.69 | +26.80 | 66.7% |
| TabM + basis | .187522 | +1.62% | 735 | 1.2530 | +32.22 | +18.61 | 65.3% |
| Blend + basis | **.185688** | **+2.59%** | 689 | 1.3215 | +38.41 | +19.52 | 71.4% |

Blend+basis даёт −0.004929 Brier, но −0.1137 lift и −9.27 forward bps против incumbent. Его weekly coverage немного выше, но старый 90% gate всё ещё не проходит. Для оценки probabilistic calibration это перспективно; для текущей communication policy нет превосходства.

### Парные доверительные интервалы

10,000 month-block bootstrap draws, стратифицированных по outer year; пять corridors каждой даты переносятся совместно. Интервалы exploratory, выбор модели и весь предшествующий исследовательский поиск не включены.

| Challenger vs соответствующий HGB | ΔBrier (меньше лучше) | 95% CI |
|---|---:|---:|
| TabM base | −.002870 | [−.007320; +.001413] |
| Blend base | −.001272 | [−.004855; +.002140] |
| TabM + basis | −.003095 | [−.009614; +.003370] |
| Blend + basis | −.004929 | **[−.010220; +.000254]** |

Все интервалы включают ноль. Первичный 2,000-draw screen для blend+basis имел очень близкую к нулю отрицательную верхнюю границу; после 10,000 она стала положительной. Финальным является 10,000-draw файл. Такой borderline результат нельзя описывать как установленную статистическую значимость.

### Устойчивость по годам и ячейкам

| Модель | 2023 ΔBrier | 2024 ΔBrier | 2025 ΔBrier | Улучшенные development cells |
|---|---:|---:|---:|---:|
| TabM base | −.004086 | −.010697 | +.006206 | 10/15 |
| Blend base | .000000 | −.009984 | +.006206 | 5/15; ещё 5 ties |
| TabM + basis | +.004464 | −.008502 | −.005224 | 10/15 |
| Blend + basis | −.002437 | −.008502 | −.003832 | **15/15** |

15/15 для blend — переносимость условно на один общий market path; коррелированные коридоры не превращают результат в 15 независимых доказательств. Это согласуется с широким date-block interval.

## Diagnostic 2026: 156 дат, 13 января — 25 августа

Частичный период был просмотрен до v3 и **не является confirmation**.

| Модель | Brier | Candidates | Lift | Forward bps |
|---|---:|---:|---:|---:|
| HGB base | .238059 | 166 | .9564 | +11.41 |
| TabM base | .242027 | 170 | 1.0204 | −.46 |
| Blend base | .238859 | 177 | 1.0893 | +14.91 |
| HGB + basis | **.206627** | 168 | **1.3753** | **+37.77** |
| TabM + basis | .211265 | 171 | 1.2301 | +27.12 |
| Blend + basis | .206668 | 175 | 1.2706 | +27.04 |

TabM+basis ухудшает Brier на 2.24%; blend практически равен HGB (+.000041 абсолютного Brier, −.02% относительного improvement). Blended probability улучшена в 3/5 cells, TabM — в 1/5. Policy advantage всё ещё хуже incumbent, поэтому нет оснований продвигать этот neural challenger.

## Обучение и ensemble weights

| Features | Test year | Выбранные epochs | TabM weight, prior validation |
|---|---:|---:|---:|
| Base | 2023 | 43 | 0 |
| Base | 2024 | 17 | .75 |
| Base | 2025 | 2 | 1 |
| Base | 2026 | 19 | .25 |
| Base+basis | 2023 | 21 | .25 |
| Base+basis | 2024 | 17 | 1 |
| Base+basis | 2025 | 5 | .5 |
| Base+basis | 2026 | 12 | .25 |

Ранний stop на 2/5 epochs — фактический исход temporal validation, не ручной выбор после test. Он показывает, что длительное neural fitting не обязательно полезно в меняющемся режиме. Более сильная regularization/feature embeddings/ensemble search могут быть отдельными будущими trials, но их gain этим запуском не доказан.

## Воспроизводимые артефакты и статус

`run_tabm_benchmark.py` воспроизводит обучение; `finalize_results.py` пересчитывает 10,000 intervals, paired cell/year stability и exact parity без refit. `_SUCCESS.json` обновляет hashes и явно отмечает единственное изменение runner после обучения: default bootstrap 2000→10000, не параметры модели. Исходный training code hash сохранён в frozen protocol, финальный readout hash — в receipt.

Полные метрики: `output/aggregate_metrics.csv`, `fold_corridor_metrics.csv`, `paired_year_stability.csv`, `paired_cell_stability.csv`, `paired_brier_intervals.csv`; probabilities: `predictions.csv`; fitted models: `artifacts/`.

**Рекомендация:** сохранить blend как исследовательский вероятностный challenger, использовать данный результат как наглядное обоснование отдельной decision-metric. В текущий product winner его не добавлять. Дополнительный neural complexity пока не окупает ухудшение policy utility и отсутствие подтверждения в 2026.
