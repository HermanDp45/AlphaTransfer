# Сводное сравнение AlphaTransfer v3

Реестр содержит **166 строк** по моделям, горизонтам и периодам: [COMPARISON.csv](COMPARISON.csv). Он пересобирается командой `python research_v3/build_comparison.py` из готовых исходных таблиц. Повторяющиеся модели из разных веток сохранены с `origin_folder` и `source_file`, чтобы provenance не потерялся.

**Все основные Δ Brier рассчитаны к старому `hgb_plus_cnyrub_basis` на том же горизонте и периоде.** Положительное относительное улучшение означает уменьшение Brier; `delta_brier = new − old`, поэтому для него отрицательное значение лучше. Lift, число кандидатов и reference gain везде относятся к `candidate_signal`, а не к синтетическому портфелю или доставленным пушам.

## h=5: development 2023–2025

| Вариант | Brier ↓ | Улучшение Brier к старому | Lift | Кандидаты | Reference gain, bps | Δ gain к старому, bps |
|---|---:|---:|---:|---:|---:|---:|
| Старый CNY basis | 0.190617 | +0.00% | 1.435 | 663 | 47.69 | +0.00 |
| Train 120 месяцев | 0.184850 | +3.03% | 1.371 | 723 | 40.08 | -7.61 |
| Calibration 3 месяца | 0.185818 | +2.52% | 1.371 | 727 | 43.60 | -4.08 |
| Long + ранняя история CNY | 0.187480 | +1.65% | 1.342 | 737 | 38.19 | -9.49 |
| Long/short ensemble | 0.185955 | +2.45% | 1.389 | 710 | 42.92 | -4.76 |
| TabM + basis | 0.187522 | +1.62% | 1.253 | 735 | 32.22 | -15.46 |
| HGB/TabM blend | 0.185688 | +2.59% | 1.321 | 689 | 38.41 | -9.27 |
| Treasury inflation lag7 | 0.188583 | +1.07% | 1.456 | 708 | 43.72 | -3.96 |
| FRED + CBOE + GPR | 0.199785 | -4.81% | 1.275 | 752 | 27.22 | -20.47 |
| Long120 + Treasury lag7 | 0.183295 | +3.84% | 1.343 | 767 | 31.52 | -16.17 |

Здесь 3 635 строк, сгруппированных в 727 дат. Reference gain — средняя разница с базой внутри год×коридор для официального курса, в bps. Это не исполненная банковская экономия. Улучшение Brier не означает улучшения policy-выгоды: например, long120 + Treasury улучшает proper score и одновременно снижает lift/reference gain. Нельзя превращать выбор лучшей просмотренной строки в подтверждение на независимой выборке.

**Два разных сравнения для long120 + Treasury:** общий эффект **к старому incumbent** — Δ Brier **−0.007322**, 95% month-block CI **[−0.014351; −0.000610]**, улучшение в 2/3 годах и 12/15 ячейках; этот интервал **не включает ноль**. Добавочный эффект **к уже длинной модели** — Δ **−0.001556**, CI **[−0.004148; +0.000898]**, улучшение в 3/3 годах и 13/15 ячейках; этот интервал включает ноль. Первый результат не доказывает, что именно Treasury дал весь прирост. Оба интервала retrospective/post-selection, без коррекции всей просмотренной семьи моделей, поэтому не являются confirmatory. В CSV общий эффект и интервал находятся в `delta_brier*`, добавочный — в `matched_*`; baseline не подменяется. Источники: [общий эффект](models/external_combo_vs_incumbent_ci.csv), [добавочный эффект](external_data/long_combo/development_paired_audit.csv).

## h=5: diagnostic 2026

| Вариант | Brier ↓ | Улучшение Brier к старому | Lift | Кандидаты | Reference gain, bps | Δ gain к старому, bps |
|---|---:|---:|---:|---:|---:|---:|
| Старый CNY basis | 0.206627 | +0.00% | 1.375 | 168 | 37.77 | +0.00 |
| Train 120 месяцев | 0.208406 | -0.86% | 1.337 | 173 | 29.64 | -8.13 |
| Calibration 3 месяца | 0.233049 | -12.79% | 1.387 | 173 | 20.87 | -16.90 |
| Long + ранняя история CNY | 0.209299 | -1.29% | 1.273 | 177 | 24.37 | -13.40 |
| Long/short ensemble | 0.206658 | -0.01% | 1.409 | 170 | 36.30 | -1.47 |
| TabM + basis | 0.211265 | -2.24% | 1.230 | 171 | 27.12 | -10.66 |
| HGB/TabM blend | 0.206668 | -0.02% | 1.271 | 175 | 27.04 | -10.73 |
| Treasury inflation lag7 | 0.204926 | +0.82% | 1.418 | 169 | 35.65 | -2.12 |
| FRED + CBOE + GPR | 0.224824 | -8.81% | 1.445 | 164 | 39.78 | +2.01 |
| Long120 + Treasury lag7 | 0.208938 | -1.12% | 1.306 | 175 | 24.64 | -13.13 |

Период неполный и ранее просмотрен: 780 строк, 156 дат. Он не используется как новый holdout. Для long120 + Treasury общий Δ Brier к старому равен **+0.002311**, CI **[−0.008411; +0.014146]**: point estimate ухудшается, интервал включает ноль.

## Границы сопоставимости

- Для development h=1/3/5/10/20 и diagnostic h=5 denominator взят прямо из frozen v2 `final_solution/model_bundle`. Корневые `baseline_reproduction` проверены против этих значений по Brier, lift, candidate count и reference gain.
- Для diagnostic h=1/3/10/20 frozen v2 таблица отсутствует. Использовано повторное исполнение **старого метода** `baseline_reproduction` на том же горизонте; строки помечены `old_method_recomputed_no_frozen_diagnostic`. Горизонты между собой не смешиваются.
- ICE имеет отдельный train2024/validation2025 protocol. Его глобальные `delta_brier` и `relative_brier_improvement` намеренно пусты; сравнение с matched baseline лежит в `matched_*`. Эту небольшую исследовательскую выборку нельзя представить как обычный old-baseline uplift.
- Synthetic behavior simulation исключена из FX/ML-ledger; результаты симуляции не смешиваются с фактическими прогнозами курсов. Флаг `synthetic_behavior_excluded=True` отражает это для каждой строки.
- FRED/CBOE/all-new отмечены как restricted-source counterfactual. Treasury-реплика, GPR latest-snapshot caveat, лаговая sensitivity и post-hoc long combo отражены в `flags`. Отсутствие numerical Bloomberg ablation не заменено выдуманной строкой; доступность источников описана в [source_access_decisions.csv](external_data/source_access_decisions.csv).
- Пропуски внешних признаков в ранней длинной истории, лимиты публикационного времени, отсутствие executable quotes и многократный просмотр данных сохраняются. Сводный CSV не повышает статус моделей до production или подтверждённой клиентской выгоды.

Источники: `models/summary_h*.csv`; `external_data/*metrics.csv`; `external_data/long_combo/*metrics.csv`; `tabm/output/aggregate_metrics.csv`, сверенный со `scorecard.csv`. [COMPARISON_MANIFEST.json](COMPARISON_MANIFEST.json) фиксирует hashes, finished parity и проверки.
