# Независимая перепроверка сохранённых KZT V0 событий

Это fixed-event диагностика: даты OOT-сигналов заморожены, truth пересчитан по Q&A. Она не исправляет leakage и не заменяет полный nested walk-forward rerun.
`h_moex_rows` считает следующие строки сохранённой MOEX-aligned панели; это не distinct CBR publications и не календарные дни.

| Scenario | h_moex_rows | δ, bps | Hits/signals | Random | Lift | Lift block-bootstrap 95% | ±h bps | Forward bps | Fisher p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| favorable_now | 5 | 0 | 4/7 | 0.253 | 2.260 | [0.726; 4.475] | 82.0 | -0.4 | 0.072 |
| window_closing | 5 | 35 | 9/10 | 0.634 | 1.419 | [1.137; 1.724] | 19.6 | 144.5 | 0.069 |

## Sensitivity: полностью разрешённый outcome внутри каждого fold

Последние h_moex_rows строк каждого test fold исключены, поэтому выборка ещё меньше.

| Scenario | h_moex_rows | δ, bps | Hits/signals | Random | Lift | ±h bps | Forward bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| favorable_now | 5 | 0 | 4/6 | 0.243 | 2.747 | 100.2 | 18.3 |
| window_closing | 5 | 35 | 8/9 | 0.640 | 1.390 | 21.7 | 134.5 |

## Частота на полном OOT-span

Период: 2024-11-11…2026-08-10 (92 календарных недель).

| Scenario | Signals | Signals/week | Silent weeks | Max gap incl. boundaries |
|---|---:|---:|---:|---:|
| all | 17 | 0.187 | 82.6% | 473 d |
| favorable_now | 7 | 0.077 | 92.4% | 473 d |
| window_closing | 10 | 0.110 | 90.2% | 479 d |

## Интерпретация

- `h_moex_rows` — следующие строки MOEX-aligned feature panel, не distinct CBR publications и не календарные дни; CBR значения внутри окна могут быть carry-forward.
- Ветки `favorable_now` и `window_closing` нельзя сравнивать с опубликованными headline-числами: там другие/некорректные truth-функции.
- Интервалы — диагностический circular moving-block bootstrap по 20 MOEX-aligned rows; при 7/10 сигналах они неизбежно нестабильны.
- Fisher p и circular-shift p не скорректированы за перебор моделей, порогов, горизонтов и сценариев.
- `required_signals` ниже — оптимистичная нижняя граница мощности, а не план эксперимента.
