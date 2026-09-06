"""Build the human report, unified metrics export, and artifact manifest."""
from pathlib import Path
import hashlib, json, sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")

HERE = Path(__file__).resolve().parent


def pct(value):
    return f"{100 * value:.1f}%".replace(".", ",")


def num(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    verification = json.loads((HERE / "verification.json").read_text())
    final_verification = json.loads((HERE / "final_fit/verification.json").read_text())
    assert verification["status"] == final_verification["status"] == "PASS"
    selection = json.loads((HERE / "selection.json").read_text())
    summary = pd.read_csv(HERE / "summary.csv")
    yearly = pd.read_csv(HERE / "by_year.csv")
    intervals = pd.read_csv(HERE / "paired_intervals.csv")
    report_rows = []
    for row in summary.to_dict("records"):
        report_rows.append(dict(row_type="aggregate", year="", **row))
    for row in yearly.to_dict("records"):
        report_rows.append(dict(row_type="year", period=str(int(row["year"])), **row))
    pd.DataFrame(report_rows).to_csv(HERE / "REPORTING_ALL_METRICS.csv", index=False)

    selected = summary[summary.history_mode.eq(selection["selected_recipe"])].iloc[0]
    control = summary[summary.history_mode.eq("120m")].iloc[0]
    annual = yearly[yearly.history_mode.eq(selection["selected_recipe"])].sort_values("year")
    table = ["| Год | Lift | Hit rate | Сигналов/неделю | Эффект, б.п. | Coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in annual.itertuples():
        table.append(f"| {row.year} | {num(row.lift)}× | {pct(row.hit_rate)} | {num(row.signals_per_corridor_week, 2)} | +{num(row.forward_delta_bps, 1)} | {pct(row.week_coverage)} |")
    all_ci = intervals[intervals.period.astype(str).eq("all")].iloc[0]
    receipt = json.loads((HERE / "final_fit/receipt.json").read_text())
    policy = json.loads((HERE / "final_fit/policy_calibration_receipt.json").read_text())
    report = f"""# H5: полная история и редкий режим уведомлений

## Результат

В совпадающем rolling OOT сравнении 2024–2026 выбрана **TabM H5 с расширяющейся историей от 2010 года**. Она прошла все заранее заданные ограничения: итоговая частота находится в диапазоне 0,60–0,70 сигнала в неделю, годовая частота — в диапазоне 0,45–0,85, а lift не ниже 1,30 и эффект к среднему дню положителен в каждом году.

На 641 вневыборочной дате модель сформировала {int(selected.signals)} сигналов: lift **{num(selected.lift)}×**, hit rate **{pct(selected.hit_rate)}** при base hit {pct(selected.base_hit)}, эффект **+{num(selected.forward_delta_bps, 1)} б.п.**, regret {num(selected.regret_bps, 1)} б.п. и частота **{num(selected.signals_per_corridor_week, 3)} сигнала в неделю**. Это H5-таргет: текущий RUB/KZT не хуже минимума следующих пяти фактических сессий CBR.

{chr(10).join(table)}

Coverage показывает долю недель хотя бы с одним сигналом и ожидаемо ниже частоты: в некоторых активных неделях допускается два контакта. Принудительного заполнения молчащих недель нет.

## Сравнение длины истории

Контроль на последних 120 месяцах дал lift {num(control.lift)}×, hit rate {pct(control.hit_rate)}, +{num(control.forward_delta_bps, 1)} б.п. и {num(control.signals_per_corridor_week, 3)} сигнала в неделю. Он сохранил хорошие годовые lift и utility, но вышел за итоговый продуктовый диапазон частоты: {num(control.signals_per_corridor_week, 3)} > 0,70. Полная история дала более высокий итоговый lift (+{num(selected.lift-control.lift)}), эффект (+{num(selected.forward_delta_bps-control.forward_delta_bps, 1)} б.п.) и немного меньший Brier ({num(selected.brier, 6)} против {num(control.brier, 6)}), одновременно попав в требуемую частоту.

Парный месячный bootstrap для full-history минус 120m носит описательный post-selection характер. Итоговые дельты: lift +{num(all_ci.lift_delta)}, 95% интервал [{num(all_ci.lift_ci_low)}; {num(all_ci.lift_ci_high)}]; forward delta +{num(all_ci.forward_delta_bps_delta, 1)} б.п., [{num(all_ci.forward_delta_bps_ci_low, 1)}; {num(all_ci.forward_delta_bps_ci_high, 1)}]; Brier {num(all_ci.delta_brier, 6)}, [{num(all_ci.ci_low, 6)}; {num(all_ci.ci_high, 6)}]. Интервалы пересекают ноль, поэтому эксперимент подтверждает выбор по устойчивости и продуктовому режиму, но не доказывает статистически значимое превосходство истории любой длины.

## Причинная политика `rare65`

Каждый score сравнивается только с 63 предшествующими score. Для калибровочного года ранги начинаются с отдельного 63-сессионного warmup до его начала. Порог выбирается на созревших метках калибровочного года с первичной целью 0,65 сигнала в неделю. Затем история до cutoff инициализирует cooldown; тестовые ранги используют только прошлую историю и уже прошедшие тестовые score. Cooldown равен двум сессиям, максимум — два контакта в календарную неделю.

Проверка с полностью испорченными тестовыми outcomes оставила все контакты неизменными. Это подтверждает, что target, forward return и regret теста не входят в решение. Политика финального refit выбрала percentile-порог {num(policy['threshold'], 3)} и дала {num(policy['calibration_realized_signals_per_week'], 3)} сигнала в неделю на последней калибровке.

## Временной протокол и финальный refit

Для тестов 2024/2025/2026 train начинается 2010-01-01 и заканчивается перед отдельным 12-месячным calibration. В полной истории было 3208/3455/3703 строк, на 747/993/1240 больше соответствующего 120m-контроля. Validation, history, test, H5 outcomes и warmup совпадают с контролем точно. Дата созревания метки — пятая следующая фактическая сессия; она строго предшествует каждой границе. Медианная импутация, квантильное преобразование и missing indicators обучаются только на train. Ранние пропуски Halyk и Treasury не заполняются будущими значениями.

Финальная модель переобучена на проверенной локальной панели до cutoff 2026-09-05: {receipt['train_rows']} train-строк с 2010-01-01 по {str(receipt['train_end'])[:10]}, {receipt['validation_rows']} созревших calibration-строк и {receipt['selected_epochs']} эпох. `final_fit/bundle.json` содержит веса, препроцессор, Platt-калибратор, causal policy, состояние и хеши источников. Этот refit не имеет будущего confirmatory-теста; публичные показатели выше относятся к rolling OOT recipe 2024–2026.

## Артефакты

- `selection.json`, `ranking.csv`: зафиксированный выбор и gates.
- `summary.csv`, `by_year.csv`, `REPORTING_ALL_METRICS.csv`: все итоговые метрики.
- `predictions.csv.gz`: OOT score, causal ranks и контакты обеих длин истории.
- `paired_intervals.csv`: 10 000 парных месячных bootstrap-повторов.
- `policies.json`: ежегодные калибраторы, пороги и диагностическая сетка.
- `receipts.json`, `checkpoint_replay.csv`, `verification.json`: границы, fingerprints и replay.
- `final_fit/`: выбранный production source bundle.

Воспроизведение: `experiment.py` → `evaluate.py` → `verify.py` → `final_fit.py` → `report.py`. Для обучения требуется окружение с `tabm`, `rtdl-num-embeddings`, PyTorch и PyArrow.
"""
    (HERE / "REPORT.md").write_text(report)
    files = [path for path in HERE.rglob("*") if path.is_file() and path.name != "MANIFEST.json" and
             "__pycache__" not in path.parts and "checkpoints" not in path.parts]
    save(HERE / "MANIFEST.json", dict(status="complete", selected_recipe=selection["selected_recipe"],
         files={str(path.relative_to(HERE)): dict(sha256=sha(path), bytes=path.stat().st_size) for path in sorted(files)}))
    print("H5 REPORT PASS", len(files), "manifested files", flush=True)


if __name__ == "__main__":
    main()
