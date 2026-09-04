#!/usr/bin/env python3
"""Builds presentation-ready SVGs from the canonical backtest artifacts."""

from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import cairosvg
except (ImportError, OSError):  # SVG is canonical; PNG export is optional.
    cairosvg = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "review_artifacts/quant_macro/results-final-20260904-v2"
FRONTIER = ROOT / "review_artifacts/quant_macro/cadence_quality_frontier.csv"
OUT = Path(__file__).resolve().parent / "research_charts"

W, H = 1600, 900
RED = "#EF3124"
RED_DARK = "#C7231A"
INK = "#171717"
MUTED = "#676767"
LINE = "#D9D9D9"
PALE = "#F6F6F6"
PALE_RED = "#FFF0EF"
PALE_GREEN = "#ECF7F0"
GREEN = "#1F8A55"
AMBER = "#B56A00"
WHITE = "#FFFFFF"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def svg_start(title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title)}</title>",
        f"<desc id=\"desc\">{esc(description)}</desc>",
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000000" flood-opacity="0.08"/></filter>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8A8A8A"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{WHITE}"/>',
    ]


def text(x: float, y: float, value: object, size: int = 26, weight: int = 400,
         fill: str = INK, anchor: str = "start", extra: str = "") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" {extra}>{esc(value)}</text>'
    )


def multiline(x: float, y: float, lines: list[str], size: int = 24, weight: int = 400,
              fill: str = INK, line_height: float = 1.25, anchor: str = "start") -> str:
    spans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_height
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{"".join(spans)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str = WHITE,
         stroke: str = "none", radius: float = 18, stroke_width: float = 1,
         extra: str = "") -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {extra}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = LINE,
         width: float = 2, dash: str | None = None, extra: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}"{dash_attr} {extra}/>'
    )


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = WHITE,
           stroke_width: float = 3) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def save(name: str, content: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / name
    svg_path.write_text("\n".join(content + ["</svg>"]) + "\n", encoding="utf-8")
    if cairosvg is not None:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(svg_path.with_suffix(".png")),
            output_width=W,
            output_height=H,
        )


def build_slide_1() -> None:
    metrics = rows(RESULTS / "development_metrics.csv")
    selected = {
        row["config_id"]: row
        for row in metrics
        if row["horizon_cbr_rows_pub_proxy"] == "5"
    }
    audit = next(
        row for row in rows(RESULTS / "benchmark_comparison_audit_h5.csv")
        if row["comparison_id"] == "cny_basis_hgb_vs_hgb_base"
    )

    model_rows = [
        ("Компактный + CNY-дислокация", "hgb_plus_cnyrub_basis"),
        ("Decision stump + CNY", "hgb_cnyrub_basis_stump"),
        ("Полный research-набор", "hgb_full_research"),
        ("Базовая HGB", "hgb_base"),
        ("Все public fast", "hgb_all_public_fast"),
    ]
    best = selected["hgb_plus_cnyrub_basis"]
    base = selected["hgb_base"]
    ap_gain = (float(best["average_precision"]) / float(base["average_precision"]) - 1) * 100
    auc_delta = (float(best["roc_auc"]) - float(base["roc_auc"])) * 100

    out = svg_start(
        "Компактный сигнал дал лучший backtest",
        "Сравнение Brier score и ключевых диагностических метрик на rolling out-of-time backtest 2023–2025.",
    )
    out += [
        text(72, 78, "Лучший результат дал один сильный рыночный сигнал — не максимум признаков", 37, 700),
        text(72, 120, "Rolling OOT · 2023–2025 · горизонт 5 строк ЦБ · 3 635 наблюдений · 727 дат решений", 23, 400, MUTED),
        rect(70, 165, 900, 620, WHITE, LINE, 24, 1.5, 'filter="url(#shadow)"'),
        text(110, 220, "Brier score: ниже — лучше", 27, 700),
    ]

    x0, x1 = 470, 910
    min_b, max_b = 0.18, 0.24
    for tick in [0.18, 0.20, 0.22, 0.24]:
        x = x0 + (tick - min_b) / (max_b - min_b) * (x1 - x0)
        out.append(line(x, 246, x, 720, "#ECECEC", 1))
        out.append(text(x, 750, fmt(tick, 2), 18, 400, MUTED, "middle"))

    for i, (label, key) in enumerate(model_rows):
        row = selected[key]
        value = float(row["brier"])
        y = 285 + i * 92
        is_best = key == "hgb_plus_cnyrub_basis"
        out.append(text(110, y + 8, label, 23, 700 if is_best else 400, INK))
        out.append(line(x0, y, x1, y, "#EFEFEF", 8))
        x = x0 + (value - min_b) / (max_b - min_b) * (x1 - x0)
        out.append(circle(x, y, 13 if is_best else 10, RED if is_best else "#8F8F8F"))
        out.append(text(x + 22, y + 8, fmt(value, 4), 22, 700 if is_best else 400,
                        RED_DARK if is_best else INK))

    out += [
        rect(1010, 165, 520, 175, PALE_RED, "none", 24),
        text(1050, 215, "−12,8%", 52, 700, RED_DARK),
        text(1050, 255, "Brier к базовой HGB", 25, 700),
        text(1050, 292, f"95% CI Δ: {fmt(float(audit['ci95_low']), 4)}…{fmt(float(audit['ci95_high']), 4)}", 20, 400, MUTED),
        text(1050, 321, f"Holm p = {fmt(float(audit['holm_p_brier_improvement']), 4)}", 20, 400, MUTED),
        rect(1010, 365, 245, 170, PALE, "none", 24),
        text(1045, 420, f"+{fmt(ap_gain, 1)}%", 42, 700, INK),
        text(1045, 460, "Average Precision", 21, 700),
        text(1045, 495, "0,357 → 0,507", 20, 400, MUTED),
        rect(1280, 365, 250, 170, PALE, "none", 24),
        text(1315, 420, f"+{fmt(auc_delta, 1)} п.п.", 42, 700, INK),
        text(1315, 460, "ROC AUC", 21, 700),
        text(1315, 495, "0,551 → 0,723", 20, 400, MUTED),
        rect(1010, 560, 245, 170, PALE_GREEN, "none", 24),
        text(1045, 615, "1,435×", 42, 700, GREEN),
        text(1045, 655, "lift сигнала", 21, 700),
        text(1045, 690, "663 события", 20, 400, MUTED),
        rect(1280, 560, 250, 170, PALE_GREEN, "none", 24),
        text(1315, 615, "+47,7 б.п.", 42, 700, GREEN),
        text(1315, 655, "Δ forward-reference", 21, 700),
        text(1315, 690, "+13,7 п.п. hit-rate", 20, 400, MUTED),
        rect(70, 810, 1460, 46, PALE, "none", 12),
        text(95, 841, "Устойчивость: 3/3 лет и 15/15 fold×corridor ячеек по Brier. Простая модель ≈ HGB: сигнал важнее сложности.", 21, 600),
        text(1530, 884, "Ретроспективная post-selection диагностика; не клиентская экономия и не production-результат.", 17, 400, MUTED, "end"),
    ]
    save("01_backtest_quality.svg", out)


def build_slide_2() -> None:
    data = [
        row for row in rows(RESULTS / "diagnostic_corridor_candidate_uncertainty_h5.csv")
        if row["track"] == "contract_required_market_data"
        and row["config_id"] == "hgb_plus_cnyrub_basis"
    ]
    data.sort(key=lambda row: ["AMD", "KGS", "KZT", "TJS", "UZS"].index(row["corridor"]))

    out = svg_start(
        "Матрица результатов по пяти валютным коридорам",
        "Lift, прирост hit-rate, forward-reference delta и cadence для AMD, KGS, KZT, TJS и UZS.",
    )
    out += [
        text(72, 78, "Качество переносится на 5/5 коридоров — но cadence пока блокирует запуск", 37, 700),
        text(72, 120, "Одна и та же компактная модель · cell-standardized оценки · 95% block-bootstrap CI", 23, 400, MUTED),
    ]

    x = [70, 245, 390, 700, 1040, 1370, 1530]
    header_y, row_h = 177, 112
    headers = ["Коридор", "N", "Lift", "Δ hit-rate, п.п.", "Δ forward-ref, б.п.", "Cadence min", ""]
    out.append(rect(70, 155, 1460, 72, INK, "none", 18))
    for i, label in enumerate(headers[:-1]):
        out.append(text(x[i] + 18, header_y + 25, label, 20, 700, WHITE))

    for i, row in enumerate(data):
        y = 242 + i * row_h
        is_kzt = row["corridor"] == "KZT"
        fill = PALE_RED if is_kzt else (WHITE if i % 2 == 0 else PALE)
        stroke = RED if is_kzt else "none"
        out.append(rect(70, y, 1460, 92, fill, stroke, 14, 2.5 if is_kzt else 0))
        out.append(text(92, y + 38, row["corridor"], 28, 700, RED_DARK if is_kzt else INK))
        if is_kzt:
            out.append(rect(145, y + 15, 72, 32, RED, "none", 16))
            out.append(text(181, y + 38, "MVP", 15, 700, WHITE, "middle"))
        out.append(text(278, y + 57, row["signal_count"], 24, 600, INK))

        lift = float(row["cell_standardized_lift"])
        lift_lo = float(row["cell_standardized_lift_ci95_low"])
        lift_hi = float(row["cell_standardized_lift_ci95_high"])
        out.append(text(408, y + 40, f"{fmt(lift, 3)}×", 25, 700, GREEN))
        out.append(text(408, y + 70, f"[{fmt(lift_lo, 3)}; {fmt(lift_hi, 3)}]", 17, 400, MUTED))

        hit = float(row["cell_standardized_hit_delta_pp"])
        hit_lo = float(row["cell_standardized_hit_delta_pp_ci95_low"])
        hit_hi = float(row["cell_standardized_hit_delta_pp_ci95_high"])
        out.append(text(718, y + 40, f"+{fmt(hit, 1)}", 25, 700, GREEN))
        out.append(text(718, y + 70, f"[{fmt(hit_lo, 1)}; {fmt(hit_hi, 1)}]", 17, 400, MUTED))

        fwd = float(row["cell_standardized_forward_bps_delta"])
        fwd_lo = float(row["cell_standardized_forward_bps_delta_ci95_low"])
        fwd_hi = float(row["cell_standardized_forward_bps_delta_ci95_high"])
        out.append(text(1058, y + 40, f"+{fmt(fwd, 1)}", 25, 700, GREEN))
        out.append(text(1058, y + 70, f"[{fmt(fwd_lo, 1)}; {fmt(fwd_hi, 1)}]", 17, 400, MUTED))

        cadence = float(row["minimum_fold_weekly_1_to_2_fulfillment"]) * 100
        out.append(text(1390, y + 40, f"{fmt(cadence, 1)}%", 25, 700, RED_DARK))
        out.append(text(1390, y + 70, "цель ≥ 90%", 17, 400, MUTED))

    out += [
        rect(70, 815, 710, 52, PALE_GREEN, "none", 12),
        text(95, 849, "QUALITY: 5/5 нижних границ lift CI > 1", 21, 700, GREEN),
        rect(800, 815, 730, 52, PALE_RED, "none", 12),
        text(825, 849, "CADENCE: 0/5 коридоров достигли 90%", 21, 700, RED_DARK),
        text(1530, 892, "KZT — продуктовый MVP; остальные коридоры — проверка переносимости сигнала.", 17, 400, MUTED, "end"),
    ]
    save("02_corridor_matrix.svg", out)


def build_slide_3() -> None:
    lag_rows = [
        row for row in rows(RESULTS / "moex_lag_ladder_h5.csv")
        if row["config_id"] == "hgb_plus_cnyrub_basis"
    ]
    lag_rows.sort(key=lambda row: int(row["moex_calendar_lag_days"]))
    frontier = rows(FRONTIER)

    out = svg_start(
        "Главный риск — timing и cadence данных",
        "Слева чувствительность Brier skill к лагу данных, справа компромисс между lift и недельной частотой сигналов.",
    )
    out += [
        text(72, 78, "Главный риск оказался не в алгоритме, а во времени доступности данных", 37, 700),
        text(72, 120, "Сдвиг на один день меняет вывод; ни один порог пока не даёт одновременно quality и cadence", 23, 400, MUTED),
        rect(70, 165, 700, 620, WHITE, LINE, 24, 1.5),
        text(110, 220, "Brier skill к prior-year, %", 27, 700),
        text(110, 253, "чем выше — тем лучше", 18, 400, MUTED),
    ]

    lx0, lx1, ly0, ly1 = 150, 710, 700, 300
    ymin, ymax = -6.0, 15.0
    for tick in [-5, 0, 5, 10, 15]:
        y = ly0 - (tick - ymin) / (ymax - ymin) * (ly0 - ly1)
        out.append(line(lx0, y, lx1, y, "#E6E6E6", 1.5, "6 6" if tick == 0 else None))
        out.append(text(lx0 - 15, y + 7, f"{tick}%", 17, 400, MUTED, "end"))
    points = []
    for row in lag_rows:
        lag = int(row["moex_calendar_lag_days"])
        value = float(row["brier_skill_vs_prior_year"]) * 100
        x = lx0 + lag / 5 * (lx1 - lx0)
        y = ly0 - (value - ymin) / (ymax - ymin) * (ly0 - ly1)
        points.append((x, y, lag, value))
    out.append('<polyline points="' + " ".join(f"{x},{y}" for x, y, _, _ in points) + f'" fill="none" stroke="{RED}" stroke-width="5" stroke-linejoin="round"/>')
    for x, y, lag, value in points:
        out.append(circle(x, y, 11, RED if lag == 1 else INK))
        out.append(text(x, y - 22, f"{value:+.1f}%".replace(".", ","), 18, 700, RED_DARK if lag == 1 else INK, "middle"))
        out.append(text(x, 738, f"lag {lag}", 17, 600, MUTED, "middle"))
    out += [
        rect(105, 755, 620, 1, LINE, "none", 0),
        text(110, 775, "lag 1 → lag 2: skill +13,2% → −2,7%", 20, 700, RED_DARK),
        rect(810, 165, 720, 620, WHITE, LINE, 24, 1.5),
        text(850, 220, "Quality–cadence frontier", 27, 700),
        text(850, 253, "x: min недель с 1–2 сигналами · y: lift", 18, 400, MUTED),
    ]

    fx0, fx1, fy0, fy1 = 895, 1470, 700, 300
    xmin, xmax, fmin, fmax = 0.72, 0.98, 1.05, 1.45
    x_gate = fx0 + (0.90 - xmin) / (xmax - xmin) * (fx1 - fx0)
    y_gate = fy0 - (1.30 - fmin) / (fmax - fmin) * (fy0 - fy1)
    out.append(rect(x_gate, fy1, fx1 - x_gate, y_gate - fy1, PALE_GREEN, "none", 0))
    out.append(text(fx1 - 12, fy1 + 28, "целевой сектор", 16, 700, GREEN, "end"))
    for tick in [0.75, 0.80, 0.85, 0.90, 0.95]:
        x = fx0 + (tick - xmin) / (xmax - xmin) * (fx1 - fx0)
        out.append(line(x, fy0, x, fy1, "#ECECEC", 1))
        out.append(text(x, 735, f"{tick*100:.0f}%", 17, 400, MUTED, "middle"))
    for tick in [1.1, 1.2, 1.3, 1.4]:
        y = fy0 - (tick - fmin) / (fmax - fmin) * (fy0 - fy1)
        out.append(line(fx0, y, fx1, y, "#ECECEC", 1, "6 6" if tick == 1.3 else None))
        out.append(text(fx0 - 15, y + 7, fmt(tick, 1), 17, 400, MUTED, "end"))
    scatter = []
    for row in frontier:
        cadence = float(row["minimum_fold_corridor_weekly_1_to_2_fulfillment"])
        lift = float(row["cell_standardized_lift"])
        quantile = float(row["quantile"])
        x = fx0 + (cadence - xmin) / (xmax - xmin) * (fx1 - fx0)
        y = fy0 - (lift - fmin) / (fmax - fmin) * (fy0 - fy1)
        scatter.append((x, y, quantile))
    out.append('<polyline points="' + " ".join(f"{x},{y}" for x, y, _ in scatter) + '" fill="none" stroke="#9C9C9C" stroke-width="3"/>')
    for x, y, quantile in scatter:
        out.append(circle(x, y, 10, RED))
        out.append(text(x + 14, y - 12, f"q={quantile:.2f}".replace(".", ","), 15, 600, INK))
    out += [
        line(x_gate, fy0, x_gate, fy1, GREEN, 2, "7 6"),
        line(fx0, y_gate, fx1, y_gate, GREEN, 2, "7 6"),
        rect(840, 755, 660, 1, LINE, "none", 0),
        text(850, 775, "Нет точки с lift ≥ 1,3 и cadence ≥ 90%", 20, 700, RED_DARK),
        rect(70, 810, 1460, 50, PALE_RED, "none", 12),
        text(95, 843, "Вывод: сначала фиксируем point-in-time SLA источников и shadow-режим, затем заявляем production-качество.", 21, 700, INK),
        text(1530, 892, "Lag ladder и frontier — ретроспективные stress tests, не критерий выбора после факта.", 17, 400, MUTED, "end"),
    ]
    save("03_timing_cadence.svg", out)


def build_slide_4() -> None:
    out = svg_start(
        "Новая архитектура решения",
        "Декомпозиция курса через USD-якорь, прогноз остатка и policy layer с abstention.",
    )
    out += [
        text(72, 78, "Задача ближе к pricing engine, чем к обычному FX-прогнозу", 37, 700),
        text(72, 120, "Сначала восстанавливаем справедливый cross-rate, затем прогнозируем только будущие компоненты и residual", 23, 400, MUTED),
    ]

    # Input blocks
    input_blocks = [
        (70, 190, "RUB / USD", "официальный + рынок"),
        (70, 315, "LCY / USD", "KASE / локальный рынок"),
        (70, 440, "CNY / RUB", "spot − fixing dislocation"),
        (70, 565, "Internal offers", "после интеграции"),
    ]
    for x, y, title, subtitle in input_blocks:
        out.append(rect(x, y, 285, 92, PALE, LINE, 18, 1))
        out.append(text(x + 25, y + 38, title, 24, 700))
        out.append(text(x + 25, y + 68, subtitle, 17, 400, MUTED))

    # Main flow
    blocks = [
        (435, 230, 330, 165, "1", "Deterministic reference", ["RUB/LCY ≈", "(RUB/USD) / (LCY/USD)"]),
        (845, 230, 300, 165, "2", "Forecast residual", ["будущие компоненты", "+ остаточная альфа"]),
        (1225, 230, 305, 165, "3", "Calibrated probability", ["P(улучшение > τ", "на горизонте h=5)"]),
        (845, 495, 300, 165, "4", "Utility policy", ["стоимость ошибки", "+ cadence + cooldown"]),
        (1225, 495, 305, 165, "5", "Product action", ["WAIT / SEND", "+ abstain"]),
    ]
    for x, y, width, height, num, title, body in blocks:
        out.append(rect(x, y, width, height, WHITE, LINE, 22, 1.5, 'filter="url(#shadow)"'))
        out.append(circle(x + 34, y + 34, 18, RED, RED, 0))
        out.append(text(x + 34, y + 42, num, 18, 700, WHITE, "middle"))
        out.append(text(x + 64, y + 43, title, 22, 700))
        out.append(multiline(x + 28, y + 92, body, 22, 600, INK, 1.35))

    # Arrows
    for _, y, _, _ in input_blocks[:3]:
        out.append(line(355, y + 46, 430, 295, "#8A8A8A", 2.5, None, 'marker-end="url(#arrow)"'))
    out.append(line(765, 312, 840, 312, "#8A8A8A", 3, None, 'marker-end="url(#arrow)"'))
    out.append(line(1145, 312, 1220, 312, "#8A8A8A", 3, None, 'marker-end="url(#arrow)"'))
    out.append(line(1377, 395, 1148, 528, "#8A8A8A", 3, None, 'marker-end="url(#arrow)"'))
    out.append(line(1145, 578, 1220, 578, "#8A8A8A", 3, None, 'marker-end="url(#arrow)"'))
    out.append(line(355, 611, 840, 578, "#8A8A8A", 2.5, "8 6", 'marker-end="url(#arrow)"'))

    out += [
        rect(70, 725, 450, 110, PALE_GREEN, "none", 18),
        text(100, 772, "5 / 5", 38, 700, GREEN),
        text(215, 766, "коридоров", 22, 700),
        text(215, 796, "арифметически совместимы", 19, 400, MUTED),
        rect(545, 725, 450, 110, PALE_GREEN, "none", 18),
        text(575, 772, "412 / 413", 38, 700, GREEN),
        text(760, 766, "KZT-дней", 22, 700),
        text(760, 796, "strict-prior alignment", 19, 400, MUTED),
        rect(1020, 725, 510, 110, PALE, "none", 18),
        text(1050, 772, "31 / 76", 38, 700, INK),
        text(1205, 766, "source families /", 22, 700),
        text(1205, 796, "hashed artifacts", 19, 400, MUTED),
        text(1530, 884, "Механизм подтверждён арифметически; фактический pricing denominator ещё требует идентификации.", 17, 400, MUTED, "end"),
    ]
    save("04_architecture_finding.svg", out)


def final_data() -> tuple[dict[str, str], dict[str, str], dict[str, str], list[dict[str, str]]]:
    metrics = {
        row["config_id"]: row
        for row in rows(RESULTS / "development_metrics.csv")
        if row["horizon_cbr_rows_pub_proxy"] == "5"
    }
    policy = next(
        row for row in rows(RESULTS / "development_policy_uncertainty_audit_h5.csv")
        if row["config_id"] == "hgb_plus_cnyrub_basis"
    )
    corridors = [
        row for row in rows(RESULTS / "diagnostic_corridor_candidate_uncertainty_h5.csv")
        if row["track"] == "contract_required_market_data"
        and row["config_id"] == "hgb_plus_cnyrub_basis"
    ]
    corridors.sort(key=lambda row: ["AMD", "KGS", "KZT", "TJS", "UZS"].index(row["corridor"]))
    return metrics["hgb_base"], metrics["hgb_plus_cnyrub_basis"], policy, corridors


def build_final_slide_1() -> None:
    base, final, policy, corridors = final_data()
    kzt = next(row for row in corridors if row["corridor"] == "KZT")
    base_lift = float(base["candidate_cell_standardized_lift"])
    final_lift = float(policy["candidate_cell_standardized_lift"])
    relative_gain = (final_lift / base_lift - 1) * 100
    hit_rate = float(final["candidate_hit_rate"]) * 100
    hit_delta = float(policy["candidate_cell_standardized_hit_delta_pp"])
    random_hit = hit_rate - hit_delta

    out = svg_start(
        "Итоговый результат против бейзлайнов кейса",
        "Финальный lift сравнивается со случайным днём, ML на базовых FX-признаках кейса и целевым уровнем 1.3.",
    )
    out += [
        text(72, 78, "Итог: lift 1,435× на 5 коридорах при цели кейса ≥ 1,30×", 39, 700),
        text(72, 120, "Единый rolling OOT протокол · NOW_FAVORABLE · h=5 · 2023–2025 · 663 сигнала", 23, 400, MUTED),
        rect(70, 165, 930, 535, WHITE, LINE, 24, 1.5),
        text(110, 220, "Сравнение с бейзлайнами задания", 27, 700),
        text(110, 253, "Lift к случайному дню: выше — лучше", 18, 400, MUTED),
    ]
    x0, x1, ymin = 420, 940, 0.95
    xmax = 1.62
    for tick in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
        x = x0 + (tick - ymin) / (xmax - ymin) * (x1 - x0)
        out.append(line(x, 285, x, 610, "#ECECEC", 1.2))
        out.append(text(x, 650, fmt(tick, 1), 17, 400, MUTED, "middle"))
    target_x = x0 + (1.3 - ymin) / (xmax - ymin) * (x1 - x0)
    out.append(line(target_x, 278, target_x, 620, GREEN, 3, "8 6"))
    out.append(text(target_x, 274, "цель кейса 1,30×", 18, 700, GREEN, "middle"))

    comparisons = [
        ("Случайный день", 1.0, None, None, "#777777"),
        ("Базовые FX-признаки кейса", base_lift, None, None, INK),
        (
            "Финальный сигнал",
            final_lift,
            float(policy["candidate_cell_standardized_lift_ci95_low"]),
            float(policy["candidate_cell_standardized_lift_ci95_high"]),
            RED,
        ),
    ]
    for i, (label, value, low, high, color) in enumerate(comparisons):
        y = 335 + i * 118
        out.append(text(110, y + 8, label, 23, 700 if i == 2 else 400))
        out.append(line(x0, y, x1, y, "#EFEFEF", 8))
        px = x0 + (value - ymin) / (xmax - ymin) * (x1 - x0)
        if low is not None and high is not None:
            lo = x0 + (low - ymin) / (xmax - ymin) * (x1 - x0)
            hi = x0 + (high - ymin) / (xmax - ymin) * (x1 - x0)
            out.append(line(lo, y, hi, y, RED_DARK, 7))
            out.append(line(lo, y - 12, lo, y + 12, RED_DARK, 3))
            out.append(line(hi, y - 12, hi, y + 12, RED_DARK, 3))
        out.append(circle(px, y, 14 if i == 2 else 10, color))
        out.append(text(px + 22, y - 16 if i == 2 else y + 8, f"{fmt(value, 3)}×", 23, 700 if i == 2 else 500,
                        RED_DARK if i == 2 else INK))
        if i == 2:
            out.append(text(px + 22, y + 34, "95% CI [1,305; 1,574]", 16, 400, MUTED))

    out += [
        rect(1040, 165, 490, 535, PALE_RED, "none", 24),
        text(1085, 220, "RUB → KZT", 24, 700, RED_DARK),
        text(1085, 310, f"{fmt(float(kzt['cell_standardized_lift']), 3)}×", 74, 700, RED_DARK),
        text(1085, 350, "lift итогового сигнала", 23, 700),
        text(1085, 388, "95% CI [1,321; 1,869]", 20, 400, MUTED),
        line(1085, 425, 1485, 425, "#F1C9C5", 2),
        text(1085, 475, "27,0%", 38, 700, MUTED),
        text(1215, 475, "→", 34, 600, RED_DARK),
        text(1280, 475, "42,9%", 38, 700, RED_DARK),
        text(1085, 510, "random hit-rate", 18, 400, MUTED),
        text(1280, 510, "signal hit-rate", 18, 400, MUTED),
        text(1085, 575, "+15,9 п.п.", 34, 700, GREEN),
        text(1085, 612, "абсолютный прирост hit-rate", 20, 500, INK),
        text(1085, 655, "+42,0 б.п. forward-reference", 21, 700, GREEN),
        rect(70, 735, 345, 105, PALE_GREEN, "none", 18),
        text(100, 780, f"+{fmt(relative_gain, 1)}%", 36, 700, GREEN),
        text(100, 815, "lift к FX-бейзлайну кейса", 18, 500),
        rect(440, 735, 345, 105, PALE_GREEN, "none", 18),
        text(470, 780, "−12,8%", 36, 700, GREEN),
        text(470, 815, "Brier score к бейзлайну", 18, 500),
        rect(810, 735, 345, 105, PALE_GREEN, "none", 18),
        text(840, 780, "+47,7 б.п.", 36, 700, GREEN),
        text(840, 815, "forward-reference, 5 коридоров", 18, 500),
        rect(1180, 735, 350, 105, PALE, "none", 18),
        text(1210, 780, "3/3 · 15/15", 36, 700, INK),
        text(1210, 815, "лет · fold×corridor ячеек", 18, 500),
        text(1530, 884, "Ретроспективный post-selection результат; перед production нужен frozen confirmatory test.", 17, 400, MUTED, "end"),
    ]
    save("01_backtest_quality.svg", out)


def build_final_slide_2() -> None:
    _, _, policy, corridors = final_data()
    portfolio = {
        "corridor": "5 коридоров",
        "signal_count": "663",
        "cell_standardized_lift": policy["candidate_cell_standardized_lift"],
        "cell_standardized_lift_ci95_low": policy["candidate_cell_standardized_lift_ci95_low"],
        "cell_standardized_lift_ci95_high": policy["candidate_cell_standardized_lift_ci95_high"],
        "cell_standardized_hit_delta_pp": policy["candidate_cell_standardized_hit_delta_pp"],
        "cell_standardized_forward_bps_delta": policy["candidate_cell_standardized_forward_bps_delta"],
    }
    data = [portfolio] + corridors
    out = svg_start(
        "Итоговый lift по каждому коридору",
        "Forest plot итогового lift с 95% интервалами против случайного дня и целевого уровня кейса 1.3.",
    )
    out += [
        text(72, 78, "Итог переносится: 5/5 коридоров лучше случайного дня", 39, 700),
        text(72, 120, "Точка — lift · линия — 95% block-bootstrap CI · KZT — продуктовый MVP", 23, 400, MUTED),
    ]
    plot_x0, plot_x1 = 420, 1115
    domain_min, domain_max = 0.95, 1.92
    for tick in [1.0, 1.2, 1.3, 1.4, 1.6, 1.8]:
        x = plot_x0 + (tick - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
        out.append(line(x, 190, x, 735, "#E8E8E8", 1.3, "7 6" if tick in [1.0, 1.3] else None))
        out.append(text(x, 772, fmt(tick, 1), 18, 400, MUTED, "middle"))
    x_random = plot_x0 + (1.0 - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
    x_target = plot_x0 + (1.3 - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
    out.append(text(x_random, 178, "random 1,0", 17, 700, MUTED, "middle"))
    out.append(text(x_target, 178, "цель 1,3", 17, 700, GREEN, "middle"))

    for i, row in enumerate(data):
        y = 235 + i * 91
        is_portfolio = i == 0
        is_kzt = row["corridor"] == "KZT"
        if is_portfolio:
            out.append(rect(70, y - 38, 1460, 75, PALE_GREEN, "none", 14))
        elif is_kzt:
            out.append(rect(70, y - 38, 1460, 75, PALE_RED, RED, 14, 2))
        out.append(text(95, y + 8, row["corridor"], 24, 700,
                        RED_DARK if is_kzt else (GREEN if is_portfolio else INK)))
        if is_kzt:
            out.append(rect(180, y - 22, 70, 30, RED, "none", 15))
            out.append(text(215, y, "MVP", 14, 700, WHITE, "middle"))
        out.append(text(310, y + 7, f"N={row['signal_count']}", 18, 500, MUTED, "end"))
        value = float(row["cell_standardized_lift"])
        low = float(row["cell_standardized_lift_ci95_low"])
        high = float(row["cell_standardized_lift_ci95_high"])
        lo = plot_x0 + (low - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
        hi = plot_x0 + (high - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
        px = plot_x0 + (value - domain_min) / (domain_max - domain_min) * (plot_x1 - plot_x0)
        color = RED if is_kzt else (GREEN if is_portfolio else INK)
        out.append(line(lo, y, hi, y, color, 6))
        out.append(line(lo, y - 11, lo, y + 11, color, 2.5))
        out.append(line(hi, y - 11, hi, y + 11, color, 2.5))
        out.append(circle(px, y, 12, color))
        out.append(text(px + 18, y - 14, f"{fmt(value, 3)}×", 18, 700, color))
        hit = float(row["cell_standardized_hit_delta_pp"])
        fwd = float(row["cell_standardized_forward_bps_delta"])
        out.append(text(1190, y + 7, f"+{fmt(hit, 1)} п.п.", 22, 700, GREEN))
        out.append(text(1370, y + 7, f"+{fmt(fwd, 1)} б.п.", 22, 700, GREEN))

    out += [
        text(1190, 178, "Δ hit-rate", 18, 700, MUTED),
        text(1370, 178, "Δ forward-ref", 18, 700, MUTED),
        rect(70, 800, 710, 52, PALE_GREEN, "none", 12),
        text(95, 834, "5/5 нижних границ lift CI > 1,0", 21, 700, GREEN),
        rect(800, 800, 730, 52, PALE_GREEN, "none", 12),
        text(825, 834, "Portfolio CI целиком выше цели 1,3", 21, 700, GREEN),
        text(1530, 890, "Forward-reference — официальный ряд, не фактическая клиентская цена перевода.", 17, 400, MUTED, "end"),
    ]
    save("02_corridor_matrix.svg", out)


def build_final_slide_3() -> None:
    base, _, policy, _ = final_data()
    base_lift = float(base["candidate_cell_standardized_lift"])
    final_lift = float(policy["candidate_cell_standardized_lift"])
    out = svg_start(
        "Что добавили поверх бейзлайна кейса",
        "Путь от официального курса и базовых индикаторов к расширенному набору источников, структурному рыночному фактору и строгой оценке.",
    )
    out += [
        text(72, 78, "Что дало прирост: структура рынка, а не “макро-суп”", 39, 700),
        text(72, 120, "Мы расширили поиск максимально широко, но в финал оставили только доказанно полезное", 23, 400, MUTED),
    ]
    blocks = [
        (70, 195, 310, 390, "Бейзлайн кейса", ["Курс ЦБ", "momentum / returns", "percentile level", "volatility", "random-day baseline"], PALE),
        (430, 195, 335, 390, "Расширили данные", ["31 семейство источников", "MOEX / KASE / NBK", "нефть и commodities", "ставки и risk indices", "инфляция и макро РФ/КЗ"], "#F3F7FC"),
        (815, 195, 335, 390, "Нашли рабочий фактор", ["CNY/RUB spot–fixing", "dislocation", "cross-rate mechanism", "5/5 коридоров", "412/413 KZT-дней"], PALE_RED),
        (1200, 195, 330, 390, "Довели до решения", ["purged rolling OOT", "PIT joins + lag stress", "calibration", "3× cost false positive", "cooldown + abstain"], PALE_GREEN),
    ]
    for x, y, width, height, title_value, items, fill in blocks:
        out.append(rect(x, y, width, height, fill, LINE if fill == PALE else "none", 24, 1.2))
        out.append(text(x + 28, y + 48, title_value, 24, 700,
                        RED_DARK if fill == PALE_RED else (GREEN if fill == PALE_GREEN else INK)))
        for j, item in enumerate(items):
            cy = y + 100 + j * 52
            out.append(circle(x + 35, cy - 6, 5, RED if fill == PALE_RED else INK, "none", 0))
            out.append(text(x + 53, cy, item, 19, 500, INK))
    for x in [395, 780, 1165]:
        out.append(line(x - 5, 390, x + 25, 390, "#8A8A8A", 3, None, 'marker-end="url(#arrow)"'))

    out += [
        rect(70, 625, 1460, 95, INK, "none", 18),
        text(105, 682, f"Lift {fmt(base_lift, 3)}×", 31, 700, WHITE),
        text(360, 682, "→", 32, 600, "#BBBBBB"),
        text(425, 682, f"{fmt(final_lift, 3)}×", 41, 700, "#7DE2AA"),
        text(645, 682, "+31,8% к бейзлайну признаков кейса", 25, 700, WHITE),
        text(1240, 682, "Brier −12,8%", 25, 700, WHITE),
        rect(70, 755, 710, 80, PALE_GREEN, "none", 16),
        text(100, 803, "Оставили", 22, 700, GREEN),
        text(225, 803, "CNY-dislocation: стабильный OOT-прирост", 21, 600, INK),
        rect(800, 755, 730, 80, PALE, "none", 16),
        text(830, 803, "Отсеяли", 22, 700, MUTED),
        text(955, 803, "широкие macro/public stacks: качества не добавили", 21, 600, INK),
        text(1530, 887, "37 h=5 model/feature сравнений; множественные проверки учтены Holm correction.", 17, 400, MUTED, "end"),
    ]
    save("03_timing_cadence.svg", out)


def build_final_slide_4() -> None:
    _, _, policy, corridors = final_data()
    min_cadence = min(float(row["minimum_fold_weekly_1_to_2_fulfillment"]) for row in corridors) * 100
    max_cadence = max(float(row["minimum_fold_weekly_1_to_2_fulfillment"]) for row in corridors) * 100
    checks = [
        ("Lift над случайным днём", "≥ 1,30×", "1,435× [1,305; 1,574]", "PASS", GREEN, PALE_GREEN),
        ("Точность сигнала", "Δ hit-rate > 0", "+13,7 п.п. [+9,6; +17,7]", "PASS", GREEN, PALE_GREEN),
        ("Выгода момента", "Δ reference bps > 0", "+47,7 б.п. [+32,8; +64,0]", "PASS", GREEN, PALE_GREEN),
        ("Устойчивость", "коридоры + OOT окна", "5/5 · 3/3 года · 15/15 ячеек", "PASS", GREEN, PALE_GREEN),
        ("Частота коммуникаций", "1–2/нед. в ≥90% недель", f"{fmt(min_cadence, 1)}–{fmt(max_cadence, 1)}%", "ДОРАБОТАТЬ", RED_DARK, PALE_RED),
        ("Point-in-time готовность", "данные доступны на T", "PIT-логика OK; source SLA не frozen", "ПОДТВЕРДИТЬ", AMBER, "#FFF7E8"),
    ]
    out = svg_start(
        "Scorecard по критериям кейса",
        "Четыре критерия качества пройдены ретроспективно; cadence и историческое подтверждение времени публикации остаются до production.",
    )
    out += [
        text(72, 78, "По качеству цель кейса превышена; оставшийся разрыв — операционный", 39, 700),
        text(72, 120, "Scorecard строго по формулировкам задания — без подмены offline-метрик клиентской экономикой", 23, 400, MUTED),
        rect(70, 155, 1460, 64, INK, "none", 16),
        text(95, 196, "Критерий кейса", 20, 700, WHITE),
        text(520, 196, "Требование", 20, 700, WHITE),
        text(850, 196, "Итог", 20, 700, WHITE),
        text(1400, 196, "Статус", 20, 700, WHITE, "middle"),
    ]
    for i, (criterion, target, result, status, color, fill) in enumerate(checks):
        y = 235 + i * 91
        row_fill = WHITE if i % 2 == 0 else PALE
        out.append(rect(70, y, 1460, 74, row_fill, "none", 12))
        out.append(text(95, y + 46, criterion, 22, 700))
        out.append(text(520, y + 46, target, 20, 500, MUTED))
        out.append(text(850, y + 46, result, 21, 700, color if status != "PASS" else INK))
        out.append(rect(1300, y + 17, 200, 40, fill, "none", 20))
        out.append(text(1400, y + 44, status, 16, 700, color, "middle"))
    out += [
        rect(70, 800, 1460, 56, PALE_GREEN, "none", 14),
        text(95, 837, "Финальный вывод: исследовательский quality gate пройден; следующий шаг — frozen shadow/confirmatory run.", 22, 700, INK),
        text(1530, 890, "Статусы относятся к retrospective research; production GO не заявляется.", 17, 400, MUTED, "end"),
    ]
    save("04_architecture_finding.svg", out)


def main() -> None:
    build_final_slide_1()
    build_final_slide_2()
    build_final_slide_3()
    build_final_slide_4()


if __name__ == "__main__":
    main()
