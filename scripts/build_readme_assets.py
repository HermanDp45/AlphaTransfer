#!/usr/bin/env python3
"""Build and verify the public AlphaTransfer model-report assets.

The script deliberately reads only sealed/canonical CSV and JSON artifacts.  It
does not retrain models or recompute statistical intervals.  SVG is the source
visual format; a matching PNG is drawn with Pillow so the report works in
renderers that do not display SVG.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import html
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "model-metrics"
H3_DIR = ROOT / "final_solution" / "tabm_h3"
H5_DIR = ROOT / "research_v4" / "h5_fullhistory"
H5_BASELINE_DIR = ROOT / "research_v4" / "robust_selection"

W, H = 1600, 900
RED = "#EF3124"
RED_DARK = "#C7231A"
BLUE = "#3567B7"
INK = "#171717"
MUTED = "#676767"
LINE = "#D9D9D9"
PALE = "#F6F6F6"
PALE_RED = "#FFF0EF"
PALE_BLUE = "#EEF4FD"
PALE_GREEN = "#ECF7F0"
GREEN = "#1F8A55"
AMBER = "#B56A00"
WHITE = "#FFFFFF"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class Profile:
    model: str
    label: str
    config_id: str
    horizon: int
    policy: str
    source: str
    aggregate: dict[str, str]
    annual: tuple[dict[str, str], ...]


def read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        source_context = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    else:
        source_context = path.open(encoding="utf-8", newline="")
    with source_context as source:
        return list(csv.DictReader(source))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def one(rows: Iterable[dict[str, str]], **expected: object) -> dict[str, str]:
    matches = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {expected}, found {len(matches)}")
    return matches[0]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def validate_and_load() -> tuple[Profile, Profile]:
    bundle = read_json(H3_DIR / "bundle.json")
    h3_selection = read_json(H3_DIR / "evaluation" / "selection.json")
    if bundle.get("profile_id") != "tabm_kzt_h3_expanding2010":
        raise RuntimeError("H3 bundle is no longer the selected expanding-2010 profile")
    chosen_h3 = h3_selection.get("chosen", {})
    if (chosen_h3.get("config_id"), chosen_h3.get("policy")) != (
        "tabm_kzt_fullhistory", "rank80"
    ):
        raise RuntimeError("H3 selection is no longer full-history rank80")

    h5_selection = read_json(H5_DIR / "selection.json")
    if (
        h5_selection.get("selected_config_id") != "tabm_kzt_h5_fullhistory"
        or h5_selection.get("selected_recipe") != "full_history"
        or h5_selection.get("policy") != "rare65"
        or h5_selection.get("horizon") != 5
        or h5_selection.get("selection_status") != "qualified"
    ):
        raise RuntimeError("H5 selection is no longer the qualified full-history rare65 profile")
    h5_receipts = read_json(H5_DIR / "receipts.json")
    if len(h5_receipts) != 3 or any(
        not str(item["full_history"].get("train_start", "")).startswith("2010-01-01")
        for item in h5_receipts
    ):
        raise RuntimeError("H5 annual receipts do not prove expanding history from 2010")

    h3_summary = read_csv(H3_DIR / "evaluation" / "summary.csv")
    h3_year = read_csv(H3_DIR / "evaluation" / "by_year.csv")
    h5_summary = read_csv(H5_DIR / "summary.csv")
    h5_year = read_csv(H5_DIR / "by_year.csv")

    h3_agg = one(
        h3_summary, config_id="tabm_kzt_fullhistory", policy="rank80", period="2024-2026"
    )
    h3_annual = tuple(
        one(h3_year, config_id="tabm_kzt_fullhistory", policy="rank80", year=year)
        for year in (2024, 2025, 2026)
    )
    h5_agg = one(
        h5_summary,
        config_id="tabm_kzt_h5_fullhistory",
        history_mode="full_history",
        train_horizon=5,
        evaluation_horizon=5,
        policy="rare65",
        evaluation_scope="KZT",
        period="2024-2026",
    )
    h5_annual = tuple(
        one(
            h5_year,
            config_id="tabm_kzt_h5_fullhistory",
            history_mode="full_history",
            train_horizon=5,
            evaluation_horizon=5,
            policy="rare65",
            evaluation_scope="KZT",
            year=year,
        )
        for year in (2024, 2025, 2026)
    )

    expected = {
        "h3": (h3_agg, 1.5042333624865076, 0.4900662251655629, 151),
        "h5": (h5_agg, 1.9858935404634515, 0.5280898876404494, 89),
    }
    for name, (row, lift, hit_rate, signals) in expected.items():
        if abs(f(row, "lift") - lift) > 1e-12:
            raise RuntimeError(f"{name} aggregate lift changed; review public claims")
        if abs(f(row, "hit_rate") - hit_rate) > 1e-12:
            raise RuntimeError(f"{name} aggregate hit rate changed; review public claims")
        if int(row["signals"]) != signals:
            raise RuntimeError(f"{name} aggregate signal count changed; review public claims")

    return (
        Profile(
            "h3", "H3 · регулярный сигнал", "tabm_kzt_fullhistory", 3, "rank80",
            "final_solution/tabm_h3/evaluation", h3_agg, h3_annual,
        ),
        Profile(
            "h5", "H5 · редкий режим", "tabm_kzt_h5_fullhistory", 5, "rare65",
            "research_v4/h5_fullhistory", h5_agg, h5_annual,
        ),
    )


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{fmt(100 * value, digits)}%"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class Canvas:
    def __init__(self, title: str, description: str) -> None:
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{esc(title)}</title>',
            f'<desc id="desc">{esc(description)}</desc>',
            f'<rect width="{W}" height="{H}" fill="{WHITE}"/>',
        ]
        self.image = Image.new("RGB", (W, H), WHITE)
        self.draw = ImageDraw.Draw(self.image)

    @staticmethod
    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        path = FONT_BOLD if bold else FONT_REGULAR
        if not path.exists():
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(str(path), size=size)

    def rect(self, xy: tuple[int, int, int, int], fill: str, *, radius: int = 18,
             stroke: str | None = None, width: int = 1) -> None:
        x0, y0, x1, y1 = xy
        stroke_value = stroke or "none"
        self.svg.append(
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke_value}" stroke-width="{width}"/>'
        )
        self.draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=stroke, width=width)

    def line(self, xy: tuple[int, int, int, int], fill: str = LINE, width: int = 2,
             dash: bool = False) -> None:
        x0, y0, x1, y1 = xy
        dash_attr = ' stroke-dasharray="10 8"' if dash else ""
        self.svg.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{fill}" '
            f'stroke-width="{width}"{dash_attr}/>'
        )
        if not dash:
            self.draw.line(xy, fill=fill, width=width)
        else:
            length = max(abs(x1 - x0), abs(y1 - y0))
            for start in range(0, length, 18):
                end = min(start + 10, length)
                if x0 == x1:
                    self.draw.line((x0, y0 + start, x1, y0 + end), fill=fill, width=width)
                else:
                    self.draw.line((x0 + start, y0, x0 + end, y1), fill=fill, width=width)

    def circle(self, x: int, y: int, radius: int, fill: str, stroke: str = WHITE) -> None:
        self.svg.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
        )
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=stroke, width=3)

    def polyline(self, points: list[tuple[int, int]], fill: str, width: int = 4) -> None:
        encoded = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polyline points="{encoded}" fill="none" stroke="{fill}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        self.draw.line(points, fill=fill, width=width, joint="curve")

    def diamond(self, x: int, y: int, radius: int, fill: str, stroke: str = WHITE) -> None:
        points = [(x, y-radius), (x+radius, y), (x, y+radius), (x-radius, y)]
        encoded = " ".join(f"{px},{py}" for px, py in points)
        self.svg.append(
            f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
        )
        self.draw.polygon(points, fill=fill, outline=stroke)

    def arrow(self, xy: tuple[int, int, int, int], fill: str = INK, width: int = 4) -> None:
        x0, y0, x1, y1 = xy
        self.line((x0, y0, x1 - 13, y1), fill=fill, width=width)
        points = [(x1, y1), (x1 - 18, y1 - 11), (x1 - 18, y1 + 11)]
        encoded = " ".join(f"{px},{py}" for px, py in points)
        self.svg.append(f'<polygon points="{encoded}" fill="{fill}"/>')
        self.draw.polygon(points, fill=fill)

    def text(self, x: int, y: int, value: object, size: int = 24, *, bold: bool = False,
             fill: str = INK, anchor: str = "la") -> None:
        weight = 700 if bold else 400
        svg_anchor = {"la": "start", "ma": "middle", "ra": "end", "mm": "middle"}.get(anchor, "start")
        self.svg.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{svg_anchor}">{esc(value)}</text>'
        )
        pil_anchor = anchor if anchor in {"la", "ma", "ra", "mm"} else "la"
        self.draw.text((x, y), str(value), font=self.font(size, bold), fill=fill, anchor=pil_anchor)

    def save(self, out: Path, name: str) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.svg").write_text("\n".join(self.svg + ["</svg>"]) + "\n", encoding="utf-8")
        self.image.save(out / f"{name}.png", format="PNG", optimize=False, compress_level=9)


def header(c: Canvas, title: str, subtitle: str) -> None:
    c.text(72, 78, title, 39, bold=True)
    c.text(72, 120, subtitle, 22, fill=MUTED)


def build_scorecard(out: Path, profiles: tuple[Profile, Profile]) -> None:
    c = Canvas("Два горизонта AlphaTransfer", "Scorecard выбранных H3 и H5 на rolling OOT 2024–2026.")
    header(c, "Два горизонта — два режима коммуникации", "RUB → KZT · rolling OOT 2024–2026 · только доступная на дату решения информация")
    for i, profile in enumerate(profiles):
        row = profile.aggregate
        x0 = 70 + i * 765
        fill = PALE_BLUE if profile.model == "h3" else PALE_GREEN
        accent = BLUE if profile.model == "h3" else GREEN
        c.rect((x0, 165, x0 + 730, 700), fill, radius=26)
        c.text(x0 + 38, 218, profile.label, 27, bold=True, fill=accent)
        policy_label = "для пользователей с редкими переводами" if profile.model == "h3" else "0,6–0,7 сигнала / неделю"
        c.text(x0 + 38, 268, f"Горизонт {profile.horizon} сессии · {policy_label}", 20, fill=MUTED)
        c.text(x0 + 38, 340, f"{fmt(f(row, 'lift'), 3)}×", 74, bold=True, fill=accent)
        c.text(x0 + 38, 428, "lift к случайному дню", 22, bold=True)
        metrics = [
            ("hit rate", pct(f(row, "hit_rate"))),
            ("недель с 1–2 сигналами", pct(f(row, "week_coverage"))),
            ("сигналов / неделю", fmt(f(row, "signals_per_corridor_week"), 2)),
            ("выгода момента", f"+{fmt(f(row, 'forward_delta_bps'), 1)} б.п."),
        ]
        for j, (label, value) in enumerate(metrics):
            xx = x0 + 38 + (j % 2) * 340
            yy = 520 + (j // 2) * 105
            c.text(xx, yy, value, 35, bold=True, fill=INK)
            c.text(xx, yy + 34, label, 18, fill=MUTED)
        c.line((x0 + 38, 470, x0 + 692, 470), fill="#DCCFCD" if i == 0 else "#CCDED3")
    c.rect((70, 735, 1530, 835), INK, radius=20)
    c.text(105, 787, "H3", 25, bold=True, fill=WHITE)
    c.text(165, 787, "регулярный сигнал для редких переводов", 23, fill=WHITE)
    c.text(650, 787, "H5", 25, bold=True, fill="#7DE2AA")
    c.text(710, 787, "реже, избирательнее, горизонт шире", 23, fill=WHITE)
    c.text(1488, 787, "151 + 89 ретроспективных сигналов", 19, fill="#BBBBBB", anchor="ra")
    c.text(1530, 882, "Разные горизонты и base rate: hit rate между H3 и H5 напрямую не сравнивается.", 16, fill=MUTED, anchor="ra")
    c.save(out, "01_final_models")


def build_stability(out: Path, profiles: tuple[Profile, Profile]) -> None:
    c = Canvas("Устойчивость H3 и H5", "Годовые lift и weekly coverage выбранных моделей за 2024–2026.")
    header(c, "Устойчивость важнее одного удачного 2026 года", "Все 6 годовых ячеек: lift > 1,30×; H5 держит редкую частоту в годовом guard-диапазоне")
    years = (2024, 2025, 2026)
    for panel, metric, title, lo, hi, targets in [
        (0, "lift", "Lift к случайному дню", 1.2, 1.7, [(1.3, GREEN, "цель 1,30×")]),
        (1, "signals_per_corridor_week", "Сигналов в неделю", 0.4, 1.5, [(0.45, GREEN, "H5 guard 0,45"), (0.85, AMBER, "H5 guard 0,85")]),
    ]:
        x0 = 70 + panel * 765
        c.rect((x0, 165, x0 + 730, 760), WHITE, stroke=LINE, radius=24)
        c.text(x0 + 35, 215, title, 27, bold=True)
        plot_left, plot_right = x0 + 130, x0 + 680
        plot_top, plot_bottom = 285, 675
        for target, color, label in targets:
            yy = int(plot_bottom - (target - lo) / (hi - lo) * (plot_bottom - plot_top))
            c.line((plot_left, yy, plot_right, yy), fill=color, width=2, dash=True)
            c.text(plot_right, yy - 10, label, 16, bold=True, fill=color, anchor="ra")
        for idx, year in enumerate(years):
            xx = plot_left + 75 + idx * 190
            c.text(xx, 720, year, 19, fill=MUTED, anchor="ma")
            for pidx, profile in enumerate(profiles):
                value = f(profile.annual[idx], metric)
                yy = int(plot_bottom - (value - lo) / (hi - lo) * (plot_bottom - plot_top))
                color = BLUE if profile.model == "h3" else GREEN
                offset = -18 if pidx == 0 else 18
                c.circle(xx + offset, yy, 11, color)
                label = f"{fmt(value, 3)}×" if metric == "lift" else fmt(value, 2)
                c.text(xx + offset, yy - 22, label, 16, bold=True, fill=color, anchor="ma")
        c.text(x0 + 35, 795, "● H3 · регулярный", 18, bold=True, fill=BLUE)
        c.text(x0 + 355, 795, "● H5 · редкий", 18, bold=True, fill=GREEN)
    c.text(1530, 882, "2026 — доступная часть года; результаты ретроспективные и post-selection.", 16, fill=MUTED, anchor="ra")
    c.save(out, "02_annual_stability")


def build_frontier(out: Path) -> None:
    h3_rows = read_csv(H3_DIR / "evaluation" / "summary.csv")
    h5_rows = read_csv(H5_BASELINE_DIR / "summary.csv")
    h5_selected = read_csv(H5_DIR / "summary.csv")
    candidates: list[tuple[str, str, float, float, bool]] = []
    for row in h3_rows:
        if row["config_id"] == "tabm_kzt_fullhistory" and row["policy"] in {"strict05", "cadence80", "cadence85", "cadence90", "rank80", "rank90"}:
            candidates.append(("H3", row["policy"], f(row, "signals_per_corridor_week"), f(row, "lift"), row["policy"] == "rank80"))
    for row in h5_rows:
        if row["config_id"] == "tabm_kzt" and row["train_horizon"] == "5" and row["evaluation_scope"] == "KZT" and row["policy"] in {"strict05", "cadence80", "cadence85", "cadence90", "rank80", "rank90"}:
            candidates.append(("H5", row["policy"], f(row, "signals_per_corridor_week"), f(row, "lift"), False))
    rare = one(
        h5_selected, config_id="tabm_kzt_h5_fullhistory", history_mode="full_history",
        policy="rare65", period="2024-2026",
    )
    candidates.append(("H5", "редкий режим", f(rare, "signals_per_corridor_week"), f(rare, "lift"), True))
    c = Canvas("Quality-cadence frontier", "Компромисс lift и числа сигналов в неделю для политик H3 и H5.")
    header(c, "Частота выбирается под пользовательский сценарий", "H3 поддерживает регулярный контакт; H5 даёт более редкий сигнал на расширенном горизонте")
    c.rect((70, 165, 1530, 805), WHITE, stroke=LINE, radius=24)
    left, right, top, bottom = 180, 1460, 245, 710
    xmin, xmax, ymin, ymax = 0.2, 1.5, 1.2, 2.1
    for target, label in [(0.6, "H5 0,60"), (0.7, "H5 0,70")]:
        xx = int(left + (target-xmin)/(xmax-xmin)*(right-left))
        c.line((xx, top, xx, bottom), fill=GREEN if target == 0.6 else AMBER, width=2, dash=True)
        c.text(xx, 226, label, 17, bold=True, fill=GREEN if target == 0.6 else AMBER, anchor="ma")
    yy = int(bottom - (1.3-ymin)/(ymax-ymin)*(bottom-top))
    c.line((left, yy, right, yy), fill=GREEN, width=2, dash=True)
    c.text(left, yy - 12, "lift 1,30×", 17, bold=True, fill=GREEN)
    for model, policy, coverage, lift, selected in candidates:
        xx = int(left + (coverage-xmin)/(xmax-xmin)*(right-left))
        yy = int(bottom - (lift-ymin)/(ymax-ymin)*(bottom-top))
        color = BLUE if model == "H3" else GREEN
        is_final = selected and policy in {"rank80", "редкий режим"}
        c.circle(xx, yy, 13 if is_final else 8, color if is_final else "#8F8F8F")
        if is_final or policy == "strict05":
            c.text(xx + 14, yy - 14, f"{model} {policy}", 16, bold=is_final, fill=color if is_final else MUTED)
    c.text(820, 765, "Сигналов в неделю →", 20, bold=True, anchor="ma")
    c.text(95, 505, "Lift", 20, bold=True, fill=MUTED)
    c.text(95, 532, "↑", 25, bold=True, fill=MUTED)
    c.text(1530, 865, "Выбранные точки выделены цветом; серые — альтернативные causal policies на тех же прогнозах.", 16, fill=MUTED, anchor="ra")
    c.save(out, "03_quality_cadence_frontier")


def build_utility(out: Path, profiles: tuple[Profile, Profile]) -> None:
    c = Canvas("Utility и regret", "Годовые forward delta и regret выбранных H3/H5.")
    header(c, "Сигнал должен давать выгоду, а не только попадать в класс", "Forward delta — преимущество к среднему дню; regret — упущенная выгода до лучшей точки окна")
    for i, profile in enumerate(profiles):
        x0 = 70 + i * 765
        c.rect((x0, 165, x0 + 730, 790), PALE_BLUE if i == 0 else PALE_GREEN, radius=24)
        color = BLUE if i == 0 else GREEN
        c.text(x0 + 35, 220, profile.label, 27, bold=True, fill=color)
        scale = 145.0
        for j, row in enumerate(profile.annual):
            y = 315 + j * 135
            gain = f(row, "forward_delta_bps")
            regret = f(row, "regret_bps")
            c.text(x0 + 35, y, row["year"], 22, bold=True)
            c.rect((x0 + 125, y - 30, x0 + 125 + int(480 * gain / scale), y), color, radius=8)
            c.text(x0 + 620, y - 8, f"+{fmt(gain, 1)} б.п.", 19, bold=True, fill=color, anchor="ra")
            c.rect((x0 + 125, y + 18, x0 + 125 + int(480 * regret / scale), y + 48), "#9B9B9B", radius=8)
            c.text(x0 + 620, y + 42, f"regret {fmt(regret, 1)}", 17, fill=MUTED, anchor="ra")
        agg = profile.aggregate
        c.line((x0 + 35, 705, x0 + 695, 705), fill="#D4D4D4")
        c.text(x0 + 35, 752, "2024–2026", 18, bold=True)
        c.text(x0 + 200, 752, f"+{fmt(f(agg, 'forward_delta_bps'), 1)} б.п.", 27, bold=True, fill=color)
        c.text(x0 + 465, 752, f"regret {fmt(f(agg, 'regret_bps'), 1)}", 19, fill=MUTED)
    c.text(1530, 875, "Forward-reference основан на официальном ряду, не на фактической клиентской цене перевода.", 16, fill=MUTED, anchor="ra")
    c.save(out, "04_utility_regret")


def build_h3_history(out: Path) -> None:
    rows = read_csv(H3_DIR / "evaluation" / "by_year.csv")
    c = Canvas("Польза полной истории H3", "Сравнение H3 rank80 на 120 месяцах и полной истории с 2010 года.")
    header(c, "История с 2010 года закрыла провал H3 в 2026", "Цена устойчивости: ниже lift в 2025 и немного реже коммуникации")
    c.rect((70, 165, 1530, 770), WHITE, stroke=LINE, radius=24)
    for i, year in enumerate((2024, 2025, 2026)):
        x0 = 105 + i * 480
        short = one(rows, config_id="tabm_kzt_120m", policy="rank80", year=year)
        long = one(rows, config_id="tabm_kzt_fullhistory", policy="rank80", year=year)
        c.text(x0, 230, year, 27, bold=True)
        c.text(x0, 292, "120 месяцев", 18, fill=MUTED)
        c.text(x0 + 235, 292, "с 2010", 18, bold=True, fill=BLUE)
        c.text(x0, 365, f"{fmt(f(short, 'lift'), 3)}×", 42, bold=True)
        c.text(x0 + 235, 365, f"{fmt(f(long, 'lift'), 3)}×", 42, bold=True, fill=BLUE)
        c.text(x0, 410, "lift", 17, fill=MUTED)
        c.text(x0, 495, pct(f(short, "week_coverage")), 31, bold=True)
        c.text(x0 + 235, 495, pct(f(long, "week_coverage")), 31, bold=True, fill=BLUE)
        c.text(x0, 535, "coverage", 17, fill=MUTED)
        c.text(x0, 620, f"+{fmt(f(short, 'forward_delta_bps'), 1)}", 31, bold=True)
        c.text(x0 + 235, 620, f"+{fmt(f(long, 'forward_delta_bps'), 1)}", 31, bold=True, fill=BLUE)
        c.text(x0, 660, "forward delta, б.п.", 17, fill=MUTED)
        if i < 2:
            c.line((x0 + 425, 220, x0 + 425, 700), fill=LINE)
    c.rect((100, 710, 1500, 755), PALE_GREEN, radius=12)
    c.text(125, 740, "Выбор сделан по прохождению годовых gates, а не по максимальному среднему lift.", 20, bold=True, fill=GREEN)
    c.text(1530, 875, "Парный bootstrap разницы включает ноль: статистически гарантированное превосходство над 120m не заявляется.", 16, fill=MUTED, anchor="ra")
    c.save(out, "05_h3_history")


def build_closing(out: Path) -> None:
    rows = read_csv(H3_DIR / "evaluation" / "closing_by_year.csv")
    predictions = read_csv(ROOT / "research_v4" / "h3_finalization" / "closing_predictions.csv.gz")
    annotations = [row for row in predictions if row["closing_annotation"].lower() == "true"]
    now_signals = sum(row["candidate_signal"].lower() == "true" for row in predictions)
    y = [float(row["closing_target"]) for row in predictions]
    probability = [float(row["closing_probability"]) for row in predictions]
    positives = [score for target, score in zip(y, probability) if target == 1.0]
    negatives = [score for target, score in zip(y, probability) if target == 0.0]
    auc = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))
    brier = sum((target - score) ** 2 for target, score in zip(y, probability)) / len(y)
    hit_rate = sum(float(row["closing_target"]) for row in annotations) / len(annotations)
    base_hit = sum(y) / len(y)
    lift = hit_rate / base_hit
    endpoint = sum(float(row["closing_endpoint_bps"]) for row in annotations) / len(annotations)
    if (len(annotations), now_signals) != (61, 151):
        raise RuntimeError("CLOSING aggregate cohort changed; review public metrics")
    c = Canvas("Сигнал закрытия окна", "Годовые метрики дополнительной аннотации поверх сигнала выгодного момента.")
    header(c, "Второй слой показывает, когда окно может закрыться", "Усиливает часть NOW-сигналов и не увеличивает число контактов")
    aggregate = [
        (f"{len(annotations)} / {now_signals}", "усиленных NOW"),
        (pct(hit_rate), "hit rate"),
        (f"{fmt(lift, 3)}×", "lift"),
        (f"+{fmt(endpoint, 1)} б.п.", "endpoint delta"),
        (fmt(brier, 4), "Brier · все даты"),
        (fmt(auc, 3), "AUC · все даты"),
    ]
    for idx, (value, label) in enumerate(aggregate):
        x0 = 70 + idx * 245
        c.rect((x0, 160, x0 + 220, 285), PALE_BLUE if idx < 4 else PALE, radius=18)
        c.text(x0 + 22, 215, value, 29, bold=True, fill=BLUE if idx < 4 else INK)
        c.text(x0 + 22, 255, label, 16, fill=MUTED)
    c.rect((70, 325, 1530, 800), WHITE, stroke=LINE, radius=24)
    headers = ["Год", "NOW сигналов", "CLOSING", "Hit rate", "Lift", "Endpoint delta"]
    xs = [105, 305, 560, 810, 1040, 1280]
    c.rect((90, 350, 1510, 415), INK, radius=12)
    for x, value in zip(xs, headers):
        c.text(x, 391, value, 18, bold=True, fill=WHITE)
    for idx, row in enumerate(rows):
        row_y = 475 + idx * 110
        passed = f(row, "lift") >= 1.3
        fill = WHITE if idx % 2 == 0 else PALE
        c.rect((90, row_y - 43, 1510, row_y + 33), fill, radius=10)
        values = [
            row["year"], row["now_signals"], row["closing_annotations"],
            pct(f(row, "hit_rate")), f"{fmt(f(row, 'lift'), 3)}×",
            f"+{fmt(f(row, 'endpoint_delta_bps'), 1)} б.п.",
        ]
        for x, value in zip(xs, values):
            c.text(x, row_y, value, 23, bold=x in {xs[0], xs[4]}, fill=INK if passed or x != xs[4] else RED)
        c.rect((1390, row_y - 29, 1480, row_y + 12), PALE_GREEN if passed else PALE_RED, radius=20)
        c.text(1435, row_y - 2, "PASS" if passed else "HOLD", 14, bold=True, fill=GREEN if passed else RED, anchor="ma")
    c.text(1530, 860, "Lift и endpoint считаются на 61 аннотации; Brier/AUC — на всех 647 OOT-днях. В 2025 годовой lift ниже 1,30×.", 16, fill=MUTED, anchor="ra")
    c.save(out, "06_closing_diagnostic")


def build_signal_timeline(out: Path) -> None:
    rows = read_csv(ROOT / "research_v4" / "h3_finalization" / "closing_predictions.csv.gz")
    selected = [
        row for row in rows
        if dt.date(2026, 6, 1) <= dt.date.fromisoformat(row["date"]) <= dt.date(2026, 7, 31)
    ]
    if len(selected) != 43:
        raise RuntimeError(f"unexpected number of rows in CLOSING example: {len(selected)}")

    c = Canvas(
        "Как работают два слоя сигнала",
        "Фактический официальный курс KZT за RUB и ретроспективные NOW/CLOSING сигналы за июнь–июль 2026.",
    )
    header(c, "Один контакт — два уровня уверенности", "Реальный официальный курс · июнь–июль 2026 · выше означает больше KZT за 1 RUB")
    c.rect((70, 165, 1530, 745), WHITE, stroke=LINE, radius=24)
    left, right, top, bottom = 135, 1480, 235, 610
    values = [1.0 / float(row["rub_per_unit"]) for row in selected]
    ymin, ymax = min(values) - 0.05, max(values) + 0.05
    points: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        x = int(left + idx * (right-left) / (len(values)-1))
        y = int(bottom - (value-ymin) / (ymax-ymin) * (bottom-top))
        points.append((x, y))
    for tick in range(4):
        value = ymin + tick * (ymax-ymin) / 3
        y = int(bottom - (value-ymin) / (ymax-ymin) * (bottom-top))
        c.line((left, y, right, y), fill="#ECECEC", width=1)
        c.text(left-18, y+6, fmt(value, 2), 16, fill=MUTED, anchor="ra")
    c.polyline(points, INK, width=5)
    now_count = closing_count = 0
    for idx, row in enumerate(selected):
        x, y = points[idx]
        if row["candidate_signal"].lower() == "true":
            now_count += 1
            c.circle(x, y, 10, BLUE)
        if row["closing_annotation"].lower() == "true":
            closing_count += 1
            c.diamond(x, y-23, 11, INK)
    if (now_count, closing_count) != (12, 8):
        raise RuntimeError(f"unexpected signal counts in CLOSING example: {(now_count, closing_count)}")
    for label, date_value in [("02 июня", dt.date(2026, 6, 2)), ("01 июля", dt.date(2026, 7, 1)), ("31 июля", dt.date(2026, 7, 31))]:
        idx = next(i for i, row in enumerate(selected) if dt.date.fromisoformat(row["date"]) == date_value)
        c.text(points[idx][0], 650, label, 17, fill=MUTED, anchor="ma")
    c.circle(165, 696, 9, BLUE)
    c.text(185, 703, "Выгодно сейчас", 18, bold=True)
    c.diamond(405, 696, 10, INK)
    c.text(425, 703, "Выгодно сейчас · окно может закрыться", 18, bold=True)
    c.rect((70, 775, 1530, 845), PALE_GREEN, radius=16)
    c.text(100, 818, "За 2024–2026 второй слой усилил 61 из 151 сигналов (40%); направление подтвердилось в 67% случаев.", 22, bold=True, fill=GREEN)
    c.text(1530, 880, "Исторический пример работы правила, без обещания будущей доходности.", 16, fill=MUTED, anchor="ra")
    c.save(out, "07_signal_timeline")


def build_trigger_map(out: Path) -> None:
    h3_features = set(read_json(H3_DIR / "bundle.json")["features"])
    h5_features = set(read_json(ROOT / "final_solution" / "tabm_h5" / "feature_contract.json")["features"])
    if h3_features != h5_features:
        raise RuntimeError("H3/H5 feature contracts diverged; trigger map needs review")
    groups = [
        (
            "Уровень и импульс курса",
            {*(f"ret{n}" for n in (1, 3, 5, 10, 20, 60)), *(f"pr{n}" for n in (20, 60, 120, 252))},
            "CBR · доходности 1–60 сессий · перцентили 20–252",
            ("Где сегодняшний RUB/KZT находится", "в недавнем диапазоне и насколько", "устойчиво меняется направление."),
            PALE_BLUE,
            BLUE,
        ),
        (
            "Волатильность и режим",
            {*(f"vol{n}" for n in (5, 20, 60)), "volratio"},
            "CBR · волатильность 5 / 20 / 60 · short/long ratio",
            ("Отличаем спокойное окно от резкого", "или нетипичного движения: одинаковый", "уровень курса несёт разный риск."),
            PALE,
            INK,
        ),
        (
            "Внешний валютный контекст",
            {
                "moex_cny_close_minus_fixing_same_session", "oxr_log_basis",
                "oxr_basis_chg1", "oxr_basis_chg5", "oxr_basis_z20",
                "oxr_available", "oxr_age_days",
            },
            "MOEX CNY · OXR 2010+ · basis, change, z-score, age",
            ("Ловим расхождение официального курса", "с внешними кросс-курсами и проверяем,", "не устарела ли рыночная опора."),
            PALE_GREEN,
            GREEN,
        ),
        (
            "Банк и макрорежим",
            {
                "halyk_rub_sell_official_basis", "halyk_rub_personal_legal_gap",
                "halyk_personal_rub_ret1", "halyk_personal_rub_ret5",
                "halyk_personal_usd_ret1", "halyk_personal_usd_ret5",
                *(f"treasury_{symbol}_stale_{suffix}" for symbol in ("t10yie", "t5yifr") for suffix in ("level", "chg5", "chg20")),
            },
            "Halyk sell · personal/legal gap · US inflation expectations",
            ("Добавляем банковскую цену и ликвидность", "в Казахстане, а также глобальный", "инфляционный фон для режима рынка."),
            "#FFF5E8",
            AMBER,
        ),
    ]
    covered = set().union(*(features for _, features, *_ in groups))
    if covered != h3_features:
        raise RuntimeError(f"trigger map does not cover feature contract: missing={sorted(h3_features-covered)} extra={sorted(covered-h3_features)}")

    c = Canvas("Триггеры AlphaTransfer", "Экономическая карта 33 признаков H3 и H5.")
    header(c, "На какие изменения реагируют модели", "33 point-in-time признака · одинаковый feature contract H3/H5 · RUB → KZT")
    for idx, (title, features, source, lines, fill, accent) in enumerate(groups):
        col, row = idx % 2, idx // 2
        x0, y0 = 70 + col * 765, 165 + row * 270
        c.rect((x0, y0, x0 + 730, y0 + 235), fill, radius=24)
        c.circle(x0 + 45, y0 + 48, 22, accent)
        c.text(x0 + 45, y0 + 48, len(features), 17, bold=True, fill=WHITE, anchor="mm")
        c.text(x0 + 82, y0 + 57, title, 27, bold=True, fill=accent)
        c.text(x0 + 34, y0 + 101, source, 17, fill=MUTED)
        for line_idx, line in enumerate(lines):
            c.text(x0 + 34, y0 + 147 + line_idx * 27, line, 20, bold=line_idx == 0)
    c.rect((70, 725, 1530, 840), INK, radius=20)
    c.text(105, 772, "Важно", 21, bold=True, fill="#7DE2AA")
    c.text(200, 772, "Это входы модели, а не четыре жёстких правила.", 22, bold=True, fill=WHITE)
    c.text(105, 813, "TabM нелинейно объединяет признаки; направление и сила влияния обучаются на train, а пропуски получают отдельные индикаторы.", 19, fill="#D5D5D5")
    c.text(1530, 882, "Интерпретация экономическая; визуализация не заменяет attribution-анализ конкретного дня.", 16, fill=MUTED, anchor="ra")
    c.save(out, "08_trigger_map")


def build_signal_pipeline(out: Path) -> None:
    c = Canvas("От триггера до пуша", "Причинная цепочка AlphaTransfer от доступных данных до клиентского сообщения.")
    header(c, "От рыночного изменения до клиентского сообщения", "Модель выбирает момент; продуктовые правила решают, можно ли и как связаться с клиентом")
    stages = [
        ("1", "Данные", ("CBR · MOEX · OXR", "Halyk · Treasury", "доступны к cutoff"), PALE, INK),
        ("2", "Признаки", ("уровень · импульс", "волатильность · basis", "банк · макрорежим"), PALE_BLUE, BLUE),
        ("3", "Модель TabM", ("P(NOW_H3)", "P(NOW_H5)", "отдельные цели"), PALE_GREEN, GREEN),
        ("4", "Политика", ("калибровка: прошлое", "rank: 63 score", "порог · cooldown"), "#FFF5E8", AMBER),
        ("5", "Доставка", ("коридор · намерение", "consent · срочность", "≤ 2 в неделю"), PALE, INK),
        ("6", "Сообщение", ("текущий факт", "прошлая динамика", "цена заново в app"), PALE_BLUE, BLUE),
    ]
    box_w, gap, x_start = 220, 25, 55
    for idx, (number, title, lines, fill, accent) in enumerate(stages):
        x0 = x_start + idx * (box_w + gap)
        c.rect((x0, 180, x0 + box_w, 560), fill, radius=22)
        c.circle(x0 + 35, 218, 20, accent)
        c.text(x0 + 35, 218, number, 17, bold=True, fill=WHITE, anchor="mm")
        c.text(x0 + 22, 285, title, 21, bold=True, fill=accent)
        for line_idx, line in enumerate(lines):
            c.text(x0 + 22, 350 + line_idx * 39, line, 16, bold=line_idx == 0)
        if idx < len(stages) - 1:
            c.arrow((x0 + box_w + 4, 370, x0 + box_w + gap - 5, 370), fill=INK, width=3)

    c.rect((70, 625, 760, 825), PALE, radius=22)
    c.text(104, 674, "Нет NOW или не пройден delivery gate", 23, bold=True)
    c.text(104, 718, "Проактивного пуша нет.", 26, bold=True, fill=MUTED)
    c.text(104, 758, "Шкала курса остаётся доступной при входе в приложение;", 18, fill=MUTED)
    c.text(104, 790, "политика не создаёт искусственный сигнал в тихую неделю.", 18, fill=MUTED)

    c.rect((800, 625, 1530, 825), PALE_BLUE, radius=22)
    c.text(834, 671, "NOW прошёл все gates", 23, bold=True, fill=BLUE)
    c.text(834, 713, "«Курс сегодня — среди 15% самых низких", 23, bold=True)
    c.text(834, 748, "значений за три месяца»", 23, bold=True)
    c.text(834, 790, "CLOSING_H3 может усилить тот же контакт — без второго пуша.", 18, fill=MUTED)
    c.text(1530, 882, "Перед показом сумма и курс пересчитываются; финальный текст утверждает комплаенс банка.", 16, fill=MUTED, anchor="ra")
    c.save(out, "09_signal_pipeline")


def snapshot_rows(profiles: tuple[Profile, Profile]) -> list[dict[str, str]]:
    columns = [
        "rows", "dates", "signals", "hits", "base_hit", "hit_rate", "lift",
        "forward_delta_bps", "forward_signal_bps", "regret_bps", "brier", "raw_brier",
        "auc", "log_loss", "week_cells", "weeks_with_1_2", "week_coverage",
        "mean_cell_week_coverage", "signals_per_corridor_week", "silent_weeks",
        "max_week_signals", "max_silent_week_run", "min_year_corridor_coverage",
    ]
    result: list[dict[str, str]] = []
    for profile in profiles:
        for period, row in [("2024-2026", profile.aggregate)] + [
            (annual["year"], annual) for annual in profile.annual
        ]:
            result.append({
                "model": profile.model,
                "label": profile.label,
                "config_id": profile.config_id,
                "horizon": str(profile.horizon),
                "policy": profile.policy,
                "scope": "KZT",
                "period": str(period),
                "source": profile.source,
                **{key: row.get(key, "") for key in columns},
            })
    return result


def write_snapshot(out: Path, profiles: tuple[Profile, Profile]) -> None:
    rows = snapshot_rows(profiles)
    fieldnames = list(rows[0])
    with (out / "metrics_snapshot.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sources = [
        H3_DIR / "bundle.json", H3_DIR / "evaluation" / "selection.json",
        H3_DIR / "evaluation" / "summary.csv", H3_DIR / "evaluation" / "by_year.csv",
        H5_DIR / "selection.json", H5_DIR / "summary.csv", H5_DIR / "by_year.csv",
        H5_DIR / "receipts.json", H5_DIR / "paired_intervals.csv", H5_DIR / "verification.json",
        ROOT / "research_v4" / "h3_finalization" / "closing_predictions.csv.gz",
    ]
    receipt = {
        "schema_version": 1,
        "status": "canonical_selected_profiles",
        "rows": len(rows),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ],
    }
    (out / "metrics_snapshot_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(out: Path) -> None:
    profiles = validate_and_load()
    out.mkdir(parents=True, exist_ok=True)
    write_snapshot(out, profiles)
    build_scorecard(out, profiles)
    build_stability(out, profiles)
    build_frontier(out)
    build_utility(out, profiles)
    build_h3_history(out)
    build_closing(out)
    build_signal_timeline(out)
    build_trigger_map(out)
    build_signal_pipeline(out)


def check() -> None:
    validate_and_load()
    with tempfile.TemporaryDirectory(prefix="alphatransfer-readme-") as temp:
        expected = Path(temp)
        build(expected)
        wanted = sorted(path.name for path in expected.iterdir())
        missing = [name for name in wanted if not (OUT / name).is_file()]
        changed = [
            name for name in wanted
            if (OUT / name).is_file() and (OUT / name).read_bytes() != (expected / name).read_bytes()
        ]
        if missing or changed:
            raise RuntimeError(f"README assets are stale: missing={missing}, changed={changed}")
    print(f"PASS: {len(wanted)} deterministic README report artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="write canonical README assets")
    mode.add_argument("--check", action="store_true", help="verify committed assets are current")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        build(OUT)
        print(f"Built README assets in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
