#!/usr/bin/env python3
"""Build lightweight SVG charts from aggregate chat analysis outputs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
FONT = "Inter, Arial, sans-serif"
COLORS = {"red": "#EF3124", "dark": "#222222", "gray": "#8C8C8C", "light": "#F1F2F4", "blue": "#3B82F6"}


def esc(value: Any) -> str:
    return html.escape(str(value))


def text(x: float, y: float, value: Any, size: int = 13, anchor: str = "start", weight: int = 400, fill: str = "#222222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'
    )


def svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            text(40, 38, title, 20, weight=700),
            *body,
            "</svg>",
        ]
    )


def horizontal_grouped_bars(
    title: str,
    labels: list[str],
    first: list[float],
    second: list[float],
    first_label: str,
    second_label: str,
    output: Path,
) -> None:
    width = 1040
    row_height = 48
    height = 120 + row_height * len(labels)
    left = 245
    chart_width = 700
    maximum = max(first + second) * 1.12 if first or second else 1
    body = [
        f'<rect x="{left}" y="57" width="14" height="14" fill="{COLORS["red"]}"/>',
        text(left + 20, 69, first_label, 12),
        f'<rect x="{left + 245}" y="57" width="14" height="14" fill="{COLORS["blue"]}"/>',
        text(left + 265, 69, second_label, 12),
    ]
    for index, label in enumerate(labels):
        y = 98 + index * row_height
        body.append(text(left - 14, y + 15, label, 12, anchor="end"))
        body.append(f'<rect x="{left}" y="{y}" width="{chart_width}" height="1" fill="{COLORS["light"]}"/>')
        first_width = chart_width * first[index] / maximum
        second_width = chart_width * second[index] / maximum
        body.append(f'<rect x="{left}" y="{y + 5}" width="{first_width:.1f}" height="13" rx="2" fill="{COLORS["red"]}"/>')
        body.append(f'<rect x="{left}" y="{y + 23}" width="{second_width:.1f}" height="13" rx="2" fill="{COLORS["blue"]}"/>')
        body.append(text(left + first_width + 6, y + 16, f"{first[index]:.1f}%", 11))
        body.append(text(left + second_width + 6, y + 34, f"{second[index]:.1f}%", 11))
    output.write_text(svg_document(width, height, title, body), encoding="utf-8")


def build_activity() -> None:
    with (DATA / "monthly.csv").open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["month"] not in {"2022-06", "2026-09"}]
    width, height = 1240, 620
    left, right = 65, 30
    chart_width = width - left - right
    panel_height = 205
    start_top = 85
    gap = 90
    labels = [row["month"] for row in rows]
    messages = [int(row["messages"]) for row in rows]
    authors = [int(row["unique_authors"]) for row in rows]
    body: list[str] = []

    for panel_index, (values, panel_title, color) in enumerate(
        [(messages, "Сообщения в месяц", COLORS["red"]), (authors, "Уникальные авторы в месяц", COLORS["blue"])]
    ):
        top = start_top + panel_index * (panel_height + gap)
        maximum = max(values)
        body.append(text(left, top - 15, panel_title, 14, weight=700))
        for tick in range(5):
            fraction = tick / 4
            y = top + panel_height - panel_height * fraction
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["light"]}"/>')
            body.append(text(left - 8, y + 4, f"{maximum * fraction:,.0f}".replace(",", " "), 10, anchor="end", fill=COLORS["gray"]))
        step = chart_width / max(1, len(values) - 1)
        points = []
        for index, value in enumerate(values):
            x = left + index * step
            y = top + panel_height - panel_height * value / maximum
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for index, value in enumerate(values):
            if index % 4 == 0 or index == len(values) - 1:
                x = left + index * step
                body.append(text(x, top + panel_height + 18, labels[index], 10, anchor="middle", fill=COLORS["gray"]))
    body.append(text(left, height - 18, "Граничные неполные месяцы 2022-06 и 2026-09 исключены", 11, fill=COLORS["gray"]))
    (ASSETS / "activity.svg").write_text(svg_document(width, height, "Активность чата: после пика 2022 года сохраняется меньший стабильный поток", body), encoding="utf-8")


def build_categories(summary: dict[str, Any]) -> None:
    names = {
        "cash_atm": "Наличные / банкоматы",
        "rate_information": "Курс",
        "fees_cost": "Комиссии / стоимость",
        "crossborder_howto": "Как перевести между странами",
        "trust_safety": "Доверие / безопасность",
        "bank_blocks_compliance": "Блокировки / комплаенс",
        "timing_forecast": "Тайминг / прогноз",
        "family_support": "Семья / помощь близким",
    }
    by_key = {row["category"]: row for row in summary["categories"]}
    keys = list(names)
    horizontal_grouped_bars(
        "Лексические маркеры тем: не prevalence потребностей",
        [names[key] for key in keys],
        [by_key[key]["share_of_text_messages"] * 100 for key in keys],
        [by_key[key]["share_of_authors"] * 100 for key in keys],
        "Доля текстовых сообщений",
        "Доля авторов",
        ASSETS / "need_signals.svg",
    )


def build_participation(summary: dict[str, Any]) -> None:
    rows = summary["participation"]["author_bins"]
    horizontal_grouped_bars(
        "Громкость не равна распространённости: активность крайне концентрирована",
        [row["bin"] + " сообщений" for row in rows],
        [row["author_share"] * 100 for row in rows],
        [row["message_share"] * 100 for row in rows],
        "Доля авторов",
        "Доля сообщений",
        ASSETS / "participation.svg",
    )


def build_market_reactivity(summary: dict[str, Any]) -> None:
    event = summary["market_relationships"]["event_study"]
    high = event["high_volatility_days"]
    other = event["other_days"]
    labels = ["Лексика о курсе", "Лексика о тайминге / прогнозе"]
    high_values = [
        high["mean_rate_information_share"] * 100,
        high["mean_timing_forecast_share"] * 100,
    ]
    other_values = [
        other["mean_rate_information_share"] * 100,
        other["mean_timing_forecast_share"] * 100,
    ]
    horizontal_grouped_bars(
        "Описательная связь с волатильностью: различие мало и не причинно",
        labels,
        high_values,
        other_values,
        f"Топ-10% волатильности (n={high['n_days']})",
        f"Остальные дни (n={other['n_days']})",
        ASSETS / "market_reactivity.svg",
    )


def build_banks(summary: dict[str, Any]) -> None:
    rows = summary["banks_and_rails"][:10]
    total_messages = summary["source"]["text_messages"]
    total_authors = summary["source"]["authors"]
    horizontal_grouped_bars(
        "Упоминания банков и рельс: не использование и не market share",
        [row["entity"] for row in rows],
        [row["messages"] / total_messages * 100 for row in rows],
        [row["authors"] / total_authors * 100 for row in rows],
        "Доля текстовых сообщений",
        "Доля авторов",
        ASSETS / "banks_and_rails.svg",
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    build_activity()
    build_categories(summary)
    build_participation(summary)
    build_market_reactivity(summary)
    build_banks(summary)


if __name__ == "__main__":
    main()
