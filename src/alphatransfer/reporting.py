"""Deterministic, self-contained HTML report for the V0 research gate."""

from __future__ import annotations

from datetime import date, timedelta
import html
import json
from pathlib import Path
from statistics import mean


def data_quality(rows: list[dict], calendar: list[dict] | None = None) -> dict:
    dates = [date.fromisoformat(r["date"]) for r in rows]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    result = {
        "grain": "one row per fresh MOEX KZTRUB_TOM daily candle",
        "rows": len(rows), "start": rows[0]["date"], "end": rows[-1]["date"],
        "duplicate_dates": len(rows) - len(set(r["date"] for r in rows)),
        "cbr_fresh_share": round(mean(float(r["cbr_is_fresh"]) for r in rows), 4),
        "nbk_fresh_share": round(mean(float(r["nbk_is_fresh"]) for r in rows), 4),
        "max_session_gap_days": max(gaps) if gaps else 0,
        "missing_core_values": sum(any(float(r[k]) <= 0 for k in ("cbr_rate", "nbk_rate", "moex_close")) for r in rows),
    }
    if calendar:
        result.update({"calendar_days": len(calendar),
                       "closed_calendar_days": sum(str(r["moex_is_fresh"]) == "0" for r in calendar),
                       "nbk_unavailable_calendar_days": sum(not str(r["nbk_rate"]) for r in calendar)})
    return result


def _svg(rows: list[dict], events: list[dict], width: int = 1100, height: int = 340) -> str:
    sample = rows[-900:]
    if not sample:
        return ""
    margin = 42
    keys = ("cbr_rate", "nbk_rate", "moex_close")
    values = [float(r[k]) for r in sample for k in keys]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-9)
    index = {r["date"]: i for i, r in enumerate(sample)}
    def point(i: int, value: float) -> tuple[float, float]:
        x = margin + i * (width - 2 * margin) / max(len(sample) - 1, 1)
        y = height - margin - (value - lo) * (height - 2 * margin) / span
        return x, y
    colors = {"cbr_rate": "#2563eb", "nbk_rate": "#0f766e", "moex_close": "#d97706"}
    paths = []
    for key in keys:
        coords = [point(i, float(r[key])) for i, r in enumerate(sample)]
        d = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(coords))
        paths.append(f'<path d="{d}" fill="none" stroke="{colors[key]}" stroke-width="1.5" opacity=".9"/>')
    marks = []
    for event in events:
        if event["date"] not in index:
            continue
        i = index[event["date"]]
        x, y = point(i, float(sample[i]["cbr_rate"]))
        color = "#16a34a" if event["scenario"] == "favorable_now" else "#dc2626"
        opacity = "1" if event.get("policy_eligible") else ".28"
        marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" opacity="{opacity}"><title>{html.escape(event["date"])} · {html.escape(event["scenario"])}</title></circle>')
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Три ряда RUB за KZT и OOT-сигналы">
      <rect width="100%" height="100%" fill="#fff"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#cbd5e1"/>
      {''.join(paths)}{''.join(marks)}
      <text x="{margin}" y="{height-12}" class="axis">{sample[0]['date']}</text><text x="{width-margin}" y="{height-12}" text-anchor="end" class="axis">{sample[-1]['date']}</text>
      <text x="{margin}" y="20" class="axis">{hi:.4f}</text><text x="{margin}" y="{height-margin-6}" class="axis">{lo:.4f}</text>
    </svg>'''


def _table(items: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for item in items:
        cells = "".join(f"<td>{html.escape(str(item.get(key, '—') if item.get(key) is not None else '—'))}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def build_snapshot(rows: list[dict], backtest: dict, signal: dict, calendar: list[dict] | None = None) -> dict:
    quality = data_quality(rows, calendar)
    lifts = backtest["primary_lifts"]
    robust = [float(x["lift"] or 0) >= 1.3 for x in backtest["source_robustness"]]
    gate = max(lifts.values() or [0]) >= 1.3 and sum(robust) >= 2 and quality["duplicate_dates"] == 0 and quality["missing_core_values"] == 0
    return {"as_of": rows[-1]["date"], "quality": quality, "backtest": backtest, "signal": signal,
            "decision": "go" if gate else "no-go",
            "decision_reason": "минимум один сценарий прошёл OOT lift ≥ 1.3, результат подтвердили ≥2 источника и базовые проверки данных" if gate else "не пройден один из gate: OOT lift ≥ 1.3, подтверждение ≥2 источниками или базовое качество данных"}


def render_report(snapshot: dict, rows: list[dict], calendar: list[dict] | None = None) -> str:
    bt, quality, signal = snapshot["backtest"], snapshot["quality"], snapshot["signal"]
    decision = snapshot["decision"]
    metrics = _table(bt["metrics"], [("horizon", "h"), ("scenario", "сценарий"), ("signals", "N"), ("hit_rate", "hit rate"), ("random_hit_rate", "random"), ("lift", "lift"), ("mean_timing_bps", "timing, bps"), ("fast_wait_cost_bps", "wait 1d, bps"), ("slow_wait_cost_bps", "wait 3d, bps"), ("false_positive_cost", "FP cost"), ("brier_score", "Brier")])
    robustness = _table(bt["source_robustness"], [("source", "источник"), ("signals", "N"), ("hit_rate", "hit rate"), ("random_hit_rate", "random"), ("lift", "lift")])
    empty_metric = {"signals": 0, "hit_rate": 0.0, "random_hit_rate": 0.0, "lift": None}
    primary = next((m for m in bt["metrics"] if m["horizon"] == 5 and m["scenario"] == "favorable_now"), empty_metric)
    alternate = next((m for m in bt["metrics"] if m["horizon"] == 5 and m["scenario"] == "window_closing"), empty_metric)
    chart = _svg(rows, bt["events"])
    evidence = "".join(f"<li><code>{html.escape(str(x.get('feature', x)))}</code> {html.escape(str(x.get('contribution', '')))}</li>" if isinstance(x, dict) else f"<li>{html.escape(str(x))}</li>" for x in signal.get("evidence", []))
    calendar = calendar or rows
    holiday_rows = [r for r in calendar[-180:] if int(float(r["is_ru_holiday"])) or int(float(r["is_kz_holiday"])) or str(r.get("moex_is_fresh", "1")) == "0"]
    holidays = _table(holiday_rows[-40:], [("date", "дата"), ("is_ru_holiday", "RU holiday"), ("is_kz_holiday", "KZ holiday"), ("cbr_age_days", "CBR age"), ("nbk_age_days", "NBK age")])
    data_json = html.escape(json.dumps({"quality": quality, "signal": signal}, ensure_ascii=False, sort_keys=True))
    return f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RUB→KZT V0 — {decision}</title><style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe2ea;--soft:#f5f7fa;--blue:#2563eb}}*{{box-sizing:border-box}}body{{margin:0;background:#edf1f5;color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1200px;margin:auto;background:white;min-height:100vh;padding:44px}}h1{{font-size:36px;line-height:1.1;margin:0 0 8px}}h2{{font-size:20px;margin:38px 0 12px}}h3{{font-size:15px;margin:0 0 8px}}.muted,.axis{{color:var(--muted);fill:var(--muted);font-size:11px}}.decision{{display:grid;grid-template-columns:150px 1fr;gap:24px;border-top:3px solid {'#16a34a' if decision=='go' else '#dc2626'};background:var(--soft);padding:22px;margin:24px 0}}.badge{{font-size:30px;font-weight:800;text-transform:uppercase}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid var(--line);padding:16px;border-radius:8px}}.kpi{{font-size:25px;font-weight:700}}svg{{width:100%;border:1px solid var(--line)}}.legend span{{margin-right:18px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{{text-align:left}}.table-wrap{{overflow:auto;border:1px solid var(--line)}}code{{background:#eef2ff;padding:2px 5px;border-radius:4px}}.note{{border-left:3px solid #94a3b8;padding-left:12px}}@media(max-width:720px){{main{{padding:22px}}.grid,.decision{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><main>
<p class="muted">AlphaTransfer · офлайн V0 · данные по {snapshot['as_of']}</p><h1>Сигнальный слой RUB→KZT: {decision}</h1>
<p>Проверяем, можно ли прозрачно находить выгодный момент перевода. Target — курс ЦБ РФ; НБК и MOEX не смешиваются в «истину», а проверяют устойчивость.</p>
<section class="decision"><div class="badge">{decision}</div><div><h3>Решение исследовательского gate</h3><div>{snapshot['decision_reason']}</div><div class="muted">Это не подтверждение продуктовой конверсии, deep link или stale-price UX.</div></div></section>
<p class="note"><strong>Главный результат.</strong> На h=5 сценарий favorable_now дал {primary['signals']} OOT-сигналов, hit rate {primary['hit_rate']:.1%} против {primary['random_hit_rate']:.1%} у случайного допустимого дня (lift {primary['lift']}). Но на этих датах NBK и MOEX дали lift 0. Window_closing формально достиг lift {alternate['lift']}, однако его random hit rate уже {alternate['random_hit_rate']:.1%}; различающая способность слабая. Поэтому запускать отправку рано.</p>
<div class="grid"><div class="card"><div class="muted">MOEX-сессий</div><div class="kpi">{quality['rows']}</div><div>{quality['start']} — {quality['end']}</div></div><div class="card"><div class="muted">CBR fresh на сессиях</div><div class="kpi">{quality['cbr_fresh_share']:.1%}</div></div><div class="card"><div class="muted">Сигнал сейчас</div><div class="kpi">{html.escape(str(signal.get('scenario') or 'нет'))}</div><div>send={str(signal.get('eligible_to_send', False)).lower()}</div></div></div>
<h2>Три независимых представления одного коридора</h2><p class="legend"><span><i class="dot" style="background:#2563eb"></i>CBR</span><span><i class="dot" style="background:#0f766e"></i>NBK</span><span><i class="dot" style="background:#d97706"></i>MOEX</span><span><i class="dot" style="background:#16a34a"></i>favorable</span><span><i class="dot" style="background:#dc2626"></i>closing</span></p>{chart}
<h2>Out-of-time матрица</h2><p class="muted">Expanding 36m train → 6m validation → 3m untouched test; FP стоит 3× FN. Random baseline использует те же test-периоды и MOEX-дни.</p>{metrics}
<h2>Устойчивость target</h2><p class="muted">Тот же поток favorable-кандидатов пересчитан против локальных минимумов каждого ряда отдельно.</p>{robustness}
<h2>Текущий вывод модели</h2><div class="grid"><div class="card"><h3>Вероятность</h3><div class="kpi">{float(signal.get('confidence',0)):.1%}</div><div>{html.escape(str(signal.get('suppressed_reason') or 'допущен policy'))}</div></div><div class="card"><h3>Курсы, RUB/KZT</h3><pre>{html.escape(json.dumps(signal.get('rate_snapshot',{}),ensure_ascii=False,indent=2))}</pre></div><div class="card"><h3>Основания</h3><ul>{evidence or '<li>кандидат не сформирован</li>'}</ul></div></div>
<h2>Календарь пропусков и freshness</h2><p class="note">Календарные признаки только корректируют вероятность. Сигнал появляется исключительно на свежей дневной свече MOEX; заполнение CBR/NBK хранит age_days и не создаёт momentum.</p>{holidays}
<h2>Что V0 действительно проверяет</h2><div class="grid"><div class="card"><h3>H1 · timing</h3><p>Lift и выгода момента против случайного допустимого дня.</p></div><div class="card"><h3>H4/H7 · частота и устойчивость</h3><p>Cooldown/cap и независимый пересчёт по CBR, NBK, MOEX.</p></div><div class="card"><h3>H8 + основание H2</h3><p>Детерминированные признаки, коэффициенты/стампы и фактические evidence. Конверсия H2 пока не измерена.</p></div></div>
<h2>Источники и воспроизводимость</h2><p><a href="https://www.cbr.ru/development/SXML/">ЦБ РФ · XML_dynamic</a> · <a href="https://nationalbank.kz/ru/page/rss">НБ Казахстана · официальный daily XML</a> · <a href="https://iss.moex.com/iss/reference/155">MOEX ISS · candles</a>. Raw-ответ и исходный URL сохраняются в кэше; неизвестная схема останавливает pipeline.</p>
<p class="muted">Ограничения: общий модельный интервал начинается {quality['start']}, поскольку официальный daily XML НБК не вернул более ранний архив; available_at для исторических официальных курсов реконструирован консервативно как начало effective_date, а для MOEX — после дневного закрытия. Holiday-флаги отражают закреплённые даты и перенос с выходного; фактическую доступность сигнала всегда определяет наличие свечи MOEX.</p>
<details><summary>Проверяемый snapshot</summary><pre>{data_json}</pre></details>
</main></body></html>'''


def write_report(snapshot: dict, rows: list[dict], html_path: str | Path, app_dir: str | Path, calendar: list[dict] | None = None) -> tuple[Path, Path]:
    content = render_report(snapshot, rows, calendar)
    target = Path(html_path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content, encoding="utf-8")
    dist = Path(app_dir) / "dist" / "index.html"; dist.parent.mkdir(parents=True, exist_ok=True); dist.write_text(content, encoding="utf-8")
    return target, dist
