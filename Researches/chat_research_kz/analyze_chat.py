#!/usr/bin/env python3
"""Reproducible, privacy-preserving analysis of a Telegram JSON export.

The script never writes author identifiers or raw message text. Outputs contain
only aggregate counters and distributions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SPACE_RE = re.compile(r"\s+")
SEARCH_RE = re.compile(r"(?iu)(?:#\s*)?\b(?:ищу|нужно|нужны|нужен|куплю)\b")
OFFER_RE = re.compile(r"(?iu)(?:#\s*)?\b(?:предлагаю|есть|отдам|продам)\b")
RUB_TOKEN = r"(?:\brub\b|рубл|руб\.?\b|₽|сбп|сбер|тинькофф|т-?банк|альфа)"
KZT_TOKEN = r"(?:\bкзт\b|\bkzt\b|тенг|₸|\bтг\b|каспи|kaspi)"
NEED_TOKEN = r"(?:ищу|нужно|нужны|нужен|нужна|куплю)"
OFFER_TOKEN = r"(?:предлагаю|есть|имеются|отдам|продам|готов\s+отдать)"
EXCHANGE_TOKEN = r"(?:поменя\w*|обменя\w*|продаст\w*|куплю|обмен)"

NEED_KZT_RE = re.compile(rf"(?iu)\b{NEED_TOKEN}\b.{{0,100}}{KZT_TOKEN}")
NEED_RUB_RE = re.compile(rf"(?iu)\b{NEED_TOKEN}\b.{{0,100}}{RUB_TOKEN}")
OFFER_KZT_RE = re.compile(rf"(?iu)\b{OFFER_TOKEN}\b.{{0,100}}{KZT_TOKEN}")
OFFER_RUB_RE = re.compile(rf"(?iu)\b{OFFER_TOKEN}\b.{{0,100}}{RUB_TOKEN}")
PAIR_EXCHANGE_RE = re.compile(
    rf"(?iu)(?:\b{EXCHANGE_TOKEN}\b.{{0,140}}{RUB_TOKEN}.{{0,100}}{KZT_TOKEN}|"
    rf"\b{EXCHANGE_TOKEN}\b.{{0,140}}{KZT_TOKEN}.{{0,100}}{RUB_TOKEN}|"
    rf"{RUB_TOKEN}.{{0,100}}\b{EXCHANGE_TOKEN}\b.{{0,100}}{KZT_TOKEN}|"
    rf"{KZT_TOKEN}.{{0,100}}\b{EXCHANGE_TOKEN}\b.{{0,100}}{RUB_TOKEN})"
)
WHO_NEEDS_RE = re.compile(
    rf"(?iu)кому.{{0,30}}нужн\w*.{{0,40}}(?P<offer>{RUB_TOKEN}|{KZT_TOKEN}).{{0,160}}"
    rf"(?:мне\s+)?нужн\w*.{{0,40}}(?P<need>{RUB_TOKEN}|{KZT_TOKEN})"
)

CURRENCY_PATTERNS = {
    "kzt": re.compile(r"(?iu)(?:\bкзт\b|\bkzt\b|тенг|₸|\bтг\b|каспи|kaspi)"),
    "rub": re.compile(
        r"(?iu)(?:\brub\b|рубл|руб\.?\b|₽|\bр\.?\s*(?:сбп|на|$)|сбп|сбер|тинькофф|т-?банк|альфа)"
    ),
    "usd": re.compile(r"(?iu)(?:\busd\b|доллар|\$)"),
    "eur": re.compile(r"(?iu)(?:\beur\b|евро|€)"),
}

CATEGORY_PATTERNS = {
    "exchange_intent": re.compile(
        r"(?iu)(?:#\s*(?:ищу|предлагаю|меняю)|\b(?:ищу|предлагаю|меняю|обменяю|обменяюсь)\b)"
    ),
    "rate_information": re.compile(
        r"(?iu)(?:\bкурс\w*|котиров|сколько\s+(?:сейчас\s+)?(?:стоит|дают)|по\s+какому\s+курсу)"
    ),
    "timing_forecast": re.compile(
        r"(?iu)(?:прогноз|курс.{0,30}(?:будет|выраст|упад|подним|сниз)|"
        r"(?:жду|ждём|ждем).{0,30}курс|подожд|когда.{0,30}(?:менят|обмен)|"
        r"(?:стоит|лучше).{0,30}(?:сейчас\s+)?(?:менят|обмен)|выгодн.{0,20}момент)"
    ),
    "urgency": re.compile(
        r"(?iu)(?:\bсрочн\w*|прямо\s+сейчас|как\s+можно\s+скорее|нужно\s+сегодня|до\s+вечера|горит)"
    ),
    "willing_to_wait": re.compile(
        r"(?iu)(?:(?:жду|ждём|ждем).{0,30}курс|подожд|не\s+срочн|можно\s+не\s+сейчас|"
        r"до\s+(?:завтра|выходн|конца\s+недел)|через\s+(?:день|пару\s+дней|недел))"
    ),
    "fees_cost": re.compile(r"(?iu)(?:комисс|переплат|спред|без\s+комис|процент.{0,20}перевод)"),
    "trust_safety": re.compile(
        r"(?iu)(?:мошенн|скам|кидал|обман|довер|репутац|отзыв|проверен|гарант|"
        r"безопасн|удалил.{0,20}(?:чат|переписк|аккаунт)|чёрн.{0,10}спис|черн.{0,10}спис)"
    ),
    "bank_blocks_compliance": re.compile(
        r"(?iu)(?:заблокир|блокиров|финмон|115\s*[-‑]?\s*фз|подтвержден.{0,20}операц|"
        r"документ.{0,20}(?:банк|перевод)|заморозил.{0,20}(?:счёт|счет|карт))"
    ),
    "crossborder_howto": re.compile(
        r"(?iu)(?:(?:как|чем|через\s+что|каким\s+образом).{0,60}(?:перевест|отправ|пополн)|"
        r"(?:перевест|отправ|пополн).{0,50}(?:из\s+рф|из\s+росси|в\s+казахстан|на\s+каспи|в\s+рф))"
    ),
    "cash_atm": re.compile(r"(?iu)(?:налич|банкомат|снять|снятие|обнал|обменник)"),
    "crypto": re.compile(
        r"(?iu)(?:крипт|\busdt\b|\bbtc\b|биткоин|binance|байбит|bybit|p2p.{0,15}(?:бирж|крипт))"
    ),
    "family_support": re.compile(
        r"(?iu)(?:семь[ея]|родител|родствен|мам[еуы]|пап[еуы]|детям|реб[её]нк|домой|алименты)"
    ),
    "salary_income": re.compile(r"(?iu)(?:зарплат|аванс|гонорар|доход|получк)"),
    "rent_study_medical": re.compile(r"(?iu)(?:аренд|квартир|уч[её]б|университет|лечен|врач|операц)"),
    "benchmark_reference": re.compile(
        r"(?iu)(?:курс.{0,15}(?:гугл|google|цб|нацбанк|бирж)|(?:гугл|google).{0,15}курс|рыночн.{0,15}курс)"
    ),
    "successful_close": re.compile(
        r"(?iu)(?:обменял(?:ся|ась|ись)?|успешно\s+обмен|сделка\s+(?:закрыта|состоялась)|"
        r"(?:уже\s+)?(?:не\s+актуально|закрыто)|наш[её]л(?:а)?|спасибо.{0,30}(?:обмен|перевод))"
    ),
    "partial_fill": re.compile(r"(?iu)(?:частичн|остал(?:ось|ся)|можно\s+частями|по\s+частям)"),
}

BANK_PATTERNS = {
    "Kaspi": re.compile(r"(?iu)(?:\bkaspi\b|каспи)"),
    "SBP": re.compile(r"(?iu)\bсбп\b"),
    "Sber": re.compile(r"(?iu)(?:сбер|sber)"),
    "T-Bank/Tinkoff": re.compile(r"(?iu)(?:тинькофф|т-?банк|tinkoff)"),
    "Halyk": re.compile(r"(?iu)(?:halyk|халык|народн.{0,10}банк)"),
    "BCC": re.compile(r"(?iu)(?:\bbcc\b|\bбцк\b|центркредит)"),
    "Freedom": re.compile(r"(?iu)(?:freedom|фридом)"),
    "Alfa": re.compile(r"(?iu)(?:альфа|alfa)"),
    "VTB": re.compile(r"(?iu)\bвтб\b"),
    "Forte": re.compile(r"(?iu)(?:forte|форте)"),
    "Jusan/Alatau": re.compile(r"(?iu)(?:jusan|жусан|alatau|алатау)"),
    "KoronaPay": re.compile(r"(?iu)(?:korona\s*pay|коронап|золот.{0,5}корон)"),
    "Avosend": re.compile(r"(?iu)avosend"),
    "Paysend": re.compile(r"(?iu)paysend"),
    "Qiwi": re.compile(r"(?iu)(?:qiwi|киви)"),
}

RATE_RE = re.compile(
    r"(?iu)\bкурс(?:ом|а|у)?\s*(?:[:=\-–—]|по)?\s*(\d{1,2}(?:[.,]\d{1,4})?)"
)
AMOUNT_RE = re.compile(
    r"(?iu)(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+(?:[.,]\d+)?\s*[кk]|\d+)\s*"
    r"(руб(?:лей|ля)?|₽|\brub\b|тенге|\bтг\b|₸|\bkzt\b)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def iter_messages(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Stream objects from the top-level messages array without extra packages."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        buffer = ""
        while '"messages"' not in buffer:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError("messages array not found")
            buffer += chunk
        marker = buffer.index('"messages"')
        start = buffer.find("[", marker)
        while start < 0:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError("messages array opening bracket not found")
            buffer += chunk
            start = buffer.find("[", marker)
        buffer = buffer[start + 1 :]
        position = 0

        while True:
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                if position < len(buffer):
                    break
                buffer = source.read(chunk_size)
                position = 0
                if not buffer:
                    raise ValueError("unexpected EOF inside messages array")

            if buffer[position] == "]":
                return

            while True:
                try:
                    message, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        raise
                    buffer = buffer[position:] + chunk
                    position = 0
            yield message
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            nested = item.get("text", "")
            if isinstance(nested, str):
                parts.append(nested)
    return "".join(parts)


def normalized_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def day_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p10": None, "p25": None, "median": None, "p75": None, "p90": None}
    return {
        "n": len(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def detect_currency(segment: str) -> set[str]:
    return {name for name, pattern in CURRENCY_PATTERNS.items() if pattern.search(segment)}


def strict_listing(text: str) -> bool:
    """High-precision P2P intent: two-sided offer/need or explicit pair exchange."""
    has_both_currencies = bool(CURRENCY_PATTERNS["rub"].search(text) and CURRENCY_PATTERNS["kzt"].search(text))
    if not has_both_currencies:
        return False
    two_sided = bool(
        (NEED_KZT_RE.search(text) and OFFER_RUB_RE.search(text))
        or (NEED_RUB_RE.search(text) and OFFER_KZT_RE.search(text))
        or WHO_NEEDS_RE.search(text)
    )
    return two_sided or bool(PAIR_EXCHANGE_RE.search(text))


def probable_listing(text: str) -> bool:
    if strict_listing(text):
        return True
    has_both_currencies = bool(CURRENCY_PATTERNS["rub"].search(text) and CURRENCY_PATTERNS["kzt"].search(text))
    has_need = bool(NEED_KZT_RE.search(text) or NEED_RUB_RE.search(text))
    return has_both_currencies and has_need


def listing_direction(text: str) -> str:
    who_needs = WHO_NEEDS_RE.search(text)
    if who_needs:
        need = who_needs.group("need")
        if CURRENCY_PATTERNS["kzt"].search(need):
            return "rub_to_kzt"
        if CURRENCY_PATTERNS["rub"].search(need):
            return "kzt_to_rub"
    if NEED_KZT_RE.search(text) and OFFER_RUB_RE.search(text):
        return "rub_to_kzt"
    if NEED_RUB_RE.search(text) and OFFER_KZT_RE.search(text):
        return "kzt_to_rub"
    if NEED_KZT_RE.search(text):
        return "need_kzt_offer_unknown"
    if NEED_RUB_RE.search(text):
        return "need_rub_offer_unknown"

    search = SEARCH_RE.search(text)
    offer = OFFER_RE.search(text)
    if not search:
        return "unclassified"
    search_end = offer.start() if offer and offer.start() > search.end() else min(len(text), search.end() + 220)
    need = text[search.end() : search_end]
    need_currencies = detect_currency(need)

    offered_currencies: set[str] = set()
    if offer:
        offered = text[offer.end() : min(len(text), offer.end() + 220)]
        offered_currencies = detect_currency(offered)

    if "kzt" in need_currencies and "rub" in offered_currencies:
        return "rub_to_kzt"
    if "rub" in need_currencies and "kzt" in offered_currencies:
        return "kzt_to_rub"
    if "kzt" in need_currencies:
        return "need_kzt_offer_unknown"
    if "rub" in need_currencies:
        return "need_rub_offer_unknown"
    if "usd" in need_currencies:
        return "need_usd"
    if "eur" in need_currencies:
        return "need_eur"
    return "unclassified"


def parse_number(raw: str) -> float | None:
    compact = raw.replace("\u00a0", " ").strip().lower()
    multiplier = 1.0
    if compact.endswith(("к", "k")):
        multiplier = 1000.0
        compact = compact[:-1].strip()
    compact = compact.replace(" ", "").replace(",", ".")
    try:
        return float(compact) * multiplier
    except ValueError:
        return None


def extract_amounts(text: str) -> tuple[list[float], list[float]]:
    rub_values: list[float] = []
    kzt_values: list[float] = []
    for raw_number, raw_currency in AMOUNT_RE.findall(text):
        value = parse_number(raw_number)
        if value is None:
            continue
        currency = raw_currency.casefold()
        if "руб" in currency or "₽" in currency or "rub" in currency:
            if 100 <= value <= 20_000_000:
                rub_values.append(value)
        else:
            if 1_000 <= value <= 500_000_000:
                kzt_values.append(value)
    return rub_values, kzt_values


def extract_rate(text: str) -> float | None:
    match = RATE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    if 3.0 <= value <= 10.0:
        return value
    return None


def load_kzt_rates(path: Path | None) -> dict[str, float]:
    if not path:
        return {}
    rates: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["pair"] == "RUB/KZT":
                rates[row["date"]] = float(row["quote_per_rub"])
    return rates


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denominator


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        for index in range(position, end):
            result[indexed[index][0]] = average_rank
        position = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return pearson(rank(xs), rank(ys))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path: Path, rates_path: Path | None, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    totals: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    text_shapes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_authors: dict[str, set[str]] = defaultdict(set)
    bank_counts: Counter[str] = Counter()
    bank_authors: dict[str, set[str]] = defaultdict(set)
    directions: Counter[str] = Counter()
    direction_authors: dict[str, set[str]] = defaultdict(set)
    author_counts: Counter[str] = Counter()
    author_months: dict[str, set[str]] = defaultdict(set)
    author_first: dict[str, str] = {}
    author_last: dict[str, str] = {}
    strict_listing_authors: Counter[str] = Counter()
    monthly: dict[str, Counter[str]] = defaultdict(Counter)
    monthly_authors: dict[str, set[str]] = defaultdict(set)
    daily: dict[str, Counter[str]] = defaultdict(Counter)
    daily_authors: dict[str, set[str]] = defaultdict(set)
    weekday: dict[int, Counter[str]] = defaultdict(Counter)
    hour: dict[int, Counter[str]] = defaultdict(Counter)
    duplicate_hashes: Counter[bytes] = Counter()
    seen_ids: set[int] = set()
    listing_meta: dict[int, int] = {}
    listing_reply_count: Counter[int] = Counter()
    listing_first_reply_seconds: dict[int, int] = {}
    completion_edit_seconds: list[float] = []
    rub_amounts: list[float] = []
    kzt_amounts: list[float] = []
    explicit_rates: list[float] = []
    explicit_rate_market_gaps: list[float] = []
    explicit_rate_by_month: dict[str, list[float]] = defaultdict(list)
    duplicate_id_count = 0
    out_of_order = 0
    previous_timestamp: int | None = None
    first_date: str | None = None
    last_date: str | None = None
    min_id: int | None = None
    max_id: int | None = None
    kzt_market_rates = load_kzt_rates(rates_path)

    for message in iter_messages(input_path):
        totals["events"] += 1
        key_counts.update(message.keys())
        message_type = message.get("type", "unknown")
        totals[f"type:{message_type}"] += 1
        message_id = message.get("id")
        if isinstance(message_id, int):
            if message_id in seen_ids:
                duplicate_id_count += 1
            seen_ids.add(message_id)
            min_id = message_id if min_id is None else min(min_id, message_id)
            max_id = message_id if max_id is None else max(max_id, message_id)

        date_text = message.get("date")
        if not isinstance(date_text, str):
            totals["missing_date"] += 1
            continue
        try:
            dt = datetime.fromisoformat(date_text)
        except ValueError:
            totals["bad_date"] += 1
            continue
        timestamp = int(message.get("date_unixtime", int(dt.replace(tzinfo=timezone.utc).timestamp())))
        if previous_timestamp is not None and timestamp < previous_timestamp:
            out_of_order += 1
        previous_timestamp = timestamp
        first_date = date_text if first_date is None else min(first_date, date_text)
        last_date = date_text if last_date is None else max(last_date, date_text)

        month = month_key(dt)
        day = day_key(dt)
        monthly[month]["events"] += 1
        daily[day]["events"] += 1
        weekday[dt.weekday()]["events"] += 1
        hour[dt.hour]["events"] += 1

        if message_type != "message":
            continue
        totals["messages"] += 1
        text_value = message.get("text")
        text_shapes[type(text_value).__name__] += 1
        text = flatten_text(text_value).strip()
        author = str(message.get("from_id") or "missing")
        if author == "missing":
            totals["missing_author"] += 1
        author_counts[author] += 1
        author_months[author].add(month)
        author_first.setdefault(author, day)
        author_last[author] = day
        monthly_authors[month].add(author)
        daily_authors[day].add(author)
        monthly[month]["messages"] += 1
        daily[day]["messages"] += 1
        weekday[dt.weekday()]["messages"] += 1
        hour[dt.hour]["messages"] += 1

        if "reply_to_message_id" in message:
            totals["replies"] += 1
            target = message.get("reply_to_message_id")
            if isinstance(target, int) and target not in seen_ids:
                totals["missing_reply_target"] += 1
            if isinstance(target, int) and target in listing_meta:
                delay = timestamp - listing_meta[target]
                if delay >= 0:
                    listing_reply_count[target] += 1
                    listing_first_reply_seconds.setdefault(target, delay)
        if "edited" in message:
            totals["edited"] += 1
        if "reactions" in message:
            totals["with_reactions"] += 1
        if any(key in message for key in ("media_type", "photo", "file")):
            totals["with_media"] += 1

        if not text:
            totals["empty_text_messages"] += 1
            continue
        totals["text_messages"] += 1
        monthly[month]["text_messages"] += 1
        daily[day]["text_messages"] += 1
        normalized = normalized_text(text)
        if len(normalized) >= 10:
            digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
            duplicate_hashes[digest] += 1

        is_strict = strict_listing(text)
        is_probable = probable_listing(text)
        if is_strict:
            totals["strict_listings"] += 1
            monthly[month]["strict_listings"] += 1
            daily[day]["strict_listings"] += 1
            weekday[dt.weekday()]["strict_listings"] += 1
            hour[dt.hour]["strict_listings"] += 1
            strict_listing_authors[author] += 1
            if isinstance(message_id, int):
                listing_meta[message_id] = timestamp
            direction = listing_direction(text)
            directions[direction] += 1
            direction_authors[direction].add(author)
            monthly[month][f"direction:{direction}"] += 1
            daily[day][f"direction:{direction}"] += 1
            rub_values, kzt_values = extract_amounts(text)
            rub_amounts.extend(rub_values[:1])
            kzt_amounts.extend(kzt_values[:1])
            rate = extract_rate(text)
            if rate is not None:
                explicit_rates.append(rate)
                explicit_rate_by_month[month].append(rate)
                daily[day]["explicit_rates"] += 1
                market_rate = kzt_market_rates.get(day)
                if market_rate:
                    explicit_rate_market_gaps.append((rate / market_rate - 1.0) * 100.0)
            if CATEGORY_PATTERNS["successful_close"].search(text) and "edited_unixtime" in message:
                try:
                    edit_delay = int(message["edited_unixtime"]) - timestamp
                except (TypeError, ValueError):
                    edit_delay = -1
                if 0 <= edit_delay <= 14 * 24 * 3600:
                    completion_edit_seconds.append(edit_delay)
        if is_probable:
            totals["probable_listings"] += 1
            monthly[month]["probable_listings"] += 1
            daily[day]["probable_listings"] += 1

        for category, pattern in CATEGORY_PATTERNS.items():
            if pattern.search(text):
                category_counts[category] += 1
                category_authors[category].add(author)
                monthly[month][f"category:{category}"] += 1
                daily[day][f"category:{category}"] += 1
        for bank, pattern in BANK_PATTERNS.items():
            if pattern.search(text):
                bank_counts[bank] += 1
                bank_authors[bank].add(author)
                monthly[month][f"bank:{bank}"] += 1

    total_hashed_messages = sum(duplicate_hashes.values())
    repeated_message_instances = sum(count for count in duplicate_hashes.values() if count > 1)
    repeated_excess = sum(count - 1 for count in duplicate_hashes.values() if count > 1)

    author_bins = [
        ("1-3", 1, 3),
        ("4-20", 4, 20),
        ("21-99", 21, 99),
        ("100-999", 100, 999),
        ("1000+", 1000, math.inf),
    ]
    author_bin_rows: list[dict[str, Any]] = []
    for label, lower, upper in author_bins:
        selected = [count for count in author_counts.values() if lower <= count <= upper]
        author_bin_rows.append(
            {
                "bin": label,
                "authors": len(selected),
                "author_share": safe_ratio(len(selected), len(author_counts)),
                "messages": sum(selected),
                "message_share": safe_ratio(sum(selected), totals["messages"]),
            }
        )

    sorted_author_counts = sorted(author_counts.values(), reverse=True)
    concentration = {}
    for share in (0.001, 0.01, 0.05, 0.10):
        n = max(1, math.ceil(len(sorted_author_counts) * share))
        concentration[f"top_{share:.3f}_authors"] = {
            "authors": n,
            "message_share": safe_ratio(sum(sorted_author_counts[:n]), totals["messages"]),
        }

    reply_delays_minutes = [seconds / 60.0 for seconds in listing_first_reply_seconds.values() if seconds <= 7 * 24 * 3600]
    completion_minutes = [seconds / 60.0 for seconds in completion_edit_seconds]
    listing_reply_summary = {
        "strict_listings": totals["strict_listings"],
        "with_direct_reply": len(listing_reply_count),
        "direct_reply_rate": safe_ratio(len(listing_reply_count), totals["strict_listings"]),
        "first_direct_reply_minutes": distribution(reply_delays_minutes),
        "direct_replies_per_replied_listing": distribution([float(v) for v in listing_reply_count.values()]),
        "successful_edited_listings": len(completion_minutes),
        "completion_edit_minutes_proxy": distribution(completion_minutes),
    }

    category_rows = [
        {
            "category": category,
            "messages": count,
            "share_of_text_messages": safe_ratio(count, totals["text_messages"]),
            "authors": len(category_authors[category]),
            "share_of_authors": safe_ratio(len(category_authors[category]), len(author_counts)),
        }
        for category, count in category_counts.most_common()
    ]
    bank_rows = [
        {
            "entity": bank,
            "messages": count,
            "share_of_text_messages": safe_ratio(count, totals["text_messages"]),
            "authors": len(bank_authors[bank]),
        }
        for bank, count in bank_counts.most_common()
    ]
    direction_rows = [
        {
            "direction": direction,
            "messages": count,
            "share_of_strict_listings": safe_ratio(count, totals["strict_listings"]),
            "authors": len(direction_authors[direction]),
        }
        for direction, count in directions.most_common()
    ]

    all_month_fields = sorted({field for values in monthly.values() for field in values})
    monthly_rows = []
    for month, values in sorted(monthly.items()):
        row: dict[str, Any] = {"month": month, "unique_authors": len(monthly_authors[month])}
        row.update({field: values.get(field, 0) for field in all_month_fields})
        monthly_rows.append(row)

    all_daily_fields = sorted({field for values in daily.values() for field in values})
    daily_rows: list[dict[str, Any]] = []
    sorted_rate_days = sorted(kzt_market_rates)
    prior_rate: float | None = None
    rolling_rates: list[float] = []
    rate_position = 0
    for day, values in sorted(daily.items()):
        rate = kzt_market_rates.get(day)
        while rate_position < len(sorted_rate_days) and sorted_rate_days[rate_position] <= day:
            rolling_rates.append(kzt_market_rates[sorted_rate_days[rate_position]])
            if len(rolling_rates) > 90:
                rolling_rates.pop(0)
            rate_position += 1
        row = {"date": day, "unique_authors": len(daily_authors[day])}
        row.update({field: values.get(field, 0) for field in all_daily_fields})
        row["kzt_per_rub"] = rate
        if rate is not None and prior_rate is not None:
            row["market_return_1d"] = rate / prior_rate - 1.0
            row["market_abs_return_1d"] = abs(row["market_return_1d"])
        else:
            row["market_return_1d"] = None
            row["market_abs_return_1d"] = None
        if rate is not None and rolling_rates:
            row["market_90d_percentile"] = sum(value <= rate for value in rolling_rates) / len(rolling_rates)
        else:
            row["market_90d_percentile"] = None
        prior_rate = rate if rate is not None else prior_rate
        messages = values.get("messages", 0)
        strict = values.get("strict_listings", 0)
        r2k = values.get("direction:rub_to_kzt", 0) + values.get("direction:need_kzt_offer_unknown", 0)
        k2r = values.get("direction:kzt_to_rub", 0) + values.get("direction:need_rub_offer_unknown", 0)
        row["strict_listing_share"] = safe_ratio(strict, messages)
        row["rate_information_share"] = safe_ratio(values.get("category:rate_information", 0), messages)
        row["timing_forecast_share"] = safe_ratio(values.get("category:timing_forecast", 0), messages)
        row["urgency_share_in_strict"] = safe_ratio(values.get("category:urgency", 0), strict)
        row["rub_to_kzt_share_classified"] = safe_ratio(r2k, r2k + k2r)
        row["direction_net"] = safe_ratio(r2k - k2r, r2k + k2r)
        daily_rows.append(row)

    correlation_specs = {
        "abs_return_vs_rate_information_share": ("market_abs_return_1d", "rate_information_share"),
        "abs_return_vs_timing_forecast_share": ("market_abs_return_1d", "timing_forecast_share"),
        "rate_percentile_vs_strict_listing_share": ("market_90d_percentile", "strict_listing_share"),
        "rate_percentile_vs_rub_to_kzt_share": ("market_90d_percentile", "rub_to_kzt_share_classified"),
        "market_return_vs_direction_net": ("market_return_1d", "direction_net"),
    }
    correlations: dict[str, Any] = {}
    for name, (x_field, y_field) in correlation_specs.items():
        pairs = [
            (float(row[x_field]), float(row[y_field]))
            for row in daily_rows
            if row.get(x_field) is not None and row.get(y_field) is not None
        ]
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        correlations[name] = {
            "n_days": len(pairs),
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
        }

    eligible_event_rows = [
        row
        for row in daily_rows
        if row.get("market_abs_return_1d") is not None and row.get("messages", 0) >= 20
    ]
    abs_return_cutoff = percentile([row["market_abs_return_1d"] for row in eligible_event_rows], 0.90)
    event_study: dict[str, Any] = {"abs_return_p90_cutoff": abs_return_cutoff}
    if abs_return_cutoff is not None:
        for label, selected in {
            "high_volatility_days": [row for row in eligible_event_rows if row["market_abs_return_1d"] >= abs_return_cutoff],
            "other_days": [row for row in eligible_event_rows if row["market_abs_return_1d"] < abs_return_cutoff],
        }.items():
            event_study[label] = {
                "n_days": len(selected),
                "mean_rate_information_share": statistics.fmean(row["rate_information_share"] for row in selected),
                "mean_timing_forecast_share": statistics.fmean(row["timing_forecast_share"] for row in selected),
                "mean_strict_listing_share": statistics.fmean(row["strict_listing_share"] for row in selected),
            }

    first_month = first_date[:7] if first_date else None
    last_month = last_date[:7] if last_date else None
    id_span = (max_id - min_id + 1) if min_id is not None and max_id is not None else None
    summary = {
        "source": {
            "file": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "period": {"first": first_date, "last": last_date},
            "partial_boundary_months": [first_month, last_month],
            "events": totals["events"],
            "messages": totals["messages"],
            "service_events": totals["type:service"],
            "authors": len(author_counts),
            "text_messages": totals["text_messages"],
            "empty_text_messages": totals["empty_text_messages"],
            "id_min": min_id,
            "id_max": max_id,
            "id_span": id_span,
            "id_gaps_or_unexported": id_span - len(seen_ids) if id_span is not None else None,
            "duplicate_ids": duplicate_id_count,
            "out_of_order_timestamps": out_of_order,
            "replies": totals["replies"],
            "missing_reply_targets": totals["missing_reply_target"],
            "missing_reply_target_share": safe_ratio(totals["missing_reply_target"], totals["replies"]),
            "edited": totals["edited"],
            "with_media": totals["with_media"],
            "with_reactions": totals["with_reactions"],
            "text_shapes": dict(text_shapes),
            "key_counts": dict(sorted(key_counts.items())),
        },
        "marketplace": {
            "reporting_status": "UNVALIDATED BROAD AUTOMATED CANDIDATES; do not use as P2P prevalence. Use data/p2p_validation.json instead.",
            "strict_listing_definition": "broad automated two-sided RUB/KZT offer/need or explicit pair exchange; not precision-validated",
            "strict_listings": totals["strict_listings"],
            "strict_listing_share_of_text": safe_ratio(totals["strict_listings"], totals["text_messages"]),
            "strict_listing_authors": len(strict_listing_authors),
            "probable_listing_definition": "RUB and KZT markers plus an explicit need marker; includes strict listings",
            "probable_listings": totals["probable_listings"],
            "probable_listing_share_of_text": safe_ratio(totals["probable_listings"], totals["text_messages"]),
            "directions": direction_rows,
            "amounts_rub": distribution(rub_amounts),
            "amounts_kzt": distribution(kzt_amounts),
            "explicit_listing_rates_kzt_per_rub": distribution(explicit_rates),
            "listing_rate_gap_vs_oxr_percent": distribution(explicit_rate_market_gaps),
            "response_and_close_proxies": listing_reply_summary,
        },
        "participation": {
            "author_bins": author_bin_rows,
            "concentration": concentration,
            "authors_active_multiple_months": sum(len(months) >= 2 for months in author_months.values()),
            "authors_active_6plus_months": sum(len(months) >= 6 for months in author_months.values()),
            "listing_author_bins": {
                "1": sum(count == 1 for count in strict_listing_authors.values()),
                "2-5": sum(2 <= count <= 5 for count in strict_listing_authors.values()),
                "6-20": sum(6 <= count <= 20 for count in strict_listing_authors.values()),
                "21-99": sum(21 <= count <= 99 for count in strict_listing_authors.values()),
                "100+": sum(count >= 100 for count in strict_listing_authors.values()),
            },
        },
        "categories": category_rows,
        "banks_and_rails": bank_rows,
        "duplicate_content": {
            "messages_hashed": total_hashed_messages,
            "instances_belonging_to_repeated_text": repeated_message_instances,
            "share_instances_belonging_to_repeated_text": safe_ratio(repeated_message_instances, total_hashed_messages),
            "repeat_excess_after_first_instance": repeated_excess,
            "share_repeat_excess": safe_ratio(repeated_excess, total_hashed_messages),
            "unique_normalized_text_hashes": len(duplicate_hashes),
        },
        "market_relationships": {
            "reference": "Open Exchange Rates RUB/KZT quote_per_rub, end-of-day UTC",
            "correlations": correlations,
            "event_study": event_study,
        },
    }

    with (output / "summary.json").open("w", encoding="utf-8") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
        target.write("\n")

    write_csv(output / "monthly.csv", monthly_rows, ["month", "unique_authors", *all_month_fields])
    daily_fieldnames = list(daily_rows[0]) if daily_rows else ["date"]
    write_csv(output / "daily.csv", daily_rows, daily_fieldnames)
    write_csv(
        output / "categories.csv",
        category_rows,
        ["category", "messages", "share_of_text_messages", "authors", "share_of_authors"],
    )
    write_csv(
        output / "banks_and_rails.csv",
        bank_rows,
        ["entity", "messages", "share_of_text_messages", "authors"],
    )
    write_csv(
        output / "directions.csv",
        direction_rows,
        ["direction", "messages", "share_of_strict_listings", "authors"],
    )
    write_csv(
        output / "author_bins.csv",
        author_bin_rows,
        ["bin", "authors", "author_share", "messages", "message_share"],
    )
    write_csv(
        output / "weekday.csv",
        [
            {"weekday": index, "messages": weekday[index]["messages"], "strict_listings": weekday[index]["strict_listings"]}
            for index in range(7)
        ],
        ["weekday", "messages", "strict_listings"],
    )
    write_csv(
        output / "hour.csv",
        [
            {"hour": index, "messages": hour[index]["messages"], "strict_listings": hour[index]["strict_listings"]}
            for index in range(24)
        ],
        ["hour", "messages", "strict_listings"],
    )
    write_csv(
        output / "listing_rates_monthly.csv",
        [
            {"month": month, **distribution(values)}
            for month, values in sorted(explicit_rate_by_month.items())
        ],
        ["month", "n", "p10", "p25", "median", "p75", "p90"],
    )


def main() -> None:
    args = parse_args()
    analyze(args.input, args.rates, args.output)


if __name__ == "__main__":
    main()
