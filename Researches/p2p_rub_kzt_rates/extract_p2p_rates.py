#!/usr/bin/env python3
"""Extract privacy-safe RUB/KZT P2P rate observations from Telegram exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator


PIPELINE_VERSION = "1.1.1"
TARGET_SEGMENTS = ("card_transfer", "cash", "crypto")
TARGET_DIRECTIONS = ("rub_to_kzt", "kzt_to_rub")

SPACE_RE = re.compile(r"\s+")
RUB_TOKEN = r"(?:\brub\b|руб(?:л(?:ей|я|и|ь|ю|ём|ем)?|\.)?|₽)"
KZT_TOKEN = r"(?:\bkzt\b|\bкзт\b|тенг(?:е)?|₸|\bтг\b)"
RUB_RE = re.compile(RUB_TOKEN, re.IGNORECASE)
KZT_RE = re.compile(KZT_TOKEN, re.IGNORECASE)
OTHER_FIAT_RE = re.compile(
    r"(?iu)(?:\busd\b|\beur\b|\bcny\b|доллар|евро|юан|\$|€|¥)"
)
CRYPTO_RE = re.compile(
    r"(?iu)(?:крипт|\busd[tc]\b|юсд[тс]|tether|\bbtc\b|биткоин|\beth\b|"
    r"binance|бинанс|bybit|байбит|trc\s*20|erc\s*20|кошел[её]к|стейблкоин)"
)
CASH_RE = re.compile(
    r"(?iu)(?:налич|\bнал(?:ом|ик|а)?\b|обменник|обменный\s+пункт|личн(?:ая|ой)\s+встреч|"
    r"встреча|курьер|офис|купюр|кэш\b|cash\b|из\s+рук\s+в\s+руки)"
)
CARD_RE = re.compile(
    r"(?iu)(?:карт|\bсбп\b|безнал|перевод|на\s+сч[её]т|по\s+номеру\s+телефон|"
    r"kaspi|каспи|сбер|sber|тинькофф|tinkoff|т-?банк|halyk|халык|bcc|бцк|"
    r"freedom|фридом|forte|форте|jusan|жусан|alatau|алатау|альфа|alfa|втб)"
)
TRANSACTION_RE = re.compile(
    r"(?iu)(?:курс|обмен|помен|меняю|меня[йе]те|куплю|покупаю|покупка|продам|"
    r"продаю|продажа|ищу|нужн|отдам|предлагаю|расч[её]т|котиров|\bпо\s+\d)"
)

NUMBER = r"(?:\d{1,2}(?:[.,]\d{1,4})?)"
RATE_LABEL_RE = re.compile(
    rf"(?iu)\bкурс\w*(?:\s+(?:рубл\w*|тенг\w*|покупк\w*|продаж\w*|"
    rf"обмен\w*|сейчас|сегодня|будет|равен|составля\w*)){{0,4}}\s*"
    rf"(?:[:=\-–—]|по)?\s*(?P<value>{NUMBER})(?![.,]\d|\s*%)"
)
PO_RATE_RE = re.compile(rf"(?iu)\bпо\s+(?P<value>{NUMBER})(?![.,]\d|\s*[%кk])")
RUB_EQUALITY_RE = re.compile(
    rf"(?iu)(?:1\s*)?{RUB_TOKEN}.{{0,28}}?(?:=|это|да(?:ю|ют|м)|за|по)?\s*"
    rf"(?P<value>{NUMBER})\s*{KZT_TOKEN}"
)
KZT_EQUALITY_RE = re.compile(
    rf"(?iu)(?:1\s*)?{KZT_TOKEN}.{{0,28}}?(?:=|это|да(?:ю|ют|м)|за|по)?\s*"
    rf"(?P<value>0[.,]\d{{1,5}})\s*{RUB_TOKEN}"
)
BUY_SELL_RE = re.compile(
    rf"(?iu)(?P<label>покупк\w*|продаж\w*)[^\d]{{0,18}}(?P<value>{NUMBER})(?![.,]\d|\s*%)"
)
ROUTE_RESULT_RE = re.compile(
    rf"(?iu)\bкурс\w*\s+(?:в\s+итоге\s+)?(?:получил\w*|вышел)\s*"
    rf"(?:[:=\-–—]|по)?\s*(?P<value>{NUMBER})(?![.,]\d|\s*%)"
)
BARE_RATE_RE = re.compile(rf"^\s*(?P<value>{NUMBER})\s*[.!]?\s*$", re.IGNORECASE)

AMOUNT_RE = re.compile(
    rf"(?iu)(?P<number>\d{{1,3}}(?:[\s\u00a0]\d{{3}})+|\d+(?:[.,]\d+)?)\s*"
    rf"(?P<scale>млн|миллион\w*|тыс\w*|[кk])?\s*(?P<currency>{RUB_TOKEN}|{KZT_TOKEN})"
)

URL_RE = re.compile(r"(?iu)https?://\S+|www\.\S+|t\.me/\S+")
EMAIL_RE = re.compile(r"(?iu)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
HANDLE_RE = re.compile(r"(?u)(?<!\w)@[A-Za-z0-9_]{3,}")
PHONE_RE = re.compile(r"(?u)(?<!\d)(?:\+?\d[\s()\-]*){10,15}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?u)(?<!\d)\d{6,}(?!\d)")

NO_PAIR_CURRENCY = rf"(?:(?!{RUB_TOKEN}|{KZT_TOKEN}).)"
NEED_KZT_RE = re.compile(
    rf"(?iu)(?:ищу|нужн\w*|куплю|покупаю){NO_PAIR_CURRENCY}{{0,80}}{KZT_TOKEN}|"
    rf"{KZT_TOKEN}\s+(?:нужн\w*|куплю|покупаю)"
)
NEED_RUB_RE = re.compile(
    rf"(?iu)(?:ищу|нужн\w*|куплю|покупаю){NO_PAIR_CURRENCY}{{0,80}}{RUB_TOKEN}|"
    rf"{RUB_TOKEN}\s+(?:нужн\w*|куплю|покупаю)"
)
SELL_RUB_RE = re.compile(
    rf"(?iu)(?:продам|продаю|отдам|предлагаю){NO_PAIR_CURRENCY}{{0,80}}{RUB_TOKEN}|"
    rf"{RUB_TOKEN}\s+(?:продам|продаю|отдам)"
)
SELL_KZT_RE = re.compile(
    rf"(?iu)(?:продам|продаю|отдам|предлагаю){NO_PAIR_CURRENCY}{{0,80}}{KZT_TOKEN}|"
    rf"{KZT_TOKEN}\s+(?:продам|продаю|отдам)"
)
POSSESS_RUB_RE = re.compile(
    rf"(?iu)(?:у\s+меня|есть|имею|на\s+руках|остал\w*){NO_PAIR_CURRENCY}{{0,45}}{RUB_TOKEN}"
)
POSSESS_KZT_RE = re.compile(
    rf"(?iu)(?:у\s+меня|есть|имею|на\s+руках|остал\w*){NO_PAIR_CURRENCY}{{0,45}}{KZT_TOKEN}"
)
ARROW_RUB_KZT_RE = re.compile(rf"(?iu){RUB_TOKEN}\s*(?:→|->|>|на|в)\s*{KZT_TOKEN}")
ARROW_KZT_RUB_RE = re.compile(rf"(?iu){KZT_TOKEN}\s*(?:→|->|>|на|в)\s*{RUB_TOKEN}")
RECIPROCAL_RUB_TO_KZT_RE = re.compile(
    rf"(?iu)(?:вы|мне).{{0,35}}{KZT_TOKEN}.{{0,80}}(?:я|вам).{{0,35}}{RUB_TOKEN}|"
    rf"(?:я|вам).{{0,35}}{RUB_TOKEN}.{{0,80}}(?:вы|мне).{{0,35}}{KZT_TOKEN}"
)
RECIPROCAL_KZT_TO_RUB_RE = re.compile(
    rf"(?iu)(?:вы|мне).{{0,35}}{RUB_TOKEN}.{{0,80}}(?:я|вам).{{0,35}}{KZT_TOKEN}|"
    rf"(?:я|вам).{{0,35}}{KZT_TOKEN}.{{0,80}}(?:вы|мне).{{0,35}}{RUB_TOKEN}"
)
EXCHANGE_RUB_KZT_RE = re.compile(
    rf"(?iu)(?:{RUB_TOKEN}.{{0,80}}(?:поменя|обменя|меня)\w*.{{0,30}}(?:на|в).{{0,20}}{KZT_TOKEN}|"
    rf"(?:поменя|обменя|меня)\w*.{{0,40}}{RUB_TOKEN}.{{0,40}}(?:на|в).{{0,20}}{KZT_TOKEN})"
)
EXCHANGE_KZT_RUB_RE = re.compile(
    rf"(?iu)(?:{KZT_TOKEN}.{{0,80}}(?:поменя|обменя|меня)\w*.{{0,30}}(?:на|в).{{0,20}}{RUB_TOKEN}|"
    rf"(?:поменя|обменя|меня)\w*.{{0,40}}{KZT_TOKEN}.{{0,40}}(?:на|в).{{0,20}}{RUB_TOKEN})"
)
WHO_NEEDS_RUB_RE = re.compile(
    rf"(?iu)(?:кому|может\s+кому){NO_PAIR_CURRENCY}{{0,55}}"
    rf"нужн\w*{NO_PAIR_CURRENCY}{{0,30}}{RUB_TOKEN}"
)
WHO_NEEDS_KZT_RE = re.compile(
    rf"(?iu)(?:кому|может\s+кому){NO_PAIR_CURRENCY}{{0,55}}"
    rf"нужн\w*{NO_PAIR_CURRENCY}{{0,30}}{KZT_TOKEN}"
)
FLOW_KZT_RUB_RE = re.compile(
    rf"(?iu){KZT_TOKEN}.{{0,80}}(?:преврат\w*|получил\w*|пришл\w*).{{0,60}}{RUB_TOKEN}"
)
FLOW_RUB_KZT_RE = re.compile(
    rf"(?iu){RUB_TOKEN}.{{0,80}}(?:преврат\w*|получил\w*|пришл\w*).{{0,60}}{KZT_TOKEN}"
)
KZT_TO_RUSSIAN_CARD_RE = re.compile(
    rf"(?iu)(?:у\s+меня|есть).{{0,45}}{KZT_TOKEN}.{{0,160}}"
    r"(?:вывест|получ|перевест).{0,55}(?:российск|карт\w*\s+рф|сбер|тинькофф)"
)

PEER_EXCHANGE_RE = re.compile(
    rf"(?iu)(?:(?:#\s*)?(?:меняю|обменяю|поменяю|обменяюсь|поменяюсь|"
    rf"кто\s+(?:обменяет|поменяет)|(?:могу|готов(?:а)?)\s+(?:обменять|поменять)|"
    rf"есть\s+человек.{{0,30}}меняет)\b.{{0,220}}"
    rf"(?:{RUB_TOKEN}.{{0,120}}{KZT_TOKEN}|{KZT_TOKEN}.{{0,120}}{RUB_TOKEN})|"
    rf"(?:кому|может\s+кому).{{0,70}}(?:нужн|надо).{{0,80}}"
    rf"(?:{RUB_TOKEN}|{KZT_TOKEN}))"
)
PEER_COMPLETED_RE = re.compile(
    r"(?iu)(?:обменял(?:ся|ась|ись)|поменял(?:ся|ась|ись)|"
    r"(?:обменял|поменял|менял\w*).{0,45}(?:у|с)\s+(?:человек|знаком|менял)|"
    r"через\s+(?:человек|знаком)|у\s+менял\w*|с\s+рук|"
    r"(?:местные|люди|человек|знаком\w*).{0,35}(?:меняю|обменива)|"
    r"p2p\s*сделк|сделка\s+(?:закрыта|состоялась))"
)
CRYPTO_ROUTE_RE = re.compile(
    r"(?iu)(?:через\s+(?:крипт|usdt|binance|бинанс|bybit|байбит)|курс\s+(?:вышел|получился)|"
    r"по\s+итогу.{0,80}(?:получил|пришло)|конвертировал.{0,80}(?:usdt|крипт))"
)
CRYPTO_RUB_TO_KZT_RE = re.compile(
    rf"(?iu)(?:покуп\w*.{{0,60}}за\s+{RUB_TOKEN}.{{0,100}}прода\w*.{{0,60}}(?:за|в)\s+{KZT_TOKEN}|"
    rf"{RUB_TOKEN}.{{0,60}}(?:через\s+крипт|через\s+usdt).{{0,100}}{KZT_TOKEN}|"
    rf"(?:через\s+(?:крипт|usdt|binance|бинанс|bybit|байбит)).{{0,160}}"
    rf"(?:отда\w*.{{0,30}})?{RUB_TOKEN}.{{0,220}}{KZT_TOKEN})"
)
KZT_RECEIVED_PER_RUB_RE = re.compile(
    rf"(?iu){KZT_TOKEN}.{{0,70}}(?:пришл\w*|зачисл\w*).{{0,100}}"
    rf"(?:поделил\w*.{{0,60}})?(?:сумм\w*.{{0,30}})?{RUB_TOKEN}"
)
APPROXIMATE_EQUIVALENCE_RE = re.compile(
    r"(?iu)(?:эквивалент|приблизитель|примерно\s+\d|около\s+\d|ориентировочн)"
)
CONTEXT_OVERRIDE_RE = re.compile(
    r"(?iu)(?:обменник|обменный\s+пункт|банк|приложени|корсч[её]т|банкомат|курс\s+мир)"
)
INSTITUTIONAL_REFERENCE_RE = re.compile(
    r"(?iu)(?:цб\s*(?:кз|рф)?|нацбанк|таблиц\w*\s+курс|бот\s+с\s+курс|"
    r"продать\s+1\s*rub\s+за\s+kzt|корсч[её]т|kztrub|bcc\s+fx|"
    r"курс\s+(?:банка|карты|автоконвертац)|в\s+приложени\w*.{0,30}курс)"
)


@dataclass
class MessageContext:
    text: str
    has_rub: bool
    has_kzt: bool
    has_transaction: bool


@dataclass
class Observation:
    source_file: str
    source_chat: str
    message_ref: str
    participant_key: str
    timestamp: str
    day: str
    segment: str
    market_scope: str
    direction: str
    rate_kzt_per_rub: float
    extraction_method: str
    pair_basis: str
    confidence_score: float
    confidence: str
    quality_status: str
    official_rate_kzt_per_rub: float | None
    deviation_from_official_pct: float | None
    evidence_excerpt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--official-rates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-per-stratum", type=int, default=20)
    return parser.parse_args()


def iter_messages(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Потоково читает объекты верхнеуровневого массива messages."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        buffer = ""
        while '"messages"' not in buffer:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError(f"messages array not found in {path}")
            buffer += chunk
        marker = buffer.index('"messages"')
        start = buffer.find("[", marker)
        while start < 0:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError(f"messages array opening bracket not found in {path}")
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
                    raise ValueError(f"unexpected EOF inside messages array in {path}")
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
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def redact_text(text: str, limit: int = 320) -> str:
    redacted = URL_RE.sub("<URL>", text)
    redacted = EMAIL_RE.sub("<EMAIL>", redacted)
    redacted = HANDLE_RE.sub("<HANDLE>", redacted)
    redacted = PHONE_RE.sub("<PHONE>", redacted)
    redacted = LONG_NUMBER_RE.sub("<NUMBER>", redacted)
    return SPACE_RE.sub(" ", redacted).strip()[:limit]


def parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(" ", "").replace("\u00a0", "").replace(",", "."))
    except ValueError:
        return None


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_chat_name(path: Path) -> str:
    with path.open("r", encoding="utf-8") as source:
        header = source.read(8192)
    match = re.search(r'^\s*\{\s*"name"\s*:\s*"(?P<name>(?:[^"\\]|\\.)*)"', header)
    if not match:
        return path.stem
    return json.loads(f'"{match.group("name")}"')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_rates(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("pair") == "RUB/KZT":
                result[row["date"]] = float(row["quote_per_rub"])
    return result


def classify_segment(text: str) -> str:
    if CRYPTO_RE.search(text):
        return "crypto"
    if CASH_RE.search(text):
        return "cash"
    if CARD_RE.search(text):
        return "card_transfer"
    return "unspecified"


def infer_direction(text: str) -> str:
    if CRYPTO_RUB_TO_KZT_RE.search(text):
        return "rub_to_kzt"
    if KZT_RECEIVED_PER_RUB_RE.search(text):
        return "rub_to_kzt"
    if FLOW_RUB_KZT_RE.search(text):
        return "rub_to_kzt"
    if FLOW_KZT_RUB_RE.search(text):
        return "kzt_to_rub"
    if KZT_TO_RUSSIAN_CARD_RE.search(text):
        return "kzt_to_rub"
    if ARROW_RUB_KZT_RE.search(text):
        return "rub_to_kzt"
    if ARROW_KZT_RUB_RE.search(text):
        return "kzt_to_rub"
    if RECIPROCAL_RUB_TO_KZT_RE.search(text):
        return "rub_to_kzt"
    if RECIPROCAL_KZT_TO_RUB_RE.search(text):
        return "kzt_to_rub"
    if WHO_NEEDS_RUB_RE.search(text) and not WHO_NEEDS_KZT_RE.search(text):
        return "rub_to_kzt"
    if WHO_NEEDS_KZT_RE.search(text) and not WHO_NEEDS_RUB_RE.search(text):
        return "kzt_to_rub"
    if EXCHANGE_RUB_KZT_RE.search(text):
        return "rub_to_kzt"
    if EXCHANGE_KZT_RUB_RE.search(text):
        return "kzt_to_rub"
    if POSSESS_RUB_RE.search(text) and NEED_KZT_RE.search(text):
        return "rub_to_kzt"
    if POSSESS_KZT_RE.search(text) and NEED_RUB_RE.search(text):
        return "kzt_to_rub"
    rub_to_kzt = bool(NEED_KZT_RE.search(text) or SELL_RUB_RE.search(text))
    kzt_to_rub = bool(NEED_RUB_RE.search(text) or SELL_KZT_RE.search(text))
    if rub_to_kzt and not kzt_to_rub:
        return "rub_to_kzt"
    if kzt_to_rub and not rub_to_kzt:
        return "kzt_to_rub"
    return "unclassified"


def strict_peer_offer(text: str) -> bool:
    has_pair = bool(RUB_RE.search(text) and KZT_RE.search(text))
    if not has_pair:
        return False
    two_sided = bool(
        (NEED_KZT_RE.search(text) and (SELL_RUB_RE.search(text) or POSSESS_RUB_RE.search(text)))
        or (NEED_RUB_RE.search(text) and (SELL_KZT_RE.search(text) or POSSESS_KZT_RE.search(text)))
    )
    if re.search(r"(?iu)нужн\w*\s+был", text) and not re.search(
        r"(?iu)(?:кому|отдам|предлагаю|продам|обменяю|поменяю|пишите\s+в\s+лс)",
        text,
    ):
        two_sided = False
    return two_sided or bool(PEER_EXCHANGE_RE.search(text))


def classify_market_scope(text: str, segment: str) -> str:
    if strict_peer_offer(text):
        return "peer_offer"
    if PEER_COMPLETED_RE.search(text) and RUB_RE.search(text) and KZT_RE.search(text):
        return "peer_trade_report"
    if segment == "crypto" and CRYPTO_ROUTE_RE.search(text):
        return "crypto_route_report"
    if segment == "cash" and re.search(r"(?iu)(?:обменник|обменный\s+пункт)", text):
        return "cash_exchange_reference"
    if INSTITUTIONAL_REFERENCE_RE.search(text):
        return "institutional_reference"
    if re.search(
        r"(?iu)(?:переводил|перевела|получил|пришло|конверт|снял|снимал|поменял|обменял)",
        text,
    ):
        return "bank_or_card_execution"
    return "unclassified"


def amount_value(match: re.Match[str]) -> tuple[float, str] | None:
    value = parse_number(match.group("number"))
    if value is None:
        return None
    scale = (match.group("scale") or "").casefold()
    if scale.startswith(("млн", "миллион")):
        value *= 1_000_000
    elif scale.startswith(("тыс", "к", "k")):
        value *= 1_000
    currency_text = match.group("currency")
    currency = "rub" if RUB_RE.search(currency_text) else "kzt"
    return value, currency


def extract_amount_ratio(text: str) -> float | None:
    rub_values: list[float] = []
    kzt_values: list[float] = []
    for match in AMOUNT_RE.finditer(text):
        parsed = amount_value(match)
        if parsed is None:
            continue
        value, currency = parsed
        if currency == "rub" and 100 <= value <= 100_000_000:
            rub_values.append(value)
        if currency == "kzt" and 500 <= value <= 1_000_000_000:
            kzt_values.append(value)
    if len(rub_values) != 1 or len(kzt_values) != 1:
        return None
    return kzt_values[0] / rub_values[0]


def _rate_value(raw: str, inverse: bool = False) -> float | None:
    value = parse_number(raw)
    if value is None or value == 0:
        return None
    value = 1.0 / value if inverse else value
    if 2.5 <= value <= 12.0:
        return value
    return None


def extract_rate_quotes(text: str, parent_used: bool = False) -> list[tuple[float, str, str | None]]:
    """Возвращает rate, method и direction override для табличных buy/sell."""
    quotes: list[tuple[float, str, str | None]] = []
    occupied: set[tuple[int, int]] = set()

    def add(match: re.Match[str], method: str, inverse: bool = False, direction: str | None = None) -> None:
        value = _rate_value(match.group("value"), inverse=inverse)
        if value is None:
            return
        span = match.span("value")
        if span in occupied:
            return
        occupied.add(span)
        quotes.append((value, method, direction))

    for match in RUB_EQUALITY_RE.finditer(text):
        add(match, "currency_equality")
    for match in KZT_EQUALITY_RE.finditer(text):
        add(match, "inverse_currency_equality", inverse=True)
    for match in BUY_SELL_RE.finditer(text):
        label = match.group("label").casefold()
        direction = "rub_to_kzt" if label.startswith("покуп") else "kzt_to_rub"
        add(match, "buy_sell_table", direction=direction)
    for match in ROUTE_RESULT_RE.finditer(text):
        add(match, "route_result_rate")
    for match in RATE_LABEL_RE.finditer(text):
        add(match, "explicit_rate")
    for match in PO_RATE_RE.finditer(text):
        add(match, "po_rate")

    ratio = extract_amount_ratio(text)
    if ratio is not None and 2.5 <= ratio <= 12.0:
        quotes.append((ratio, "amount_ratio", None))

    if parent_used:
        bare = BARE_RATE_RE.fullmatch(text.strip())
        if bare:
            value = _rate_value(bare.group("value"))
            if value is not None:
                quotes.append((value, "bare_reply", None))

    unique: list[tuple[float, str, str | None]] = []
    seen: set[tuple[float, str | None]] = set()
    for value, method, direction in quotes:
        key = (round(value, 6), direction)
        if key not in seen:
            unique.append((value, method, direction))
            seen.add(key)
    return unique


def pair_basis(current: str, combined: str, parent: MessageContext | None, source_chat: str) -> str | None:
    if RUB_RE.search(current) and KZT_RE.search(current):
        return "direct_message"
    if parent and parent.has_rub and parent.has_kzt:
        return "reply_context"
    chat_is_kazakhstan = bool(re.search(r"(?iu)казахстан|банки\s+казахстан", source_chat))
    current_has_other = bool(OTHER_FIAT_RE.search(current) or CRYPTO_RE.search(current))
    if chat_is_kazakhstan and RUB_RE.search(combined) and not current_has_other:
        if re.search(r"(?iu)курс\w*\s+руб|рубл\w*\s+курс|по\s+\d", combined):
            return "chat_kzt_context"
    return None


def confidence_for(
    method: str,
    basis: str,
    segment: str,
    market_scope: str,
    direction: str,
    has_other_currency: bool,
    deviation: float | None,
) -> float:
    base = {
        "route_result_rate": 0.94,
        "currency_equality": 0.96,
        "inverse_currency_equality": 0.96,
        "amount_ratio": 0.90,
        "buy_sell_table": 0.88,
        "explicit_rate": 0.84,
        "po_rate": 0.78,
        "bare_reply": 0.76,
    }[method]
    if basis == "reply_context":
        base -= 0.05
    elif basis == "chat_kzt_context":
        base -= 0.15
    if segment == "unspecified":
        base -= 0.08
    if market_scope == "peer_offer":
        base += 0.05
    elif market_scope == "peer_trade_report":
        base += 0.02
    elif market_scope == "crypto_route_report":
        base += 0.01
    else:
        base -= 0.35
    if direction == "unclassified":
        base -= 0.12
    if has_other_currency and method not in {"currency_equality", "inverse_currency_equality", "amount_ratio"}:
        base -= 0.16
    if deviation is not None:
        absolute = abs(deviation)
        if absolute <= 0.15:
            base += 0.03
        elif absolute <= 0.30:
            base -= 0.04
        elif absolute <= 0.50:
            base -= 0.20
        else:
            base -= 0.45
    return min(0.99, max(0.0, base))


def quality_label(score: float, segment: str, market_scope: str, direction: str) -> tuple[str, str]:
    allowed_scope = market_scope in {"peer_offer", "peer_trade_report", "crypto_route_report"}
    if score >= 0.72 and segment in TARGET_SEGMENTS and direction in TARGET_DIRECTIONS and allowed_scope:
        status = "accepted"
    elif score >= 0.45:
        status = "review"
    else:
        status = "rejected"
    if score >= 0.85:
        label = "high"
    elif score >= 0.72:
        label = "medium"
    else:
        label = "low"
    return status, label


def message_ref(source_file: str, message_id: Any) -> str:
    return hashlib.sha256(f"{source_file}:{message_id}".encode()).hexdigest()[:16]


def participant_key(source_file: str, message: dict[str, Any]) -> str:
    raw = message.get("from_id") or message.get("actor_id") or f"message:{message.get('id')}"
    return hashlib.sha256(f"{source_file}:{raw}".encode()).hexdigest()


def analyze_source(
    path: Path,
    official_rates: dict[str, float],
) -> tuple[list[Observation], dict[str, Any], Counter[str]]:
    source_file = path.name
    source_chat = read_chat_name(path)
    contexts: dict[int, MessageContext] = {}
    observations: list[Observation] = []
    counters: Counter[str] = Counter()
    first_day: str | None = None
    last_day: str | None = None

    for message in iter_messages(path):
        counters["events"] += 1
        if message.get("type") != "message":
            counters["non_message_events"] += 1
            continue
        counters["messages"] += 1
        raw_text = flatten_text(message.get("text"))
        current = normalize_text(raw_text)
        if not current:
            counters["empty_text_messages"] += 1
            continue
        counters["text_messages"] += 1
        timestamp = message.get("date")
        if not isinstance(timestamp, str):
            counters["missing_timestamp"] += 1
            continue
        try:
            day = datetime.fromisoformat(timestamp).date().isoformat()
        except ValueError:
            counters["invalid_timestamp"] += 1
            continue
        first_day = day if first_day is None or day < first_day else first_day
        last_day = day if last_day is None or day > last_day else last_day

        reply_id = message.get("reply_to_message_id")
        parent = contexts.get(reply_id) if isinstance(reply_id, int) else None
        combined = current if parent is None else f"{current} || {parent.text}"
        basis = pair_basis(current, combined, parent, source_chat)
        parent_used = basis == "reply_context"
        quotes = extract_rate_quotes(current, parent_used=parent_used)

        if quotes:
            counters["messages_with_rate_shape"] += 1
            if basis is None:
                counters["rate_shape_without_pair"] += 1
            elif (
                not TRANSACTION_RE.search(combined)
                and not parent_used
                and not (CRYPTO_RE.search(current) and extract_amount_ratio(current) is not None)
            ):
                counters["rate_shape_without_transaction_context"] += 1
            else:
                short_contextual_reply = parent_used and (
                    len(current) <= 48 or bool(BARE_RATE_RE.fullmatch(current.strip()))
                ) and not CONTEXT_OVERRIDE_RE.search(current)
                analysis_text = combined if short_contextual_reply else current
                segment = classify_segment(analysis_text)
                market_scope = classify_market_scope(analysis_text, segment)
                inferred_direction = infer_direction(analysis_text)
                other_currency = bool(OTHER_FIAT_RE.search(analysis_text) or CRYPTO_RE.search(analysis_text))
                has_route_result = any(method == "route_result_rate" for _, method, _ in quotes)
                official = official_rates.get(day)
                for rate, method, direction_override in quotes:
                    if market_scope == "crypto_route_report" and inferred_direction in TARGET_DIRECTIONS:
                        direction = inferred_direction
                    else:
                        direction = direction_override or inferred_direction
                    deviation = rate / official - 1 if official else None
                    score = confidence_for(
                        method,
                        basis,
                        segment,
                        market_scope,
                        direction,
                        other_currency,
                        deviation,
                    )
                    if method == "amount_ratio" and APPROXIMATE_EQUIVALENCE_RE.search(analysis_text):
                        score = max(0.0, score - 0.35)
                    if has_route_result and market_scope == "crypto_route_report" and method == "explicit_rate":
                        score = max(0.0, score - 0.30)
                    status, label = quality_label(score, segment, market_scope, direction)
                    observations.append(
                        Observation(
                            source_file=source_file,
                            source_chat=source_chat,
                            message_ref=message_ref(source_file, message.get("id")),
                            participant_key=participant_key(source_file, message),
                            timestamp=timestamp,
                            day=day,
                            segment=segment,
                            market_scope=market_scope,
                            direction=direction,
                            rate_kzt_per_rub=rate,
                            extraction_method=method,
                            pair_basis=basis,
                            confidence_score=score,
                            confidence=label,
                            quality_status=status,
                            official_rate_kzt_per_rub=official,
                            deviation_from_official_pct=deviation,
                            evidence_excerpt=redact_text(analysis_text),
                        )
                    )
                    counters[f"observation:{status}"] += 1

        message_id = message.get("id")
        relevant_context = bool(
            RUB_RE.search(current)
            or KZT_RE.search(current)
            or TRANSACTION_RE.search(current)
            or CRYPTO_RE.search(current)
            or CASH_RE.search(current)
            or CARD_RE.search(current)
        )
        if isinstance(message_id, int) and relevant_context:
            contexts[message_id] = MessageContext(
                text=current[:1200],
                has_rub=bool(RUB_RE.search(current)),
                has_kzt=bool(KZT_RE.search(current)),
                has_transaction=bool(TRANSACTION_RE.search(current)),
            )

    metadata = {
        "source_file": source_file,
        "source_chat": source_chat,
        "size_bytes": path.stat().st_size,
        "first_message_day": first_day,
        "last_message_day": last_day,
        "sha256": file_sha256(path),
    }
    return observations, metadata, counters


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def public_observation_row(observation: Observation) -> dict[str, Any]:
    row = asdict(observation)
    row.pop("participant_key")
    row.pop("evidence_excerpt")
    row["rate_kzt_per_rub"] = round(observation.rate_kzt_per_rub, 6)
    row["confidence_score"] = round(observation.confidence_score, 4)
    if observation.official_rate_kzt_per_rub is not None:
        row["official_rate_kzt_per_rub"] = round(observation.official_rate_kzt_per_rub, 6)
    if observation.deviation_from_official_pct is not None:
        row["deviation_from_official_pct"] = round(observation.deviation_from_official_pct, 6)
    return row


def deduplicate_observations(observations: list[Observation]) -> list[Observation]:
    """Оставляет лучшую интерпретацию одной котировки одного сообщения."""
    method_priority = {
        "route_result_rate": 8,
        "amount_ratio": 7,
        "currency_equality": 6,
        "inverse_currency_equality": 6,
        "explicit_rate": 5,
        "buy_sell_table": 4,
        "po_rate": 3,
        "bare_reply": 2,
    }
    best: dict[tuple[str, str, float, str, str], Observation] = {}
    for observation in observations:
        key = (
            observation.source_file,
            observation.message_ref,
            round(observation.rate_kzt_per_rub, 8),
            observation.segment,
            observation.direction,
        )
        current = best.get(key)
        rank = (observation.confidence_score, method_priority[observation.extraction_method])
        current_rank = (
            (current.confidence_score, method_priority[current.extraction_method])
            if current is not None
            else (-1.0, -1)
        )
        if rank > current_rank:
            best[key] = observation
    return list(best.values())


def aggregate_daily(
    observations: list[Observation],
    first_day: str,
    last_day: str,
    official_rates: dict[str, float],
) -> list[dict[str, Any]]:
    method_priority = {
        "route_result_rate": 8,
        "amount_ratio": 7,
        "currency_equality": 6,
        "inverse_currency_equality": 6,
        "explicit_rate": 5,
        "buy_sell_table": 4,
        "po_rate": 3,
        "bare_reply": 2,
    }
    message_cells: dict[tuple[str, str, str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.quality_status != "accepted":
            continue
        key = (observation.message_ref, observation.segment, observation.direction, observation.participant_key)
        message_cells[key].append(observation)

    primary_messages: list[Observation] = []
    for items in message_cells.values():
        primary_messages.append(
            max(
                items,
                key=lambda item: (
                    method_priority[item.extraction_method],
                    item.confidence_score,
                    -items.index(item),
                ),
            )
        )

    author_cells: dict[tuple[str, str, str, str], list[Observation]] = defaultdict(list)
    for observation in primary_messages:
        key = (observation.day, observation.segment, observation.direction, observation.participant_key)
        author_cells[key].append(observation)

    daily_cells: dict[tuple[str, str, str], list[tuple[float, set[str]]]] = defaultdict(list)
    for (day, segment, direction, _participant), items in author_cells.items():
        selected = max(items, key=lambda item: item.timestamp)
        rate = selected.rate_kzt_per_rub
        sources = {selected.source_file}
        daily_cells[(day, segment, direction)].append((rate, sources))

    start = date.fromisoformat(first_day)
    end = date.fromisoformat(last_day)
    rows: list[dict[str, Any]] = []
    last_seen: dict[tuple[str, str], tuple[str, float]] = {}
    current = start
    while current <= end:
        day = current.isoformat()
        for segment in TARGET_SEGMENTS:
            for direction in TARGET_DIRECTIONS:
                cell = daily_cells.get((day, segment, direction), [])
                rates = [item[0] for item in cell]
                source_files = sorted({source for _, sources in cell for source in sources})
                observed = statistics.median(rates) if rates else None
                key = (segment, direction)
                if observed is not None:
                    last_seen[key] = (day, observed)
                    effective = observed
                    effective_day = day
                    age = 0
                    fill_method = "observed"
                elif key in last_seen:
                    effective_day, effective = last_seen[key]
                    age = (current - date.fromisoformat(effective_day)).days
                    fill_method = "forward_fill"
                else:
                    effective = None
                    effective_day = None
                    age = None
                    fill_method = "no_prior_observation"
                official = official_rates.get(day)
                premium = effective / official - 1 if effective is not None and official else None
                rows.append(
                    {
                        "date": day,
                        "segment": segment,
                        "direction": direction,
                        "is_observed": bool(rates),
                        "observed_rate_kzt_per_rub": round(observed, 6) if observed is not None else None,
                        "observed_p25": round(percentile(rates, 0.25), 6) if rates else None,
                        "observed_p75": round(percentile(rates, 0.75), 6) if rates else None,
                        "observed_min": round(min(rates), 6) if rates else None,
                        "observed_max": round(max(rates), 6) if rates else None,
                        "distinct_posters": len(rates),
                        "source_count": len(source_files),
                        "source_files": "|".join(source_files),
                        "effective_rate_kzt_per_rub": round(effective, 6) if effective is not None else None,
                        "effective_source_date": effective_day,
                        "fill_method": fill_method,
                        "days_since_observed": age,
                        "is_stale_7d": age is not None and age > 7,
                        "is_stale_30d": age is not None and age > 30,
                        "official_rate_kzt_per_rub": round(official, 6) if official else None,
                        "effective_premium_to_official_pct": round(premium, 6) if premium is not None else None,
                    }
                )
        current += timedelta(days=1)
    return rows


def make_wide_rows(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in daily_rows:
        output = by_day[row["date"]]
        output["date"] = row["date"]
        prefix = f'{row["segment"]}_{row["direction"]}'
        output[f"{prefix}_rate"] = row["effective_rate_kzt_per_rub"]
        output[f"{prefix}_is_observed"] = row["is_observed"]
        output[f"{prefix}_age_days"] = row["days_since_observed"]
    return [by_day[day] for day in sorted(by_day)]


def deterministic_review_sample(
    observations: list[Observation],
    per_stratum: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        strata[(observation.quality_status, observation.segment, observation.direction)].append(observation)
    selected: list[Observation] = []
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda item: hashlib.sha256(f"review-v1:{item.message_ref}:{item.rate_kzt_per_rub}".encode()).hexdigest(),
        )
        selected.extend(ranked[:per_stratum])
    result: list[dict[str, Any]] = []
    for observation in selected:
        row = public_observation_row(observation)
        row["evidence_excerpt_redacted"] = observation.evidence_excerpt
        row["manual_relevant"] = ""
        row["manual_segment"] = ""
        row["manual_direction"] = ""
        row["manual_rate"] = ""
        row["manual_notes"] = ""
        result.append(row)
    return result


def distribution(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_quality_report(
    observations: list[Observation],
    daily_rows: list[dict[str, Any]],
    source_metadata: list[dict[str, Any]],
    source_counters: dict[str, Counter[str]],
) -> dict[str, Any]:
    by_status = Counter(item.quality_status for item in observations)
    by_segment = Counter(item.segment for item in observations if item.quality_status == "accepted")
    by_direction = Counter(item.direction for item in observations if item.quality_status == "accepted")
    by_method = Counter(item.extraction_method for item in observations if item.quality_status == "accepted")
    by_source = Counter(item.source_file for item in observations if item.quality_status == "accepted")

    coverage: dict[str, Any] = {}
    for segment in TARGET_SEGMENTS:
        for direction in TARGET_DIRECTIONS:
            cell = [row for row in daily_rows if row["segment"] == segment and row["direction"] == direction]
            observed_rows = [row for row in cell if row["is_observed"]]
            ages = [row["days_since_observed"] for row in cell if row["days_since_observed"] is not None]
            key = f"{segment}:{direction}"
            coverage[key] = {
                "calendar_days": len(cell),
                "observed_days": len(observed_rows),
                "observed_share": round(len(observed_rows) / len(cell), 6) if cell else 0,
                "first_observed_day": observed_rows[0]["date"] if observed_rows else None,
                "last_observed_day": observed_rows[-1]["date"] if observed_rows else None,
                "max_days_since_observed": max(ages) if ages else None,
                "days_stale_over_30": sum(bool(row["is_stale_30d"]) for row in cell),
            }

    unique_public_keys = {
        (item.source_file, item.message_ref, round(item.rate_kzt_per_rub, 8), item.direction)
        for item in observations
    }
    accepted = [item for item in observations if item.quality_status == "accepted"]
    validation = {
        "all_three_sources_processed": len(source_metadata) == 3,
        "observation_public_keys_unique": len(unique_public_keys) == len(observations),
        "accepted_rates_in_hard_range": all(2.5 <= item.rate_kzt_per_rub <= 12.0 for item in accepted),
        "accepted_have_target_segment": all(item.segment in TARGET_SEGMENTS for item in accepted),
        "accepted_have_direction": all(item.direction in TARGET_DIRECTIONS for item in accepted),
        "daily_calendar_complete": len(daily_rows)
        == (
            (date.fromisoformat(max(item["last_message_day"] for item in source_metadata))
             - date.fromisoformat(min(item["first_message_day"] for item in source_metadata))).days
            + 1
        )
        * len(TARGET_SEGMENTS)
        * len(TARGET_DIRECTIONS),
        "forward_fill_never_marks_observed": all(
            not row["is_observed"] for row in daily_rows if row["fill_method"] == "forward_fill"
        ),
    }
    return {
        "pipeline_version": PIPELINE_VERSION,
        "rate_unit": "KZT per 1 RUB",
        "source_metadata": source_metadata,
        "source_counters": {key: distribution(value) for key, value in sorted(source_counters.items())},
        "extraction": {
            "observations_total": len(observations),
            "by_quality_status": distribution(by_status),
            "accepted_by_segment": distribution(by_segment),
            "accepted_by_direction": distribution(by_direction),
            "accepted_by_method": distribution(by_method),
            "accepted_by_source": distribution(by_source),
        },
        "daily_coverage": coverage,
        "validation": validation,
        "interpretation": {
            "observed": "textual quote found; transaction settlement is not observed",
            "effective": "last textual quote carried forward; inspect days_since_observed before use",
            "official_rate": "external sanity benchmark only, never substituted for P2P quote",
        },
    }


def output_sha256(path: Path) -> str:
    return file_sha256(path)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    official_rates = load_official_rates(args.official_rates)
    all_observations: list[Observation] = []
    source_metadata: list[dict[str, Any]] = []
    source_counters: dict[str, Counter[str]] = {}

    for input_path in args.input:
        observations, metadata, counters = analyze_source(input_path, official_rates)
        all_observations.extend(observations)
        source_metadata.append(metadata)
        source_counters[input_path.name] = counters

    source_metadata.sort(key=lambda item: item["source_file"])
    all_observations = deduplicate_observations(all_observations)
    all_observations.sort(
        key=lambda item: (item.timestamp, item.source_file, item.message_ref, item.rate_kzt_per_rub)
    )
    first_day = min(item["first_message_day"] for item in source_metadata)
    last_day = max(item["last_message_day"] for item in source_metadata)
    daily_rows = aggregate_daily(all_observations, first_day, last_day, official_rates)
    wide_rows = make_wide_rows(daily_rows)

    observation_fields = [
        "source_file", "source_chat", "message_ref", "timestamp", "day", "segment", "market_scope", "direction",
        "rate_kzt_per_rub", "extraction_method", "pair_basis", "confidence_score", "confidence",
        "quality_status", "official_rate_kzt_per_rub", "deviation_from_official_pct",
    ]
    observation_path = args.output / "rate_observations.csv"
    write_csv(observation_path, (public_observation_row(item) for item in all_observations), observation_fields)
    confirmed_path = args.output / "confirmed_rate_observations.csv"
    write_csv(
        confirmed_path,
        (public_observation_row(item) for item in all_observations if item.quality_status == "accepted"),
        observation_fields,
    )

    daily_fields = [
        "date", "segment", "direction", "is_observed", "observed_rate_kzt_per_rub", "observed_p25",
        "observed_p75", "observed_min", "observed_max", "distinct_posters", "source_count", "source_files",
        "effective_rate_kzt_per_rub", "effective_source_date", "fill_method", "days_since_observed",
        "is_stale_7d", "is_stale_30d", "official_rate_kzt_per_rub", "effective_premium_to_official_pct",
    ]
    daily_path = args.output / "daily_rates.csv"
    write_csv(daily_path, daily_rows, daily_fields)

    card_path = args.output / "daily_card_transfer_rates.csv"
    write_csv(card_path, (row for row in daily_rows if row["segment"] == "card_transfer"), daily_fields)
    cash_path = args.output / "daily_cash_rates.csv"
    write_csv(cash_path, (row for row in daily_rows if row["segment"] == "cash"), daily_fields)
    crypto_path = args.output / "daily_crypto_rates.csv"
    write_csv(crypto_path, (row for row in daily_rows if row["segment"] == "crypto"), daily_fields)

    wide_fields = ["date"]
    for segment in TARGET_SEGMENTS:
        for direction in TARGET_DIRECTIONS:
            prefix = f"{segment}_{direction}"
            wide_fields.extend([f"{prefix}_rate", f"{prefix}_is_observed", f"{prefix}_age_days"])
    wide_path = args.output / "daily_rates_wide.csv"
    write_csv(wide_path, wide_rows, wide_fields)

    review_fields = observation_fields + [
        "evidence_excerpt_redacted", "manual_relevant", "manual_segment", "manual_direction",
        "manual_rate", "manual_notes",
    ]
    review_path = args.output / "review_sample.csv"
    review_rows = deterministic_review_sample(all_observations, args.review_per_stratum)
    write_csv(review_path, review_rows, review_fields)

    quality_report = build_quality_report(
        all_observations,
        daily_rows,
        source_metadata,
        source_counters,
    )
    quality_path = args.output / "quality_report.json"
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = [
        observation_path,
        confirmed_path,
        daily_path,
        card_path,
        cash_path,
        crypto_path,
        wide_path,
        review_path,
        quality_path,
    ]
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "parameters": {
            "review_per_stratum": args.review_per_stratum,
            "target_segments": list(TARGET_SEGMENTS),
            "target_directions": list(TARGET_DIRECTIONS),
            "rate_unit": "KZT per 1 RUB",
        },
        "inputs": source_metadata
        + ([{"source_file": args.official_rates.name, "sha256": file_sha256(args.official_rates)}]
           if args.official_rates else []),
        "outputs": [
            {"file": path.name, "size_bytes": path.stat().st_size, "sha256": output_sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not all(quality_report["validation"].values()):
        failed = [key for key, value in quality_report["validation"].items() if not value]
        raise RuntimeError(f"validation failed: {', '.join(failed)}")

    print(json.dumps({
        "status": "ok",
        "observations": len(all_observations),
        "accepted": quality_report["extraction"]["by_quality_status"].get("accepted", 0),
        "review": quality_report["extraction"]["by_quality_status"].get("review", 0),
        "date_range": [first_day, last_day],
        "output": str(args.output),
    }, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
