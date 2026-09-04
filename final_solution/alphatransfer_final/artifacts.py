"""Small, dependency-free helpers for reproducible artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock(repo_root: Path, lock_path: Path) -> dict[str, str]:
    lock = load_json(lock_path)
    failures: list[str] = []
    observed: dict[str, str] = {}
    for relative, expected in lock["files"].items():
        path = repo_root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        observed[relative] = actual
        if actual != expected:
            failures.append(f"sha256 mismatch: {relative}")
    if failures:
        raise RuntimeError("Locked input verification failed:\n- " + "\n- ".join(failures))
    return observed


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_one(
    rows: Iterable[dict[str, str]],
    predicates: dict[str, str],
    label: str,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if all(str(row.get(column)) == str(value) for column, value in predicates.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {label} row, found {len(selected)}")
    return selected[0]


def as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise RuntimeError(f"Missing numeric field: {name}")
    return float(value)


def as_int(row: dict[str, str], name: str) -> int:
    return int(round(as_float(row, name)))


def as_bool(row: dict[str, str], name: str) -> bool:
    value = row.get(name, "").strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"Invalid boolean field {name}: {value!r}")
    return value == "true"

