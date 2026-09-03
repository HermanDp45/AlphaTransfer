"""Typed configuration loaded from the checked-in TOML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class V0Config:
    raw: dict
    root: Path

    @classmethod
    def load(cls, path: str | Path = "configs/kzt_v0.toml") -> "V0Config":
        p = Path(path).resolve()
        with p.open("rb") as handle:
            return cls(tomllib.load(handle), p.parent.parent)

    def section(self, name: str) -> dict:
        return dict(self.raw[name])

    def path(self, name: str) -> Path:
        p = Path(self.raw["paths"][name])
        return p if p.is_absolute() else self.root / p

