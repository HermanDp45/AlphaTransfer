"""Configuration loading and validation."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path, repo_root: Path) -> "PipelineConfig":
        # Значения TOML-подмножества совместимы с JSON.
        # Поэтому системному Python 3.9 не нужны tomllib/tomli.
        parser = configparser.ConfigParser(interpolation=None)
        with path.open(encoding="utf-8") as source:
            parser.read_file(source)
        raw = {
            section: {key: json.loads(value) for key, value in parser[section].items()}
            for section in parser.sections()
        }
        required_sections = {"solution", "paths", "policy", "gates"}
        missing = required_sections - set(raw)
        if missing:
            raise ValueError(f"Missing config sections: {', '.join(sorted(missing))}")
        config = cls(repo_root=repo_root.resolve(), raw=raw)
        config.validate()
        return config

    def validate(self) -> None:
        solution = self.raw["solution"]
        gates = self.raw["gates"]
        date.fromisoformat(solution["default_as_of"])
        if int(solution["primary_horizon"]) not in {1, 3, 5, 10, 20}:
            raise ValueError("primary_horizon must be one of 1/3/5/10/20")
        if not solution["corridors"]:
            raise ValueError("at least one corridor is required")
        if float(gates["minimum_lift"]) <= 1.0:
            raise ValueError("minimum_lift must be greater than one")
        if not 0.0 <= float(gates["minimum_weekly_coverage"]) <= 1.0:
            raise ValueError("minimum_weekly_coverage must be between zero and one")

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    def path(self, name: str) -> Path:
        return (self.repo_root / self.raw["paths"][name]).resolve()
