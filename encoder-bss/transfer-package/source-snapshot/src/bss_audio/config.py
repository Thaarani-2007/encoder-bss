"""Load YAML config with env overrides."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var, default)

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw=_expand_env(raw))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def data_root(self) -> Path:
        return Path(self.get("data", "root", default="D:/bss-audio-data"))

    @property
    def sample_rate(self) -> int:
        return int(self.get("data", "sample_rate", default=16000))

    @property
    def clip_samples(self) -> int:
        sec = float(self.get("data", "clip_seconds", default=4.0))
        return int(sec * self.sample_rate)
