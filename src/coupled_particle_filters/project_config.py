"""Loader for the legacy project-path YAML used by data precomputation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


def load_project_config(path: str | Path = "configs/project.yml") -> ConfigNode:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ConfigNode(yaml.safe_load(handle) or {})
