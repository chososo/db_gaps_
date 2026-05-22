"""Project-wide config helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root.

    Resolves by walking up until a marker file is found, or falls back to
    the env var ``DB_GAPS_ROOT`` if set.
    """
    env = os.environ.get("DB_GAPS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "config").exists():
            return parent
    # Last resort: assume two levels up from src/db_gaps/utils/
    return here.parents[3]


@lru_cache(maxsize=1)
def load_settings(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load global settings.yml as a dict (cached)."""
    if path is None:
        path = project_root() / "config" / "settings.yml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return project_root() / p
