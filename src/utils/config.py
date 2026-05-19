"""Small YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "PyYAML is required to load config files. Install requirements.txt first."
        ) from error

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}
