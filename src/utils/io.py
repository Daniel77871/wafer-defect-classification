"""File-system helpers used across scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
