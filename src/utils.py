"""
src/utils.py
------------
Shared utility helpers used across the monitoring pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    """Create outputs/ and models/ directories if they do not exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: dict[str, Any], filepath: Path | str) -> None:
    """Serialise *data* to a JSON file at *filepath*."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"  [saved] {filepath.relative_to(PROJECT_ROOT)}")


def load_json(filepath: Path | str) -> dict[str, Any]:
    """Load and return JSON content from *filepath*."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------------

def section(title: str, width: int = 60) -> None:
    """Print a section banner to stdout.

    Use ASCII characters so the output remains portable across Windows terminal
    encodings such as cp1252, which do not support the box-drawing glyphs used
    in the original banner.
    """
    bar = "-" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
