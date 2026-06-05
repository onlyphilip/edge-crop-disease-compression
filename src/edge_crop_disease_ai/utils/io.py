"""General I/O helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save a JSON-serializable dictionary to disk.

    Args:
        data: Dictionary to serialize.
        output_path: Destination file path.

    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_text_lines(lines: list[str], output_path: str | Path) -> None:
    """Save a list of text lines to disk.

    Args:
        lines: Ordered lines to write.
        output_path: Destination file path.

    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
