"""Configuration loading utilities for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the loaded YAML is not a mapping.
    """
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_file}")

    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create required output directories from config.

    Args:
        config: Parsed project configuration.

    """
    paths = config.get("paths", {})
    output_keys = (
        "checkpoints_dir",
        "results_dir",
        "split_cache_dir",
        "label_map_path",
    )

    for key in output_keys:
        raw_path = paths.get(key)
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        target = path.parent if path.suffix else path
        target.mkdir(parents=True, exist_ok=True)

    export_dir = config.get("export", {}).get("export_dir")
    if export_dir:
        Path(export_dir).expanduser().mkdir(parents=True, exist_ok=True)
