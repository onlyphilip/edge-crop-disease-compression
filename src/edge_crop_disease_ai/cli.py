"""Shared CLI helpers for project scripts."""

from __future__ import annotations

import argparse


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build a common argument parser with a config path.

    Args:
        description: Short command description for CLI help.

    Returns:
        Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    return parser
