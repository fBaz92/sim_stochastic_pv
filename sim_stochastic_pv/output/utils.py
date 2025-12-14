"""
Utility functions for output generation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def slugify(value: str) -> str:
    """
    Convert a free-form string into a filesystem-safe slug.

    Args:
        value: Input string.

    Returns:
        Slug containing only alphanumeric characters, dash, or underscore.
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip()).strip("_")


def create_run_directory(scenario_name: str, output_root: Path) -> Path:
    """
    Create the timestamped directory for optimization results.

    Args:
        scenario_name: Name of the scenario or optimization.
        output_root: Base directory for outputs.

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    slug = slugify(scenario_name) or "scenario"
    output_dir = output_root / f"{timestamp}_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_results_directory(scenario_name: str, base_dir: Path) -> Path:
    """
    Create the timestamped results directory for scenario analysis.

    Args:
        scenario_name: Name of the scenario.
        base_dir: Base directory for results.

    Returns:
        Path to the created directory.
    """
    return create_run_directory(scenario_name, base_dir)
