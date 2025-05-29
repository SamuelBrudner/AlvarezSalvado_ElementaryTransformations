"""Utilities for plume intensity scaling.

This module provides helpers to rescale plume intensity arrays so that they
match reference ranges derived from experimental data. Statistics for the
reference plumes are stored in ``configs/plume_intensity_stats.yaml`` relative
to the repository root.

Example
-------
>>> import numpy as np
>>> from Code import plume_utils
>>> arr = np.linspace(0, 1, num=5)
>>> plume_utils.rescale_to_crim_range(arr)
array([...])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

__all__ = ["get_intensity_stats", "rescale_to_crim_range"]

try:  # pragma: no cover - fallback if PyYAML is unavailable
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - use a minimal parser
    import types

    def _minimal_load(path: Path) -> Dict[str, Dict[str, float]]:
        """Very small YAML subset parser used when PyYAML is unavailable."""
        data: Dict[str, Dict[str, float]] = {}
        current: str | None = None
        for raw_line in path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith("  "):
                current = stripped.rstrip(":")
                data[current] = {}
            else:
                key, value = stripped.split(":", 1)
                data[current][key] = float(value)
        return data

    yaml = types.SimpleNamespace(safe_load=lambda f: _minimal_load(Path(f.name)))


def get_intensity_stats(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Return stored intensity statistics for the default plumes."""
    if path is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "plume_intensity_stats.yaml"
        )
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _rescale(arr: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    src_min = float(np.min(arr))
    src_max = float(np.max(arr))
    if src_max == src_min:
        return np.full_like(arr, target_min)
    scale = (target_max - target_min) / (src_max - src_min)
    return (arr - src_min) * scale + target_min


def rescale_to_crim_range(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale ``arr`` to match the Crimaldi min and max."""
    stats = get_intensity_stats()
    return _rescale(arr, stats["CRIM"]["min"], stats["CRIM"]["max"])
