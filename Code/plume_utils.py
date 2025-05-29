"""Utilities for plume intensity scaling."""
from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np

__all__ = ["get_intensity_stats", "rescale_to_crim_range"]

try:  # pragma: no cover - fall back if PyYAML is unavailable
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight YAML parser
    import types

    def _minimal_load(path: Path) -> Dict[str, Dict[str, float]]:
        """Very small YAML subset parser."""
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


_YAML_PATH = Path(__file__).resolve().parent.parent / "configs" / "plume_intensity_stats.yaml"
with _YAML_PATH.open("r", encoding="utf-8") as fh:
    _INTENSITY_STATS: Dict[str, Dict[str, Any]] = yaml.safe_load(fh)


def get_intensity_stats() -> Dict[str, Dict[str, Any]]:
    """Return stored intensity statistics for the default plumes."""
    return _INTENSITY_STATS


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
