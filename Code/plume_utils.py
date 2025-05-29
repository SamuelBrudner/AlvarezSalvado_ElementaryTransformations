"""Utilities for plume intensity scaling.

The intensity statistics for the default plumes are stored in the YAML file
``configs/plume_intensity_stats.yaml`` located at the repository root.

Examples
--------
>>> from Code.plume_utils import get_intensity_stats, rescale_to_crim_range
>>> stats = get_intensity_stats()
>>> rescaled = rescale_to_crim_range([0, 1])
>>> round(rescaled[0], 3)
-0.02
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - minimal fallback
    np = None

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback parser
    import json
    import types

    def _minimal_safe_load(text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            data = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                try:
                    if value.lower() in ("true", "false"):
                        data[key] = value.lower() == "true"
                    elif "." in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
            return data

    yaml = types.SimpleNamespace(safe_load=_minimal_safe_load)

__all__ = ["get_intensity_stats", "rescale_to_crim_range"]


_STATS_CACHE: Dict[str, Dict[str, Any]] | None = None
_STATS_PATH = Path(__file__).resolve().parent.parent / "configs" / "plume_intensity_stats.yaml"


def get_intensity_stats() -> Dict[str, Dict[str, Any]]:
    """Return stored intensity statistics for the default plumes."""
    global _STATS_CACHE
    if _STATS_CACHE is None:
        _STATS_CACHE = yaml.safe_load(_STATS_PATH.read_text())
    return {k: dict(v) for k, v in _STATS_CACHE.items()}


def _rescale(arr: Sequence[float] | np.ndarray,
             target_min: float,
             target_max: float) -> Sequence[float] | np.ndarray:
    """Rescale ``arr`` to ``[target_min, target_max]``."""
    if np is not None:
        arr_np = np.asarray(arr, dtype=float)
        src_min = float(np.min(arr_np))
        src_max = float(np.max(arr_np))
        if src_max == src_min:
            return np.full_like(arr_np, target_min)
        scale = (target_max - target_min) / (src_max - src_min)
        return (arr_np - src_min) * scale + target_min

    # Minimal fallback without NumPy
    arr_list = [float(x) for x in arr]
    src_min = min(arr_list)
    src_max = max(arr_list)
    if src_max == src_min:
        return [target_min for _ in arr_list]
    scale = (target_max - target_min) / (src_max - src_min)
    return [(x - src_min) * scale + target_min for x in arr_list]


def rescale_to_crim_range(arr: Sequence[float] | np.ndarray) -> Sequence[float] | np.ndarray:
    """Linearly rescale ``arr`` to match the Crimaldi min and max."""
    stats = get_intensity_stats()
    return _rescale(arr, stats["CRIM"]["min"], stats["CRIM"]["max"])
