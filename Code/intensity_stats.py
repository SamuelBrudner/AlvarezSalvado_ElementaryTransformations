"""Utilities for computing statistics on intensity arrays."""

from __future__ import annotations

from typing import Dict, List

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("calculate_intensity_stats_dict requires numpy") from exc


_PERCENTILES: List[float] = [
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9,
]


def calculate_intensity_stats_dict(
    intensities_array: np.ndarray, min_threshold: float = 0.01
) -> Dict[str, float | int | Dict[float, float]]:
    """Return a dictionary of summary statistics for ``intensities_array``.

    Parameters
    ----------
    intensities_array : numpy.ndarray
        1D array of intensity values.
    min_threshold : float, optional
        Minimum value to include in statistics, by default 0.01.

    Returns
    -------
    dict
        Dictionary containing ``mean``, ``median``, ``std``, ``min``, ``max``,
        ``percentiles`` and pixel counts.
    """

    num_total = int(len(intensities_array))
    filtered = intensities_array[intensities_array >= min_threshold]
    num_filtered = int(len(filtered))

    stats: Dict[str, float | int | Dict[float, float]] = {
        "num_pixels_total": num_total,
        "num_pixels_analyzed_post_threshold": num_filtered,
        "percentiles": {},
        "mean": float("nan"),
        "median": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
    }

    if num_filtered == 0:
        return stats

    stats["mean"] = float(np.mean(filtered))
    stats["median"] = float(np.median(filtered))
    stats["std"] = float(np.std(filtered))
    stats["min"] = float(np.min(filtered))
    stats["max"] = float(np.max(filtered))

    percentiles = np.percentile(filtered, _PERCENTILES)
    stats["percentiles"] = {
        p: float(val) for p, val in zip(_PERCENTILES, percentiles)
    }

    return stats
