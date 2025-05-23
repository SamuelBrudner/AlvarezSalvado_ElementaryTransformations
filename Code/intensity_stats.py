"""Utility functions for plume intensity statistics."""

from __future__ import annotations

import argparse
from typing import Sequence, Dict

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise ImportError("numpy is required for intensity statistics") from exc


def calculate_intensity_stats_dict(intensities: Sequence[float]) -> Dict[str, float]:
    """Return basic statistics for the provided intensities."""
    arr = np.asarray(intensities, dtype=float)
    stats = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }
    return stats


def _print_stats(identifier: str, file_path: str, stats: Dict[str, float]) -> None:
    print(f"Plume: {identifier}")
    print(f"File: {file_path}")
    for key in ["mean", "median", "p95", "p99", "min", "max", "count"]:
        print(f"{key}: {stats[key]}")


def main(args: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Calculate intensity statistics")
    parser.add_argument("identifier", help="Plume identifier")
    parser.add_argument("file", help="Path to text file containing intensities")
    parser.add_argument("--plot_histogram", action="store_true", help="Plot intensity histogram")
    ns = parser.parse_args(args)

    data = np.loadtxt(ns.file)
    stats = calculate_intensity_stats_dict(data)
    _print_stats(ns.identifier, ns.file, stats)

    if ns.plot_histogram:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("matplotlib is required for plotting") from exc
        plt.hist(data, bins=30)
        plt.title(f"Intensity Histogram: {ns.identifier}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
