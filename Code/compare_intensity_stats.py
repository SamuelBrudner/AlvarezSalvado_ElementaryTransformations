"""Compare intensity statistics across multiple datasets."""

from __future__ import annotations

import argparse
import csv
from typing import Iterable, List, Tuple

import numpy as np

from Code.analyze_crimaldi_data import get_intensities_from_crimaldi
from Code.intensity_stats import calculate_intensity_stats_dict


Stats = dict[str, float]


def load_intensities(path: str) -> np.ndarray:
    """Load intensity vector from an HDF5 file using :func:`get_intensities_from_crimaldi`."""
    return get_intensities_from_crimaldi(path)


def compare_intensity_stats(sources: Iterable[Tuple[str, str]]) -> List[Tuple[str, Stats]]:
    """Return statistics for each identifier/path pair."""
    results: List[Tuple[str, Stats]] = []
    for identifier, path in sources:
        intensities = load_intensities(path)
        stats = calculate_intensity_stats_dict(intensities)
        results.append((identifier, stats))
    return results


def format_table(results: Iterable[Tuple[str, Stats]]) -> str:
    keys = ["mean", "median", "p95", "p99", "min", "max", "count"]
    header = ["identifier"] + keys
    lines = ["\t".join(header)]
    for ident, stats in results:
        row = [ident] + [f"{stats[k]:.3f}" if k != "count" else str(stats[k]) for k in keys]
        lines.append("\t".join(row))
    return "\n".join(lines)


def write_csv(results: Iterable[Tuple[str, Stats]], csv_path: str) -> None:
    keys = ["mean", "median", "p95", "p99", "min", "max", "count"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier"] + keys)
        for ident, stats in results:
            writer.writerow([ident] + [stats[k] for k in keys])


def main(argv: List[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Compare intensity statistics")
    parser.add_argument("pairs", nargs="+", help="identifier and file path pairs")
    parser.add_argument("--csv", dest="csv_path", help="Output CSV file")
    ns = parser.parse_args(argv)

    if len(ns.pairs) % 2 != 0:
        parser.error("Expected pairs of identifier and path")

    pairs = [(ns.pairs[i], ns.pairs[i + 1]) for i in range(0, len(ns.pairs), 2)]
    results = compare_intensity_stats(pairs)

    if ns.csv_path:
        write_csv(results, ns.csv_path)
    else:
        print(format_table(results))


if __name__ == "__main__":  # pragma: no cover
    main()
