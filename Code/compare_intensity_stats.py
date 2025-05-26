"""Compare intensity statistics across multiple datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from Code.analyze_crimaldi_data import get_intensities_from_crimaldi
from Code.video_intensity import get_intensities_from_video_via_matlab
from Code.intensity_stats import calculate_intensity_stats_dict


Stats = dict[str, float]


def load_intensities(
    path: str,
    plume_type: str | None = None,
    matlab_exec_path: str = "matlab",
) -> np.ndarray:
    """Load intensity vector based on plume type.

    If ``plume_type`` is ``None``, the function tries to infer the type from the
    file extension: ``.h5`` or ``.hdf5`` files are treated as ``crimaldi``.
    Everything else is assumed to be a MATLAB script for ``video`` plumes.
    """

    if plume_type is None:
        if path.lower().endswith((".h5", ".hdf5")):
            plume_type = "crimaldi"
        else:
            plume_type = "video"

    if plume_type == "crimaldi":
        return get_intensities_from_crimaldi(path)
    if plume_type == "video":
        script_contents = Path(path).read_text()
        return get_intensities_from_video_via_matlab(script_contents, matlab_exec_path)

    raise ValueError(f"Unknown plume_type: {plume_type}")


def compare_intensity_stats(
    sources: Iterable[Tuple[str, str, str | None]]
) -> List[Tuple[str, Stats]]:
    """Return statistics for each identifier/path/type triple."""

    results: List[Tuple[str, Stats]] = []
    for identifier, path, plume_type in sources:
        intensities = load_intensities(path, plume_type)
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
    parser.add_argument(
        "--item",
        action="append",
        nargs=2,
        metavar=("ID[:TYPE]", "PATH"),
        required=True,
        help="Add a dataset: ID optionally followed by ':TYPE' and its PATH",
    )
    parser.add_argument("--csv", dest="csv_path", help="Output CSV file")
    ns = parser.parse_args(argv)

    entries = []
    for id_type, path in ns.item:
        if ":" in id_type:
            identifier, plume_type = id_type.split(":", 1)
        else:
            identifier, plume_type = id_type, None
        entries.append((identifier, path, plume_type))

    results = compare_intensity_stats(entries)

    if ns.csv_path:
        write_csv(results, ns.csv_path)
    else:
        print(format_table(results))


if __name__ == "__main__":  # pragma: no cover
    main()
