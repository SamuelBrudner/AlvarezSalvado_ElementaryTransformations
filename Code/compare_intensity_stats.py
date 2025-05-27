"""Compare intensity statistics across multiple datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from Code.analyze_crimaldi_data import get_intensities_from_crimaldi
from Code.intensity_stats import calculate_intensity_stats_dict
from Code.video_intensity import get_intensities_from_video_via_matlab

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
        return get_intensities_from_video_via_matlab(
            script_contents,
            matlab_exec_path,
            work_dir=str(Path(path).parent),
            orig_script_path=str(Path(path)),
        )

    raise ValueError(f"Unknown plume_type: {plume_type}")


def compare_intensity_stats(
    sources: Iterable[Tuple[str, str, str | None]],
    matlab_exec_path: str = "matlab",
) -> List[Tuple[str, Stats]]:
    """Return statistics for each identifier/path/type triple."""

    results: List[Tuple[str, Stats]] = []
    for identifier, path, plume_type in sources:
        intensities = load_intensities(path, plume_type, matlab_exec_path)
        stats = calculate_intensity_stats_dict(intensities)
        results.append((identifier, stats))
    return results


def compute_differences(results: Iterable[Tuple[str, Stats]]) -> Stats:
    """Return differences between two result dictionaries."""
    res = list(results)
    if len(res) != 2:
        raise ValueError("Exactly two datasets are required to compute differences")
    a, b = res[0][1], res[1][1]
    return {
        "delta_mean": a["mean"] - b["mean"],
        "delta_median": a["median"] - b["median"],
        "delta_p95": a["p95"] - b["p95"],
        "delta_p99": a["p99"] - b["p99"],
        "delta_min": a["min"] - b["min"],
        "delta_max": a["max"] - b["max"],
        "delta_count": a["count"] - b["count"],
    }


def format_table(
    results: Iterable[Tuple[str, Stats]], diff: Stats | None = None
) -> str:
    keys = ["mean", "median", "p95", "p99", "min", "max", "count"]
    header = ["identifier"] + keys
    lines = ["\t".join(header)]
    for ident, stats in results:
        row = [ident] + [
            f"{stats[k]:.3f}" if k != "count" else str(stats[k]) for k in keys
        ]
        lines.append("\t".join(row))
    if diff is not None:
        diff_row = ["DIFF"] + [
            (f"{diff['delta_' + k]:.3f}" if k != "count" else str(diff["delta_" + k]))
            for k in keys
        ]
        lines.append("\t".join(diff_row))
    return "\n".join(lines)


def write_csv(
    results: Iterable[Tuple[str, Stats]], csv_path: str, diff: Stats | None = None
) -> None:
    keys = ["mean", "median", "p95", "p99", "min", "max", "count"]
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["identifier"] + keys)
        for ident, stats in results:
            writer.writerow([ident] + [stats[k] for k in keys])
        if diff is not None:
            writer.writerow(
                [
                    "DIFF",
                    diff["delta_mean"],
                    diff["delta_median"],
                    diff["delta_p95"],
                    diff["delta_p99"],
                    diff["delta_min"],
                    diff["delta_max"],
                    diff["delta_count"],
                ]
            )


def write_json(
    results: Iterable[Tuple[str, Stats]], json_path: str, diff: Stats | None = None
) -> None:
    """Write statistics to a JSON file."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"identifier": ident, "statistics": stats} for ident, stats in results]
    if diff is not None:
        entries.append({"identifier": "DIFF", "statistics": diff})
    path.write_text(json.dumps(entries, indent=4))


def main(argv: List[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Compare intensity statistics")
    parser.add_argument(
        "items",
        nargs="+",
        help="identifier [plume_type] path entries; plume_type optional",
    )
    parser.add_argument("--csv", dest="csv_path", help="Output CSV file")
    parser.add_argument("--json", dest="json_path", help="Output JSON file")
    parser.add_argument(
        "--matlab_exec", default="matlab", help="Path to MATLAB executable"
    )
    parser.add_argument(
        "--diff", action="store_true", help="Show differences for two datasets"
    )
    ns = parser.parse_args(argv)

    if len(ns.items) % 3 == 0:
        entries = [
            (ns.items[i], ns.items[i + 2], ns.items[i + 1])
            for i in range(0, len(ns.items), 3)
        ]
    elif len(ns.items) % 2 == 0:
        entries = [
            (ns.items[i], ns.items[i + 1], None) for i in range(0, len(ns.items), 2)
        ]
    else:
        parser.error(
            "Expected pairs (identifier path) or triples (identifier plume_type path)"
        )

    results = compare_intensity_stats(entries, ns.matlab_exec)
    if ns.diff:
        try:
            diff = compute_differences(results)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        diff = None

    if ns.csv_path:
        write_csv(results, ns.csv_path, diff)
    if ns.json_path:
        write_json(results, ns.json_path, diff)
    if not ns.csv_path and not ns.json_path:
        print(format_table(results, diff))


if __name__ == "__main__":  # pragma: no cover
    main()
