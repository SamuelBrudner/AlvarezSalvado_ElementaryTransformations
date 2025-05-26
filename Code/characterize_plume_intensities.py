"""Utilities to characterize plume intensities and manage JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from Code.intensity_stats import calculate_intensity_stats_dict


def process_plume(
    plume_id: str, intensities: List[float], output_json: Path | str
) -> Dict[str, Any]:
    """Compute statistics for a plume and update an output JSON file.

    Parameters
    ----------
    plume_id : str
        Identifier for the plume being processed.
    intensities : list of float
        Intensity values for the plume.
    output_json : Path or str
        Path to the JSON file used for storing plume statistics.

    Returns
    -------
    dict
        The dictionary written for this plume.
    """
    output_path = Path(output_json)

    if intensities:
        stats = calculate_intensity_stats_dict(intensities)
    else:
        stats = {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }

    new_entry = {
        "plume_id": plume_id,
        "statistics": stats,
    }

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    existing = [e for e in existing if e.get("plume_id") != plume_id]
    existing.append(new_entry)
    output_path.write_text(json.dumps(existing, indent=4))
    return new_entry


def main(args: List[str] | None = None) -> None:  # pragma: no cover - CLI entry
    """Command-line interface for characterizing plume intensities.

    Parameters
    ----------
    args : list of str or None, optional
        Command-line arguments. ``--px_per_mm`` and ``--frame_rate`` are passed
        through to :func:`get_intensities_from_video_via_matlab` when processing
        video plumes.
    """
    import argparse

    from Code.analyze_crimaldi_data import get_intensities_from_crimaldi
    from Code.video_intensity import get_intensities_from_video_via_matlab

    parser = argparse.ArgumentParser(description="Characterize plume intensities")
    parser.add_argument("--plume_type", choices=["crimaldi", "video"], required=True)
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--plume_id", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--px_per_mm", type=float)
    parser.add_argument("--frame_rate", type=float)
    parser.add_argument("--matlab_exec", default="matlab", help="Path to MATLAB executable")

    ns = parser.parse_args(args)

    if ns.plume_type == "video":
        if ns.px_per_mm is None or ns.frame_rate is None:
            parser.error("--px_per_mm and --frame_rate are required for video plumes")
        script_contents = Path(ns.file_path).read_text()
        intensities = get_intensities_from_video_via_matlab(
            script_contents,
            ns.matlab_exec,
            px_per_mm=ns.px_per_mm,
            frame_rate=ns.frame_rate,
        )
    else:
        intensities = get_intensities_from_crimaldi(ns.file_path)

    if hasattr(intensities, "tolist"):
        intensities = intensities.tolist()

    process_plume(ns.plume_id, intensities, ns.output_json)


if __name__ == "__main__":  # pragma: no cover
    main()
