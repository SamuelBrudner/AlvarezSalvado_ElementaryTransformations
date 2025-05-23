"""Utilities to characterize plume intensities and manage JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import statistics


def process_plume(plume_id: str, intensities: List[float], output_json: Path | str) -> Dict[str, Any]:
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
        mean_val = statistics.mean(intensities)
        std_val = statistics.stdev(intensities) if len(intensities) > 1 else 0.0
    else:
        mean_val = float("nan")
        std_val = float("nan")

    new_entry = {
        "plume_id": plume_id,
        "statistics": {
            "mean": mean_val,
            "std": std_val,
            "count": len(intensities),
        },
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


if __name__ == "__main__":  # pragma: no cover - simple CLI for manual use
    import argparse

    parser = argparse.ArgumentParser(description="Characterize plume intensities")
    parser.add_argument("plume_id", help="Identifier for the plume")
    parser.add_argument("intensities", nargs="+", type=float, help="Intensity values")
    parser.add_argument("output_json", help="Path to the output JSON file")

    args = parser.parse_args()
    process_plume(args.plume_id, args.intensities, args.output_json)
