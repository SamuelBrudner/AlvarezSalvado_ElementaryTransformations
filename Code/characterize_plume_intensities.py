"""CLI for characterizing plume intensities."""

from __future__ import annotations

import argparse
import json  # noqa: F401  # placeholder for future use
import os  # noqa: F401  # placeholder for future use
try:
    import numpy as np  # noqa: F401  # placeholder for future use
except ImportError:  # pragma: no cover - optional dependency
    np = None


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Characterize plume intensities"
    )
    parser.add_argument(
        "--plume_type",
        choices=["video", "crimaldi"],
        required=True,
        help="Type of plume data to analyze",
    )
    parser.add_argument("--file_path", required=True, help="Path to input file")
    parser.add_argument(
        "--output_json", required=True, help="Path to output JSON file"
    )
    parser.add_argument("--plume_id", required=True, help="Identifier for the run")
    parser.add_argument(
        "--px_per_mm",
        type=float,
        help="Pixels per millimeter (required for video)",
    )
    parser.add_argument(
        "--frame_rate", type=float, help="Frame rate in Hz (required for video)"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.01,
        help="Minimum intensity threshold",
    )
    parser.add_argument(
        "--matlab_path", default="matlab", help="Path to MATLAB executable"
    )

    args = parser.parse_args(argv)

    if args.plume_type == "video":
        if args.px_per_mm is None or args.frame_rate is None:
            parser.error("--px_per_mm and --frame_rate are required for video")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    print(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
