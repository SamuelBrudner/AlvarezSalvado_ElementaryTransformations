#!/usr/bin/env python3
"""Command-line interface for :func:`Code.plume_pipeline.video_to_scaled_rotated_h5`.

This script converts an AVI video to HDF5, rescales intensities to the
Crimaldi range, rotates frames 90° clockwise, and updates the plume
registry. Paths are given relative to the repository root.

Example
-------
Run the pipeline on ``input.avi`` and store results under ``data/processed``::

    conda run --prefix ./dev_env python -m scripts.run_plume_pipeline \
        data/raw/input.avi data/processed/input_raw.h5 \
        data/processed/input_scaled.h5 data/processed/input_rotated.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure the repo root containing the Code package is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Code.plume_pipeline import video_to_scaled_rotated_h5  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert AVI to HDF5, scale and rotate plume frames"
    )
    parser.add_argument("avi", help="Input AVI file")
    parser.add_argument("raw_h5", help="Output HDF5 file with raw intensities")
    parser.add_argument(
        "scaled_h5", help="Output HDF5 file with intensities scaled to CRIM range"
    )
    parser.add_argument(
        "rotated_h5", help="Output HDF5 file with scaled frames rotated clockwise"
    )
    ns = parser.parse_args(argv)

    video_to_scaled_rotated_h5(ns.avi, ns.raw_h5, ns.scaled_h5, ns.rotated_h5)


if __name__ == "__main__":  # pragma: no cover
    main()
