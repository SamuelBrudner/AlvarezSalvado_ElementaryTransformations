#!/usr/bin/env python3
"""Interactive setup for plume configuration files.

This script requests the user to provide the file paths for the smoke and
Crimaldi plume datasets. Configuration files are then generated in the
specified directory (default: configs/plumes) with hard coded frame rates
and pixel sizes.
"""
import argparse
import json
import logging
from pathlib import Path


SMOKE_FPS = 60
SMOKE_MM_PER_PIXEL = 0.153
CRIMALDI_FPS = 15
CRIMALDI_MM_PER_PIXEL = 0.74
DATASET_NAME = "/dataset2"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def ask_path(prompt: str) -> str:
    """Prompt the user for a path."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def write_config(path: Path, plume_file: str, mm_per_pixel: float, fps: int) -> None:
    """Write a simple JSON configuration."""
    data = {
        "plume_file": plume_file,
        "dataset_name": DATASET_NAME,
        "mm_per_pixel": mm_per_pixel,
        "frame_rate": fps,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plume configuration files")
    parser.add_argument("--smoke-file", help="Path to smoke plume HDF5 file")
    parser.add_argument("--crimaldi-file", help="Path to crimaldi plume HDF5 file")
    parser.add_argument("--config-dir", default="configs/plumes", help="Directory for output configs")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    smoke_file = args.smoke_file or ask_path("Path to smoke plume file: ")
    crimaldi_file = args.crimaldi_file or ask_path("Path to crimaldi plume file: ")

    logger.info("Smoke file: %s", smoke_file)
    logger.info("Crimaldi file: %s", crimaldi_file)

    write_config(config_dir / "plumes_smoke_info.json", smoke_file, SMOKE_MM_PER_PIXEL, SMOKE_FPS)
    write_config(config_dir / "plumes_crimaldi_info.json", crimaldi_file, CRIMALDI_MM_PER_PIXEL, CRIMALDI_FPS)


if __name__ == "__main__":
    main()
