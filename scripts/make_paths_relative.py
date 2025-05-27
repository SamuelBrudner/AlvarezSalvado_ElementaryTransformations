#!/usr/bin/env python3
"""Relativise YAML paths to the project root.

Usage: make_paths_relative.py <config_path> <project_root>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

SKIP_PATHS = {
    "matlab.executable",
    "scripts.temp",
    "tmp_dir",
    "output.matlab_temp",
}


def process_value(value: Any, project_root: str, key_path: str = "") -> Any:
    """Return value with paths made relative to *project_root* where possible."""
    if isinstance(value, dict):
        return {
            k: process_value(v, project_root, f"{key_path}.{k}" if key_path else k)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [process_value(item, project_root, f"{key_path}[]") for item in value]
    if (
        isinstance(value, str)
        and value.startswith(project_root)
        and key_path not in SKIP_PATHS
    ):
        rel_path = os.path.relpath(value, project_root)
        if not rel_path.startswith(".."):
            return rel_path
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: make_paths_relative.py <config_path> <project_root>",
            file=sys.stderr,
        )
        return 1

    config_path = Path(sys.argv[1])
    project_root = sys.argv[2]

    with config_path.open() as fh:
        config = yaml.safe_load(fh) or {}

    processed = process_value(config, project_root)

    with config_path.open("w") as fh:
        yaml.safe_dump(processed, fh, sort_keys=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
