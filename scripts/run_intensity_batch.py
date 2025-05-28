#!/usr/bin/env python3
"""Helper to compare Crimaldi data with a smoke plume script."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from Code import compare_intensity_stats as cis


def _load_matlab_exec(config_path: Path) -> str | None:
    if not config_path.is_file():
        return None
    try:
        text = config_path.read_text()
        if yaml is not None:
            cfg = yaml.safe_load(text) or {}
        else:  # very naive YAML parser for one level
            cfg = {}
            for line in text.splitlines():
                if line.startswith("matlab:"):
                    cfg["matlab"] = {}
                elif line.startswith("  executable:"):
                    if "matlab" not in cfg:
                        cfg["matlab"] = {}
                    cfg["matlab"]["executable"] = line.split(":", 1)[1].strip()
        return cfg.get("matlab", {}).get("executable")
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run intensity comparison between Crimaldi and smoke datasets"
    )
    parser.add_argument("crimaldi_path", help="Path to Crimaldi HDF5 file")
    parser.add_argument("smoke_script", help="Path to smoke plume MATLAB script")
    parser.add_argument(
        "--matlab_exec",
        help="Path to MATLAB executable (overrides config)",
    )
    parser.add_argument(
        "--config",
        default="configs/project_paths.yaml",
        help="Path to project_paths.yaml",
    )
    ns = parser.parse_args(argv)

    matlab_exec = ns.matlab_exec or _load_matlab_exec(Path(ns.config)) or "matlab"

    results = cis.compare_intensity_stats(
        [
            ("CRIM", ns.crimaldi_path, "crimaldi"),
            ("SMOKE", ns.smoke_script, "video"),
        ],
        matlab_exec,
    )
    print(cis.format_table(results))


if __name__ == "__main__":  # pragma: no cover
    main()
