"""Utilities for discovering processed data files based on a config."""

from __future__ import annotations

import json
import re
import csv
from pathlib import Path
from typing import Dict, Iterator, Any


def _template_to_regex(template: str) -> re.Pattern:
    pattern = re.escape(template)
    pattern = pattern.replace(r"\{", "{").replace(r"\}", "}")
    pattern = re.sub(r"{(\w+)}", r"(?P<\1>[^/]+)", pattern)
    return re.compile(f"^{pattern}$")


def discover_processed_data(cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield discovered run information from processed data directories.

    Parameters
    ----------
    cfg : dict
        Analysis configuration dictionary loaded via :func:`load_analysis_config`.

    Yields
    ------
    dict
        Dictionary with keys ``path`` and ``metadata`` plus optional entries
        for ``config``, ``summary``, ``params``, and ``trajectories`` depending
        on ``data_loading_options``.
    """
    base_dirs = cfg.get("data_paths", {}).get("processed_base_dirs", [])
    template = cfg.get("metadata_extraction", {}).get(
        "directory_template",
        "{plume}_{mode}/agent_{agent_id}/seed_{seed}",
    )

    options = cfg.get("data_loading_options", {})
    load_summary = options.get("load_summary_json", False)
    load_traj = options.get("load_trajectories_csv", False)
    load_params = options.get("load_params_json", False)
    load_run_cfg = options.get("load_config_used_yaml", cfg.get("load_run_config", False))

    regex = _template_to_regex(template)

    for base in base_dirs:
        root = Path(base)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                rel = path.relative_to(root)
                m = regex.match(str(rel))
                if m:
                    record = {
                        "path": str(path),
                        "metadata": m.groupdict(),
                    }
                    if load_run_cfg:
                        cfg_file = path / "config_used.yaml"
                        if cfg_file.is_file():
                            try:
                                record["config"] = json.loads(cfg_file.read_text())
                            except Exception:
                                record["config"] = {}

                    if load_summary:
                        summary_file = path / "summary.json"
                        if summary_file.is_file():
                            try:
                                record["summary"] = json.loads(summary_file.read_text())
                            except Exception:
                                record["summary"] = {}

                    if load_params:
                        param_file = path / "params.json"
                        if param_file.is_file():
                            try:
                                record["params"] = json.loads(param_file.read_text())
                            except Exception:
                                record["params"] = {}

                    if load_traj:
                        traj_file = path / "trajectories.csv"
                        if traj_file.is_file():
                            with traj_file.open() as f:
                                reader = csv.DictReader(f)
                                record["trajectories"] = [row for row in reader]

                    yield record
