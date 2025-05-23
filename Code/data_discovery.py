"""Utilities for discovering processed data files based on a config."""

from __future__ import annotations

import json
import re
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
        Dictionary with keys ``path``, ``metadata``, and ``config`` (optional).
    """
    base_dirs = cfg.get("data_paths", {}).get("processed_base_dirs", [])
    template = cfg.get("metadata_extraction", {}).get(
        "directory_template",
        "{plume}_{mode}/agent_{agent_id}/seed_{seed}",
    )
    load_run_cfg = cfg.get("load_run_config", False)

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
                    yield record
