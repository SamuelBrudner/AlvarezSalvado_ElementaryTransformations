"""Utilities for discovering processed data files based on a config."""

from __future__ import annotations

import json
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without PyYAML
    import types
    import re

    def _minimal_safe_load(text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            data = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.lower() in ('true', 'false'):
                    data[key] = value.lower() == 'true'
                else:
                    try:
                        if '.' in value:
                            data[key] = float(value)
                        else:
                            data[key] = int(value)
                    except ValueError:
                        data[key] = value
            return data

    yaml = types.SimpleNamespace(safe_load=_minimal_safe_load)
import re
import csv
from pathlib import Path
from typing import Dict, Iterator, Any, List


def _maybe_float(value: Any) -> Any:
    """Return a float if ``value`` looks numeric, otherwise the original value."""
    if not isinstance(value, str):
        return value
    try:
        if value.lower() in ("nan", "inf", "-inf"):
            return float(value)
        return float(value)
    except ValueError:
        return value


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
    param_opts = cfg.get("parameter_usage", {})
    load_opts = cfg.get("data_loading_options", {})

    load_run_cfg = (
        cfg.get("load_run_config", False)
        or param_opts.get("use_config_used_for_dt", False)
        or load_opts.get("load_config_used_yaml", False)
    )

    need_params = (
        param_opts.get("check_model_parameter_consistency", {}).get("enabled", False)
        or load_opts.get("load_params_json", False)
    )

    load_summary = load_opts.get("load_summary_json", False)
    load_traj = load_opts.get("load_trajectories_csv", False)


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
                                record["config"] = yaml.safe_load(cfg_file.read_text())
                            except Exception:
                                record["config"] = {}
                        else:
                            record["config"] = {}
                        if param_opts.get("use_config_used_for_dt", False):
                            fr_key = param_opts.get(
                                "framerate_field_in_config_used", "frame_rate"
                            )
                            fr = record["config"].get(fr_key)
                            if fr:
                                record["dt"] = 1.0 / float(fr)
                    if need_params:

                        param_file = path / "params.json"
                        if param_file.is_file():
                            try:
                                record["params"] = yaml.safe_load(param_file.read_text())
                            except Exception:
                                record["params"] = {}
                        else:
                            record["params"] = {}

                    if load_summary:
                        summary_file = path / "summary.json"
                        if summary_file.is_file():
                            try:
                                record["summary"] = json.loads(summary_file.read_text())
                            except Exception:
                                record["summary"] = {}

                    if load_traj:
                        traj_file = path / "trajectories.csv"
                        if traj_file.is_file():
                            try:
                                with open(traj_file, "r", newline="") as f:
                                    reader = csv.DictReader(f)
                                    rows = []
                                    for row in reader:
                                        rows.append({k: _maybe_float(v) for k, v in row.items()})
                                    record["trajectories"] = rows
                            except Exception:
                                record["trajectories"] = []

                    yield record


def check_parameter_consistency(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Validate that model parameters are consistent across runs.

    Parameters
    ----------
    records : list of dict
        Records returned by :func:`discover_processed_data`.
    cfg : dict
        Analysis configuration containing ``parameter_usage`` options.
    """
    param_cfg = cfg.get("parameter_usage", {}).get(
        "check_model_parameter_consistency", {}
    )
    if not param_cfg.get("enabled", False):
        return

    to_check = param_cfg.get("parameters_to_check", [])
    expected = param_cfg.get("expected_values", {})

    seen: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        group = "{plume}_{mode}".format(**rec.get("metadata", {}))
        params = rec.get("params", {})
        gdict = seen.setdefault(group, {})
        exp_group = expected.get(group, {})
        for p in to_check:
            if p not in params:
                continue
            val = params[p]
            if p in gdict and gdict[p] != val:
                raise ValueError(f"Inconsistent {p} in group {group}")
            gdict.setdefault(p, val)
            if p in exp_group and exp_group[p] != val:
                raise ValueError(
                    f"Parameter {p} in group {group} expected {exp_group[p]}, got {val}"
                )
