"""Validate exported simulation results against expected schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def validate_exported_data(path: str | Path, cfg: Dict[str, Any] | None = None) -> None:
    """Validate exported trajectories, params, and summary files.

    Parameters
    ----------
    path : str or Path
        Directory containing ``trajectories.csv``, ``params.json`` and
        ``summary.json``.
    cfg : dict, optional
        Analysis configuration that may define required trajectory columns under
        ``trajectory_processing.required_columns``.

    Raises
    ------
    ValueError
        If required fields are missing or have the wrong type.
    """
    run_dir = Path(path)

    csv_file = run_dir / "trajectories.csv"
    summary_file = run_dir / "summary.json"
    params_file = run_dir / "params.json"

    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)
    if not summary_file.is_file():
        raise FileNotFoundError(summary_file)
    if not params_file.is_file():
        raise FileNotFoundError(params_file)

    df = pd.read_csv(csv_file)

    required_cols = None
    if cfg is not None:
        required_cols = cfg.get("trajectory_processing", {}).get("required_columns")

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column {col} must be numeric")

    with open(summary_file, "r") as f:
        summary = json.load(f)

    if not isinstance(summary.get("successrate"), (int, float)):
        raise TypeError("successrate must be numeric")
    if not 0 <= float(summary["successrate"]) <= 1:
        raise ValueError("successrate out of range")
    if not isinstance(summary.get("latency"), list) or not all(
        isinstance(v, (int, float)) for v in summary["latency"]
    ):
        raise TypeError("latency must be a list of numbers")
    if not isinstance(summary.get("n_trials"), int):
        raise TypeError("n_trials must be integer")
    if not isinstance(summary.get("timesteps"), int):
        raise TypeError("timesteps must be integer")

    with open(params_file, "r") as f:
        json.load(f)
