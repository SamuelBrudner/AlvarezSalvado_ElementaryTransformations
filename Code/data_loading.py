"""Utilities for loading exported simulation data.

Examples
--------
>>> from Code.data_loading import load_trajectories
>>> df = load_trajectories("run/trajectories.csv")
>>> len(df)
0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_trajectories(
    path: str | Path, cfg: Dict[str, Any] | None = None
) -> pd.DataFrame:
    """Load trajectories from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the trajectories CSV file.
    cfg : dict, optional
        Analysis configuration dictionary. If provided and it contains the key
        ``trajectory_processing.required_columns``, only those columns are
        returned.

    Returns
    -------
    pandas.DataFrame
        DataFrame of trajectory data.
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    if cfg is not None:
        proc_cfg = cfg.get("trajectory_processing", {})
        columns = proc_cfg.get("required_columns")
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")
            df = df[columns]

    return df
