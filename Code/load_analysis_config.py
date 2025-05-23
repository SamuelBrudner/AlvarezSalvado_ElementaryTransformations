"""Load analysis configuration from a YAML file."""

from __future__ import annotations

import yaml
from typing import Any, Dict
from pathlib import Path


def load_analysis_config(path: str | Path) -> Dict[str, Any]:
    """Load the analysis configuration.

    Parameters
    ----------
    path : str or Path
        Location of the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    content = file_path.read_text()
    return yaml.safe_load(content)
