"""Load analysis configuration from a YAML file.

Examples
--------
>>> from Code.load_analysis_config import load_analysis_config
>>> cfg = load_analysis_config("configs/example_analysis.yaml")
>>> isinstance(cfg, dict)
True
"""

from __future__ import annotations

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback if PyYAML is missing
    import json
    import re  # noqa: F401 - fallback parser does not use regex
    import types

    def _minimal_safe_load(text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            data = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.lower() in ("true", "false"):
                    data[key] = value.lower() == "true"
                else:
                    try:
                        if "." in value:
                            data[key] = float(value)
                        else:
                            data[key] = int(value)
                    except ValueError:
                        data[key] = value
            return data

    yaml = types.SimpleNamespace(safe_load=_minimal_safe_load)

from pathlib import Path
from typing import Any, Dict


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
