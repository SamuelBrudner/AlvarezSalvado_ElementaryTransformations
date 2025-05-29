"""Utilities for plume intensity scaling.

This module provides helpers to rescale plume intensity arrays so that they
match reference ranges derived from experimental data. Statistics for the
reference plumes are stored in ``configs/plume_intensity_stats.yaml`` relative
to the repository root.

Example
-------
>>> import numpy as np
>>> from Code import plume_utils
>>> arr = np.linspace(0, 1, num=5)
>>> plume_utils.rescale_to_crim_range(arr)
array([...])
"""

from __future__ import annotations


from typing import Any, Dict
from pathlib import Path

import numpy as np
import h5py
import imageio.v2 as imageio

__all__ = ["get_intensity_stats", "rescale_to_crim_range", "get_plume_frame"]

try:  # pragma: no cover - fall back if PyYAML is unavailable
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - lightweight YAML parser
    import types

    def _minimal_load(path: Path) -> Dict[str, Dict[str, float]]:
        """Very small YAML subset parser."""
        data: Dict[str, Dict[str, float]] = {}
        current: str | None = None
        for raw_line in path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith("  "):
                current = stripped.rstrip(":")
                data[current] = {}
            else:
                key, value = stripped.split(":", 1)
                data[current][key] = float(value)
        return data

    yaml = types.SimpleNamespace(safe_load=lambda f: _minimal_load(Path(f.name)))


_YAML_PATH = Path(__file__).resolve().parent.parent / "configs" / "plume_intensity_stats.yaml"
with _YAML_PATH.open("r", encoding="utf-8") as fh:
    _INTENSITY_STATS: Dict[str, Dict[str, Any]] = yaml.safe_load(fh)


def get_intensity_stats(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Return stored intensity statistics for the default plumes.

    Parameters
    ----------
    path:
        Optional path to a YAML file with intensity statistics. When ``None``,
        returns cached stats from ``configs/plume_intensity_stats.yaml``.
    """
    if path is None:
        return _INTENSITY_STATS
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _rescale(arr: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    src_min = float(np.min(arr))
    src_max = float(np.max(arr))
    if src_max == src_min:
        return np.full_like(arr, target_min)
    scale = (target_max - target_min) / (src_max - src_min)
    return (arr - src_min) * scale + target_min


def rescale_to_crim_range(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale ``arr`` to match the Crimaldi min and max."""
    stats = get_intensity_stats()
    return _rescale(arr, stats["CRIM"]["min"], stats["CRIM"]["max"])


def get_plume_frame(
    plume: str,
    frame: int = 0,
    *,
    config_path: str | Path = "configs/project_paths.yaml",
) -> np.ndarray:
    """Return a single frame from a configured plume video or HDF5 file.

    Parameters
    ----------
    plume:
        ``"crimaldi"`` to load from the HDF5 plume or ``"video"`` for the smoke
        video.
    frame:
        Index of the frame to load.
    config_path:
        Optional path to ``project_paths.yaml`` specifying ``data.crimaldi`` and
        ``data.video``. When absent, defaults to canonical filenames under
        ``data/``.

    Returns
    -------
    numpy.ndarray
        Array of the requested frame.

    Examples
    --------
    >>> import numpy as np
    >>> from Code import plume_utils
    >>> frame0 = plume_utils.get_plume_frame("crimaldi", 0)
    >>> isinstance(frame0, np.ndarray)
    True
    """

    config_file = Path(config_path)
    default_h5 = Path("data/10302017_10cms_bounded_2.h5")
    default_vid = Path("data/smoke_video.avi")

    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        h5_path = Path(data_cfg.get("crimaldi", default_h5))
        vid_path = Path(data_cfg.get("video", default_vid))
    else:
        h5_path = default_h5
        vid_path = default_vid

    if plume == "crimaldi":
        with h5py.File(h5_path, "r") as f:
            return np.asarray(f["intensity"][frame])
    elif plume == "video":
        reader = imageio.get_reader(vid_path)
        try:
            return np.asarray(reader.get_data(frame))
        finally:
            reader.close()
    raise ValueError(f"unknown plume '{plume}'")

