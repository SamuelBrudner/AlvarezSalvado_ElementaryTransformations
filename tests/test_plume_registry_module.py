# ruff: noqa: E402
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.plume_registry import update_plume_registry


def _simple_load(path: Path) -> dict:
    data = {}
    current = None
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


def test_update_creates_entry(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    update_plume_registry("plume.h5", 1.0, 2.0, reg_path)
    data = _simple_load(reg_path)
    assert data["plume.h5"]["min"] == 1.0
    assert data["plume.h5"]["max"] == 2.0


def test_update_expands_range(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    update_plume_registry("plume.h5", 1.0, 2.0, reg_path)
    update_plume_registry("plume.h5", 0.5, 3.0, reg_path)
    data = _simple_load(reg_path)
    assert data["plume.h5"]["min"] == 0.5
    assert data["plume.h5"]["max"] == 3.0
