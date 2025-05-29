# ruff: noqa: E402
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.plume_registry import load_registry, update_plume_registry


def test_update_creates_entry(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    update_plume_registry("plume.h5", 1.0, 2.0, reg_path)
    data = load_registry(reg_path)
    assert data["plume.h5"]["min"] == 1.0
    assert data["plume.h5"]["max"] == 2.0


def test_update_expands_range(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    update_plume_registry("plume.h5", 1.0, 2.0, reg_path)
    update_plume_registry("plume.h5", 0.5, 3.0, reg_path)
    data = load_registry(reg_path)
    assert data["plume.h5"]["min"] == 0.5
    assert data["plume.h5"]["max"] == 3.0


def test_load_returns_empty_for_missing(tmp_path):
    reg_path = tmp_path / "missing.yaml"
    assert load_registry(reg_path) == {}
