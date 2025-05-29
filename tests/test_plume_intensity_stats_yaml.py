import sys
import types
from pathlib import Path

import pytest


def load_yaml_simple(path):
    data = {}
    with open(path, "r") as f:
        current = None
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith("  "):
                key = stripped.rstrip(":")
                current = key
                data[current] = {}
            else:
                k, v = stripped.split(":", 1)
                data[current][k] = float(v)
    return data


def _patch_numpy(monkeypatch):
    fake_np = types.SimpleNamespace(min=lambda a: 0, max=lambda a: 0, full_like=lambda arr, v: arr)
    monkeypatch.setitem(sys.modules, "numpy", fake_np)


def test_stats_match_yaml(monkeypatch):
    _patch_numpy(monkeypatch)
    import importlib
    import Code.plume_utils as pu
    importlib.reload(pu)
    stats_yaml = load_yaml_simple("configs/plume_intensity_stats.yaml")
    assert pu.get_intensity_stats() == stats_yaml


def test_get_stats_from_explicit_path(monkeypatch, tmp_path):
    _patch_numpy(monkeypatch)
    import importlib
    import Code.plume_utils as pu
    importlib.reload(pu)
    src = Path("configs/plume_intensity_stats.yaml")
    tmp_copy = tmp_path / "copy.yaml"
    tmp_copy.write_text(src.read_text())
    expected = load_yaml_simple(src)
    assert pu.get_intensity_stats(tmp_copy) == expected


def test_get_stats_file_not_found(monkeypatch, tmp_path):
    _patch_numpy(monkeypatch)
    import importlib
    import Code.plume_utils as pu
    importlib.reload(pu)
    with pytest.raises(FileNotFoundError):
        pu.get_intensity_stats(tmp_path / "missing.yaml")


def test_cached_stats_reused(monkeypatch):
    _patch_numpy(monkeypatch)
    import importlib
    import Code.plume_utils as pu
    importlib.reload(pu)
    stats = pu.get_intensity_stats()

    def fail(*args, **kwargs):
        raise AssertionError("should not reload")

    monkeypatch.setattr(pu.yaml, "safe_load", fail)
    assert pu.get_intensity_stats() == stats
