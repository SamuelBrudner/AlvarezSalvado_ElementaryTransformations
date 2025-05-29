import sys
import types
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


def test_stats_match_yaml(monkeypatch):
    fake_np = types.SimpleNamespace(min=lambda a: 0, max=lambda a: 0, full_like=lambda arr, v: arr)
    monkeypatch.setitem(sys.modules, 'numpy', fake_np)
    import Code.plume_utils as pu
    stats_yaml = load_yaml_simple('configs/plume_intensity_stats.yaml')
    assert pu.get_intensity_stats() == stats_yaml
