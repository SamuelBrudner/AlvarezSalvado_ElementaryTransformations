from __future__ import annotations

import sys
import types
from pathlib import Path

from Code import rotate_video


class _Frame:
    def __init__(self, data):
        self.data = data
        self.ndim = 2

    def reshape(self, *shape):
        assert shape == (-1,)
        return self.data


def _simple_yaml(path: Path) -> dict:
    data: dict[str, dict[str, float]] = {}
    current = None
    with open(path, "r") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw.startswith("  "):
                current = stripped.rstrip(":")
                data[current] = {}
            else:
                k, v = stripped.split(":", 1)
                data[current][k.strip()] = float(v)
    return data


def test_plume_registry_video_to_hdf5(monkeypatch, tmp_path):
    frames = [_Frame([0, 1, 2]), _Frame([3, 4, 5])]

    class FakeImageIO:
        @staticmethod
        def imiter(path):
            return frames

    class DummyFile:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def create_dataset(self, name, data):
            self.data = data

    fake_np = types.SimpleNamespace(concatenate=lambda xs: sum(xs, []))
    fake_h5py = types.SimpleNamespace(File=DummyFile)

    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setattr(rotate_video, "imageio", FakeImageIO)

    rotate_video.video_to_hdf5("in.avi", tmp_path / "out.h5")

    registry_path = Path("configs") / "plume_registry.yaml"
    registry = _simple_yaml(registry_path)
    assert "out.h5" in registry
    assert "min" in registry["out.h5"]
    assert "max" in registry["out.h5"]
