import os
import importlib
import sys
import tempfile
import subprocess
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class DummyArray(list):
    def flatten(self):
        return self


class FakeDataset(list):
    def __getitem__(self, item):
        assert item == ()
        return self


def fake_loadmat(path):
    raise ValueError("unsupported format")


class FakeH5File:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def __contains__(self, name):
        return name in self._data

    def __getitem__(self, name):
        return FakeDataset(self._data[name])


def test_valueerror_falls_back_to_h5py(monkeypatch, tmp_path):
    fake_np = types.SimpleNamespace(asarray=lambda x: DummyArray(x))
    fake_scipy_io = types.SimpleNamespace(loadmat=fake_loadmat)
    fake_h5py = types.SimpleNamespace(File=lambda *a, **k: FakeH5File({"all_intensities": [5, 6]}))
    fake_yaml = types.SimpleNamespace(safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: None)

    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(io=fake_scipy_io))
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    vi = importlib.reload(importlib.import_module("Code.video_intensity"))
    monkeypatch.setattr(vi, "find_matlab_executable", lambda p=None: p or "matlab")

    def fake_run(cmd, capture_output, text, **kwargs):
        mat_file = tmp_path / "dummy.mat"
        mat_file.write_bytes(b"")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"TEMP_MAT_FILE_SUCCESS:{mat_file}\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    arr = vi.get_intensities_from_video_via_matlab('disp("hi")', 'matlab')
    assert arr == [5, 6]

