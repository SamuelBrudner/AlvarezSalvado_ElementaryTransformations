import importlib
import sys
import os
import subprocess
import tempfile
import types


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DummyArray(list):
    def flatten(self) -> "DummyArray":
        return self


def fake_loadmat(path):
    return {"all_intensities": [1]}


def test_work_dir_inserts_cd(monkeypatch, tmp_path):
    captured = {}

    fake_np = types.SimpleNamespace(asarray=lambda x: DummyArray(x))
    fake_scipy_io = types.SimpleNamespace(loadmat=fake_loadmat)
    fake_h5py = types.SimpleNamespace(File=lambda *a, **k: None)
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: None
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(io=fake_scipy_io))
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    vi = importlib.reload(importlib.import_module("Code.video_intensity"))
    monkeypatch.setattr(vi, "find_matlab_executable", lambda p=None: p or "matlab")

    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        fh = orig_ntf(*args, **kwargs)
        captured["path"] = fh.name
        return fh

    def fake_run(cmd, capture_output, text, **kwargs):
        with open(captured["path"]) as fh:
            captured["contents"] = fh.read()
        mat_file = tmp_path / "dummy.mat"
        mat_file.write_bytes(b"")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"TEMP_MAT_FILE_SUCCESS:{mat_file}\n", stderr=""
        )

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    monkeypatch.setattr(subprocess, "run", fake_run)

    arr = vi.get_intensities_from_video_via_matlab(
        'disp("hi")', "matlab", work_dir=str(tmp_path)
    )
    assert arr == [1]
    assert captured["contents"].splitlines()[0] == f"cd('{tmp_path}')"
