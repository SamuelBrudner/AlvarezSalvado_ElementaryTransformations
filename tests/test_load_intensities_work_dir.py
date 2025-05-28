import importlib
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_work_dir_passed_to_video_loader(monkeypatch, tmp_path):
    fake_np = types.SimpleNamespace(array=lambda x: x)
    fake_h5py = types.SimpleNamespace(File=lambda *a, **k: None)
    fake_scipy = types.SimpleNamespace(
        io=types.SimpleNamespace(loadmat=lambda p: {"all_intensities": [1]})
    )
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy.io)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    cis = importlib.reload(importlib.import_module("Code.compare_intensity_stats"))
    mfile = tmp_path / "nested" / "script.m"
    mfile.parent.mkdir()
    mfile.write_text('disp("hi")')
    captured = {}

    def fake_video(
        contents,
        matlab_exec_path="matlab",
        px_per_mm=None,
        frame_rate=None,
        work_dir=None,
        orig_script_path=None,
    ):
        captured["work_dir"] = work_dir
        return [1]

    monkeypatch.setattr(
        cis, "get_intensities_from_crimaldi", lambda *a, **k: [_ for _ in ()]
    )
    monkeypatch.setattr(cis, "get_intensities_from_video_via_matlab", fake_video)

    cis.load_intensities(str(mfile), plume_type="video")
    assert captured["work_dir"] == str(mfile.parent)
