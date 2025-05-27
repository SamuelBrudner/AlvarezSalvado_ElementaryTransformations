import os
import sys
import importlib
import types

fake_np = types.SimpleNamespace(array=lambda x: x)
fake_h5py = types.SimpleNamespace(File=lambda *a, **k: None)
fake_scipy = types.SimpleNamespace(io=types.SimpleNamespace(loadmat=lambda p: {"all_intensities": [1]}))
sys.modules['numpy'] = fake_np
sys.modules['h5py'] = fake_h5py
sys.modules['scipy'] = fake_scipy
sys.modules['scipy.io'] = fake_scipy.io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code import compare_intensity_stats as cis

def test_work_dir_passed_to_video_loader(monkeypatch, tmp_path):
    mfile = tmp_path / 'nested' / 'script.m'
    mfile.parent.mkdir()
    mfile.write_text('disp("hi")')
    captured = {}

    def fake_video(contents, matlab_exec_path='matlab', px_per_mm=None, frame_rate=None, work_dir=None):
        captured['work_dir'] = work_dir
        return [1]

    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', lambda *a, **k: [_ for _ in ()])
    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_video)

    cis.load_intensities(str(mfile), plume_type='video')
    assert captured['work_dir'] == str(mfile.parent)
