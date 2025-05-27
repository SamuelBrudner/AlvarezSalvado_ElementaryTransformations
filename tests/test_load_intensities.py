import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code import compare_intensity_stats as cis


def test_h5_path_uses_crimaldi_loader(monkeypatch, tmp_path):
    hfile = tmp_path / 'sample.h5'
    hfile.write_text('dummy')
    captured = {}

    def fake_crim(path):
        captured['path'] = path
        return np.array([1.0])

    def fake_video(*args, **kwargs):
        raise AssertionError('video loader should not be called')

    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', fake_crim)
    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_video)

    arr = cis.load_intensities(str(hfile))
    assert np.array_equal(arr, np.array([1.0]))
    assert captured['path'] == str(hfile)


def test_m_file_uses_video_loader(monkeypatch, tmp_path):
    mfile = tmp_path / 'script.m'
    mfile.write_text('disp("hi")')
    captured = {}

    def fake_video(contents, matlab_exec_path='matlab', orig_script_path=None):
        captured['contents'] = contents
        captured['matlab'] = matlab_exec_path
        return np.array([2.0, 3.0])

    def fake_crim(*args, **kwargs):
        raise AssertionError('crimaldi loader should not be called')

    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', fake_crim)
    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_video)

    arr = cis.load_intensities(str(mfile))
    assert np.array_equal(arr, np.array([2.0, 3.0]))
    assert captured['contents'] == mfile.read_text()
    assert captured['matlab'] == 'matlab'


def test_work_dir_passed_to_video_loader(monkeypatch, tmp_path):
    mfile = tmp_path / 'nested' / 'script.m'
    mfile.parent.mkdir()
    mfile.write_text('disp("hi")')
    captured = {}

    def fake_video(contents, matlab_exec_path='matlab', px_per_mm=None, frame_rate=None, work_dir=None, orig_script_path=None):
        captured['work_dir'] = work_dir
        return [1]

    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', lambda *a, **k: [_ for _ in ()])
    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_video)

    cis.load_intensities(str(mfile), plume_type='video')
    assert captured['work_dir'] == str(mfile.parent)


def test_orig_script_path_passed(monkeypatch, tmp_path):
    mfile = tmp_path / 'script.m'
    mfile.write_text('disp("hi")')
    captured = {}

    def fake_video(
        contents,
        matlab_exec_path='matlab',
        px_per_mm=None,
        frame_rate=None,
        work_dir=None,
        orig_script_path=None,
    ):
        captured['orig_script_path'] = orig_script_path
        return np.array([5.0])

    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', lambda *a, **k: [_ for _ in ()])
    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_video)

    cis.load_intensities(str(mfile), plume_type='video')
    assert captured['orig_script_path'] == str(mfile)
