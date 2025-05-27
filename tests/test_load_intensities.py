import os
import sys
import numpy as np

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

    def fake_video(contents, matlab_exec_path='matlab'):
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
