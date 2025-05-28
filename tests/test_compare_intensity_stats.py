import os
import sys
import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code import compare_intensity_stats as cis
import csv
import json


def create_hdf5(path, data):
    with h5py.File(path, 'w') as f:
        f.create_dataset('dataset_1', data=data)


def test_compare_intensity_stats_table(tmp_path, capsys):
    f1 = tmp_path / 'a.h5'
    f2 = tmp_path / 'b.h5'
    arr1 = np.array([1, 2, 3], dtype=float)
    arr2 = np.array([4, 5], dtype=float)
    create_hdf5(f1, arr1)
    create_hdf5(f2, arr2)

    cis.main(['A', str(f1), 'B', str(f2)])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith('identifier')
    assert out[1].split('\t')[0] == 'A'
    assert out[2].split('\t')[0] == 'B'

    # compute expected values for a simple sanity check
    mean_a = arr1.mean()
    mean_b = arr2.mean()
    assert f"{mean_a:.3f}" in out[1]
    assert f"{mean_b:.3f}" in out[2]


def test_compare_intensity_stats_video_vs_crimaldi(monkeypatch, tmp_path, capsys):
    hfile = tmp_path / 'c.h5'
    arr_crim = np.array([10.0, 20.0], dtype=float)
    create_hdf5(hfile, arr_crim)

    script = tmp_path / 'video_script.m'
    script.write_text("disp('hi')")
    arr_vid = np.array([1.0, 2.0, 3.0], dtype=float)

    captured = {}

    def fake_func(s, m='matlab', orig_script_path=None):
        captured['matlab_exec'] = m
        return arr_vid

    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_func)

    cis.main([
        'VID',
        'video',
        str(script),
        'CRIM',
        'crimaldi',
        str(hfile),
    ])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith('identifier')
    assert out[1].split('\t')[0] == 'VID'
    assert out[2].split('\t')[0] == 'CRIM'
    assert f"{arr_vid.mean():.3f}" in out[1]
    assert f"{arr_crim.mean():.3f}" in out[2]
    assert captured['matlab_exec'] == 'matlab'

def test_matlab_exec_option(monkeypatch, tmp_path):
    script = tmp_path / 'video.m'
    script.write_text('disp("hi")')
    arr_vid = np.array([1.0], dtype=float)
    captured = {}

    def fake_func(s, m='matlab', orig_script_path=None):
        captured['matlab_exec'] = m
        return arr_vid

    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_func)

    cis.main([
        'V',
        'video',
        str(script),
        '--matlab_exec', '/opt/matlab',
    ])
    assert captured['matlab_exec'] == '/opt/matlab'


def test_csv_output_and_directory_creation(monkeypatch, tmp_path):
    hfile = tmp_path / 'c.h5'
    arr_crim = np.array([5.0, 15.0], dtype=float)
    create_hdf5(hfile, arr_crim)

    script = tmp_path / 'video_script.m'
    script.write_text("disp('hi')")
    arr_vid = np.array([1.0, 2.0], dtype=float)

    monkeypatch.setattr(
        cis,
        'get_intensities_from_video_via_matlab',
        lambda s, m='matlab', orig_script_path=None: arr_vid,
    )

    csv_path = tmp_path / 'nested' / 'results' / 'stats.csv'
    cis.main([
        'VID',
        'video',
        str(script),
        'CRIM',
        'crimaldi',
        str(hfile),
        '--csv',
        str(csv_path),
    ])

    assert csv_path.exists()
    assert csv_path.parent.is_dir()

    with csv_path.open(newline='') as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == 'identifier'
    assert rows[1][0] == 'VID'
    assert rows[2][0] == 'CRIM'

    stats_vid = cis.calculate_intensity_stats_dict(arr_vid)
    stats_crim = cis.calculate_intensity_stats_dict(arr_crim)

    def approx_float(x):
        return pytest.approx(float(x), rel=1e-3)

    assert float(rows[1][1]) == approx_float(stats_vid['mean'])
    assert float(rows[2][1]) == approx_float(stats_crim['mean'])


def test_json_output_and_directory_creation(monkeypatch, tmp_path):
    hfile = tmp_path / 'c.h5'
    arr_crim = np.array([5.0, 15.0], dtype=float)
    create_hdf5(hfile, arr_crim)

    script = tmp_path / 'video_script.m'
    script.write_text("disp('hi')")
    arr_vid = np.array([1.0, 2.0], dtype=float)

    monkeypatch.setattr(
        cis,
        'get_intensities_from_video_via_matlab',
        lambda s, m='matlab', orig_script_path=None: arr_vid,
    )

    json_path = tmp_path / 'nested' / 'results' / 'stats.json'
    cis.main([
        'VID',
        'video',
        str(script),
        'CRIM',
        'crimaldi',
        str(hfile),
        '--json',
        str(json_path),
    ])

    assert json_path.exists()
    assert json_path.parent.is_dir()

    data = json.loads(json_path.read_text())

    assert data[0]['identifier'] == 'VID'
    assert data[1]['identifier'] == 'CRIM'

    stats_vid = cis.calculate_intensity_stats_dict(arr_vid)
    stats_crim = cis.calculate_intensity_stats_dict(arr_crim)

    assert data[0]['statistics']['mean'] == pytest.approx(stats_vid['mean'])
    assert data[1]['statistics']['mean'] == pytest.approx(stats_crim['mean'])



def test_video_script_does_not_prepend_video_file(monkeypatch, tmp_path):
    script = tmp_path / 'vid.m'
    script.write_text("disp('x')")
    captured = {}

    def fake_get(script_contents, m='matlab', orig_script_path=None, **kwargs):
        captured['script'] = script_contents
        return np.array([1.0], dtype=float)

    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_get)
    cis.load_video_script_intensities(str(script), 'matlab')
    assert not captured['script'].startswith('video_file =')

import logging


def test_logging_messages(monkeypatch, tmp_path, caplog):
    hfile = tmp_path / 'log.h5'
    arr = np.array([1.0, 2.0], dtype=float)
    create_hdf5(hfile, arr)

    # avoid actual file reading
    monkeypatch.setattr(cis, 'get_intensities_from_crimaldi', lambda p: arr)

    with caplog.at_level(logging.INFO):
        cis.main(['LOG', str(hfile)])

    messages = [rec.getMessage() for rec in caplog.records]
    assert any('Loading intensities from' in m for m in messages)
    assert any('Dataset LOG has length' in m for m in messages)

def test_debug_log_level_shows_matlab_stdout(monkeypatch, tmp_path, caplog):
    script = tmp_path / 'vid.m'
    script.write_text("disp('x')")

    def fake_get(script_contents, m='matlab', orig_script_path=None, **kwargs):
        logging.getLogger('Code.video_intensity').debug('MATLAB stdout:\nhello')
        return np.array([1.0], dtype=float)

    monkeypatch.setattr(cis, 'get_intensities_from_video_via_matlab', fake_get)

    with caplog.at_level(logging.DEBUG):
        cis.main(['VID', 'video', str(script), '--log-level', 'DEBUG'])

    messages = [rec.getMessage() for rec in caplog.records]
    assert any('MATLAB stdout:' in m for m in messages)

