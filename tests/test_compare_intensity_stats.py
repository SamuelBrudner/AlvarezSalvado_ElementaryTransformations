import os
import sys
import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code import compare_intensity_stats as cis
import csv


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

    def fake_func(s, m='matlab'):
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

    def fake_func(s, m='matlab'):
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
        lambda s, m='matlab': arr_vid,
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

