import os
import sys
import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.compare_intensity_stats import main


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

    main(['A', str(f1), 'B', str(f2)])
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith('identifier')
    assert out[1].split('\t')[0] == 'A'
    assert out[2].split('\t')[0] == 'B'

    # compute expected values for a simple sanity check
    mean_a = arr1.mean()
    mean_b = arr2.mean()
    assert f"{mean_a:.3f}" in out[1]
    assert f"{mean_b:.3f}" in out[2]
