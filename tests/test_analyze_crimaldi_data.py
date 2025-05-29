# ruff: noqa: E402
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from Code.analyze_crimaldi_data import (
    analyze_crimaldi_data,
    get_intensities_from_crimaldi,
)


def test_statistics(tmp_path):
    tmpfile = tmp_path / "sample_crimaldi.hdf5"
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    with h5py.File(tmpfile, "w") as f:
        f.create_dataset("dataset_1", data=data)

    stats = analyze_crimaldi_data(str(tmpfile))
    assert stats["min"] == float(data.min())
    assert stats["max"] == float(data.max())
    assert stats["mean"] == pytest.approx(float(data.mean()))
    assert stats["std"] == pytest.approx(float(data.std()))
    assert stats["percentiles"][5] == pytest.approx(np.percentile(data, 5))


def test_get_intensities_logs_dataset(tmp_path, capsys):
    tmpfile = tmp_path / "intensity.hdf5"
    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    with h5py.File(tmpfile, "w") as f:
        f.create_dataset("dataset_a", data=data)

    result = get_intensities_from_crimaldi(str(tmpfile))
    captured = capsys.readouterr()
    assert "Using dataset: dataset_a" in captured.err
    assert result.shape == (8,)
