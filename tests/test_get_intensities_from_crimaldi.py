import logging
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from Code.analyze_crimaldi_data import \
    get_intensities_from_crimaldi  # noqa: E402


class TestGetIntensitiesFromCrimaldi(unittest.TestCase):
    def setUp(self):
        self.tmpfile = "tests/sample_crimaldi.hdf5"
        data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        with h5py.File(self.tmpfile, "w") as f:
            f.create_dataset("dataset_1", data=data)
        self.data = data

    def tearDown(self):
        if os.path.isfile(self.tmpfile):
            os.remove(self.tmpfile)

    def test_returns_flat_array(self):
        arr = get_intensities_from_crimaldi(self.tmpfile)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.size, self.data.size)
        np.testing.assert_array_equal(arr, self.data.flatten())

    def test_missing_dataset_raises(self):
        with self.assertRaises(KeyError):
            get_intensities_from_crimaldi(self.tmpfile, "missing")


def test_logs_autoselected_dataset(tmp_path, caplog):
    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    hfile = tmp_path / "sample.hdf5"
    with h5py.File(hfile, "w") as f:
        f.create_dataset("dataset_1", data=data)

    with caplog.at_level(logging.INFO):
        arr = get_intensities_from_crimaldi(str(hfile))

    assert arr.size == data.size
    assert any("Using dataset: dataset_1" in r.getMessage() for r in caplog.records)


if __name__ == "__main__":
    unittest.main()
