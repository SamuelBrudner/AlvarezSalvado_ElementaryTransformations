import os
import numpy as np
import h5py
import unittest

from Code.analyze_crimaldi_data import get_intensities_from_crimaldi

class TestGetIntensitiesFromCrimaldi(unittest.TestCase):
    def setUp(self):
        self.tmpfile = 'tests/sample_crimaldi.hdf5'
        data = np.arange(27, dtype=np.float32).reshape(3,3,3)
        with h5py.File(self.tmpfile, 'w') as f:
            f.create_dataset('dataset_1', data=data)
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
            get_intensities_from_crimaldi(self.tmpfile, 'missing')

if __name__ == '__main__':
    unittest.main()
