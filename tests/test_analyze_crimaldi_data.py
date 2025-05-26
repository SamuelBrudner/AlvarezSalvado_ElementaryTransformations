import os
import numpy as np
import h5py
import unittest

from Code.analyze_crimaldi_data import analyze_crimaldi_data

class TestAnalyzeCrimaldiData(unittest.TestCase):
    def setUp(self):
        self.tmpfile = 'tests/sample_crimaldi.hdf5'
        data = np.arange(27, dtype=np.float32).reshape(3,3,3)
        with h5py.File(self.tmpfile, 'w') as f:
            f.create_dataset('dataset_1', data=data)
        self.data = data

    def tearDown(self):
        if os.path.isfile(self.tmpfile):
            os.remove(self.tmpfile)

    def test_statistics(self):
        stats = analyze_crimaldi_data(self.tmpfile)
        self.assertEqual(stats['min'], float(self.data.min()))
        self.assertEqual(stats['max'], float(self.data.max()))
        self.assertAlmostEqual(stats['mean'], float(self.data.mean()))
        self.assertAlmostEqual(stats['std'], float(self.data.std()))
        self.assertAlmostEqual(stats['percentiles'][5], np.percentile(self.data, 5))

if __name__ == '__main__':
    unittest.main()
