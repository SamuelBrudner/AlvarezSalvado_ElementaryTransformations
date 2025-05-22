import os
import unittest

class TestRunMySimulationNoAddpath(unittest.TestCase):
    def test_run_my_simulation_has_no_addpath(self):
        with open(os.path.join('Code', 'run_my_simulation.m')) as f:
            content = f.read()
        self.assertNotIn('addpath', content,
                         'run_my_simulation.m should rely on startup.m for path setup')

if __name__ == '__main__':
    unittest.main()
