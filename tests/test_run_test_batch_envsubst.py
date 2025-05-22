import os
import unittest

class TestRunTestBatchEnvsubst(unittest.TestCase):
    def test_wrapper_uses_envsubst(self):
        self.assertTrue(os.path.isfile('run_test_batch.sh'), 'run_test_batch.sh should exist')
        with open('run_test_batch.sh') as f:
            content = f.read()
        self.assertIn('envsubst', content, 'run_test_batch.sh should use envsubst to expand variables')

if __name__ == '__main__':
    unittest.main()
