import re
import unittest

class TestRunBatchNewWrapper(unittest.TestCase):
    def test_uses_wrapper(self):
        with open('run_batch_new.sh') as f:
            content = f.read()
        self.assertIn('run_batch_job_wrapper(', content,
                      'run_batch_new.sh should call run_batch_job_wrapper')
        self.assertNotIn('run_batch_job($SLURM_ARRAY_TASK_ID', content,
                         'run_batch_new.sh should not call run_batch_job with missing args')

if __name__ == '__main__':
    unittest.main()
