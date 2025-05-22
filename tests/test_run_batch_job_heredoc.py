import unittest
import os

class TestBatchJobHeredoc(unittest.TestCase):
    def test_uses_heredoc_or_tempfile(self):
        with open('run_batch_job.sh') as f:
            content = f.read()
        self.assertTrue('cat <<' in content or 'mktemp' in content,
                        'run_batch_job.sh should use a here-doc or temporary file for MATLAB commands')

    def test_loop_over_random_seeds(self):
        """Ensure the script iterates over RANDOM_SEEDS when generating MATLAB code."""
        with open('run_batch_job.sh') as f:
            content = f.read()
        self.assertIn('for ((i=0; i<${#RANDOM_SEEDS[@]}; i++))', content)

if __name__ == '__main__':
    unittest.main()
