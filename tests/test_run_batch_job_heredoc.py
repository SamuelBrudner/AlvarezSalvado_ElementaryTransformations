import unittest
import os

class TestBatchJobHeredoc(unittest.TestCase):
    def test_uses_heredoc_or_tempfile(self):
        with open('run_batch_job.sh') as f:
            content = f.read()
        self.assertTrue('cat <<' in content or 'mktemp' in content,
                        'run_batch_job.sh should use a here-doc or temporary file for MATLAB commands')

if __name__ == '__main__':
    unittest.main()
