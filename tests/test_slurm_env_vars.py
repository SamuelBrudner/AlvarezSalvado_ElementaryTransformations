import re
import unittest

class TestSlurmEnvVars(unittest.TestCase):
    def test_slurm_env_vars_in_run_batch_job(self):
        with open('run_batch_job.sh') as f:
            content = f.read()
        self.assertIn('#SBATCH --partition=day', content)
        self.assertIn('#SBATCH --time=6:00:00', content)
        self.assertIn('#SBATCH --mem-per-cpu=16G', content)
        self.assertIn('#SBATCH --cpus-per-task=1', content)
        self.assertRegex(content, r'--array=0-\$\(\(TOTAL_JOBS-1\)\)%\${SLURM_ARRAY_CONCURRENT}')

if __name__ == '__main__':
    unittest.main()
