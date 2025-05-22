import re
import unittest

class TestSlurmEnvVars(unittest.TestCase):
    def test_slurm_env_vars_in_run_batch_job(self):
        with open('run_batch_job.sh') as f:
            content = f.read()
        self.assertIn('#SBATCH --partition=${SLURM_PARTITION}', content)
        self.assertIn('#SBATCH --time=${SLURM_TIME}', content)
        self.assertIn('#SBATCH --mem-per-cpu=${SLURM_MEM}', content)
        self.assertIn('#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}', content)
        self.assertRegex(content, r'--array=0-\$\(\(TOTAL_JOBS-1\)\)%\${SLURM_ARRAY_CONCURRENT}')

if __name__ == '__main__':
    unittest.main()
