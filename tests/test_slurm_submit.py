import os
import subprocess
from pathlib import Path



def test_template_exists():
    root = Path(__file__).resolve().parents[1]
    template = root / 'slurm_job_template.slurm'

    assert template.is_file()

    content = template.read_text()
    assert 'navigation_model_vec' in content


def test_slurm_submit_creates_batch_file(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'slurm_submit.sh'
    output = tmp_path / 'job.slurm'

    env = os.environ.copy()
    env['AGENTS_PER_CONDITION'] = '20'
    env['AGENTS_PER_JOB'] = '5'
    env['EXP_NAME'] = 'test'

    subprocess.run(['bash', str(script), str(output)], capture_output=True, text=True, cwd=root, env=env, check=True)

    assert output.is_file()
    content = output.read_text()
    assert '#SBATCH --array=0-15%100' in content
    assert '#SBATCH --job-name=test_sim' in content

