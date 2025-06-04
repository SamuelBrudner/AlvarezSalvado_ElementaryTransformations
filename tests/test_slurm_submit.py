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


def test_slurm_submit_usage_option():
    root = Path(__file__).resolve().parents[1]
    script = root / 'slurm_submit.sh'

    result = subprocess.run([
        'bash', str(script), '-h'
    ], capture_output=True, text=True, cwd=root)

    assert result.returncode == 0
    help_text = result.stdout
    for var in [
        'TRIAL_LENGTH',
        'ENVIRONMENT',
        'OUTPUT_DIR',
        'AGENTS_PER_CONDITION',
        'AGENTS_PER_JOB',
        'PARTITION',
        'TIME_LIMIT',
        'MEM_PER_TASK',
        'MAX_CONCURRENT',
        'EXP_NAME',
    ]:
        assert var in help_text


def test_slurm_submit_logs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'slurm_submit.sh'
    output = tmp_path / 'job.slurm'

    env = os.environ.copy()
    env['AGENTS_PER_CONDITION'] = '20'
    env['AGENTS_PER_JOB'] = '5'

    result = subprocess.run([
        'bash', str(script), str(output)
    ], capture_output=True, text=True, cwd=root, env=env, check=True)

    # stderr should contain the computed values and paths
    stderr = result.stderr
    assert 'total_jobs' in stderr or 'total jobs' in stderr.lower()
    assert 'array_upper' in stderr or 'array upper' in stderr.lower()
    assert str(output) in stderr


def test_slurm_scripts_have_valid_directives():
    root = Path(__file__).resolve().parents[1]
    for name in ['nav_job_final.slurm', 'run_simulation.slurm']:
        script = root / name
        text = script.read_text()
        assert text.endswith('\n')
        assert '#SBATCH' in text
        assert 'slurm_logs/' in text
