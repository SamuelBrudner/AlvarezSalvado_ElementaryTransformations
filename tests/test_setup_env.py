import subprocess
from pathlib import Path


def test_setup_env_usage_option():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'setup_env.sh'
    result = subprocess.run(['bash', str(script), '-h'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Usage:' in result.stdout


def test_setup_env_requires_dev():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'setup_env.sh'
    result = subprocess.run(['bash', str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert 'Usage:' in combined


def test_setup_env_prints_conda_command():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'setup_env.sh'
    result = subprocess.run(['bash', str(script), '--dev', '--print'], capture_output=True, text=True)
    assert result.returncode == 0
    cmd = result.stdout.strip()
    assert cmd.startswith('conda env create')
    assert '--prefix dev_env' in cmd
    env_file = Path(__file__).resolve().parents[1] / 'configs' / 'environment.yaml'
    assert f'--file {env_file}' in cmd
