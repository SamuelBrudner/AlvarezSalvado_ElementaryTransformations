import os
import subprocess


def test_setup_env_script_exists():
    assert os.path.isfile('setup_env.sh'), 'setup_env.sh does not exist'


def test_setup_env_script_contains_expected_commands():
    with open('setup_env.sh') as f:
        content = f.read()
    assert '--dev' in content
    assert 'conda create' in content
    assert 'pre-commit install' in content


def test_setup_env_script_runs_idempotently():
    cmd = 'source setup_env.sh --dev'
    result1 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result1.returncode == 0
    result2 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result2.returncode == 0
