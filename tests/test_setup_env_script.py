import os
import subprocess
import shutil
import pytest


def test_setup_env_script_exists():
    assert os.path.isfile('setup_env.sh'), 'setup_env.sh does not exist'


def test_setup_env_script_contains_expected_commands():
    with open('setup_env.sh') as f:
        content = f.read()
    assert '--dev' in content
    assert 'conda env update' in content
    assert 'pre-commit install' in content
    assert 'setup_utils.sh' in content
    assert 'conda-lock' in content
    assert 'log ' in content


def test_setup_env_script_runs_idempotently():
    if shutil.which('conda') is None:
        pytest.skip('conda not available')
    cmd = 'source setup_env.sh --dev'
    result1 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result1.returncode == 0
    result2 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result2.returncode == 0
