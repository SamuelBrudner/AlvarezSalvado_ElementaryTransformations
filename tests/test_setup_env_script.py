import os
import shutil
import subprocess
import pytest


def test_setup_env_script_exists():
    assert os.path.isfile('setup_env.sh'), 'setup_env.sh does not exist'


def test_setup_env_script_contains_expected_commands():
    with open('setup_env.sh') as f:
        content = f.read()
    assert '--dev' in content
    assert 'source "${SCRIPT_DIR}/setup_utils.sh"' in content
    assert 'conda-lock' in content
    assert 'pre-commit install' in content


def test_setup_env_script_runs_idempotently():
    if shutil.which('conda') is None:
        pytest.skip('Conda not available')
    cmd = 'bash -c "source setup_env.sh --dev"'
    result1 = subprocess.run(cmd, shell=True)
    assert result1.returncode == 0
    result2 = subprocess.run(cmd, shell=True)
    assert result2.returncode == 0
