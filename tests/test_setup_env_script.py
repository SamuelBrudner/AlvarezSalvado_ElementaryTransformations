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
    cmd = 'bash ./setup_env.sh --dev'
    result1 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result1.returncode == 0
    result2 = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    assert result2.returncode == 0

def test_setup_env_has_conda_lock_pip_fallback():
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'python -m pip install --user conda-lock' in content
    assert 'conda-lock --version' in content


def test_setup_env_checks_existing_conda_lock():
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'command -v conda-lock >/dev/null 2>&1 || ! conda-lock --version >/dev/null 2>&1' in content


def test_setup_env_handles_old_conda_versions():
    """Script should support old Conda without --force."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'conda_supports_force' in content
    assert 'conda env remove --prefix "./${LOCAL_ENV_DIR}" -y' in content


def test_setup_env_attempts_module_load():
    """Script should try loading Conda via environment modules."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'try_load_conda_module' in content or 'module load' in content
