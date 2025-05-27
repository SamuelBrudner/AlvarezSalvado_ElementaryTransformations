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


def test_setup_env_exports_user_bin_for_conda_lock():
    """Script should add the pip user bin directory to PATH if needed."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'site --user-base' in content
    assert 'append_path_if_missing "${USER_BIN}"' in content
    assert 'hash -r' in content


def test_setup_env_invokes_append_path_helper_after_pip():
    """append_path_if_missing should run after pip fallback and before hash."""
    with open('setup_env.sh') as f:
        content = f.read()
    pip_idx = content.index('python -m pip install --user conda-lock')
    append_idx = content.index('append_path_if_missing "${USER_BIN}"')
    hash_idx = content.index('hash -r')
    assert pip_idx < append_idx < hash_idx


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


def test_skip_conda_lock_flag_skips_commands(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    conda_base = tmp_path / "conda"
    (conda_base / "etc/profile.d").mkdir(parents=True)
    (conda_base / "etc/profile.d/conda.sh").write_text("")

    conda_script = bin_dir / "conda"
    conda_script.write_text(
        f"""#!/bin/bash
if [ "$1" = "info" ] && [ "$2" = "--base" ]; then
  echo "{conda_base}"
elif [ "$1" = "info" ] && [ "$2" = "--json" ]; then
  echo '{{"platform":"linux-64"}}'
else
  exit 0
fi
"""
    )
    conda_script.chmod(0o755)

    conda_lock_script = bin_dir / "conda-lock"
    conda_lock_script.write_text(
        """#!/bin/bash
if [ "$1" = "--version" ]; then
  echo "conda-lock 1.0.0"
else
  echo "conda-lock $@" >&2
fi
"""
    )
    conda_lock_script.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", "./setup_env.sh", "--skip-conda-lock", "--no-tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "Unknown option" not in output
    assert "conda-lock lock" not in output

