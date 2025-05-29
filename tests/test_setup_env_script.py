import os
import subprocess
import shutil
import pytest
from pathlib import Path


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
    assert '--debug' not in content
    assert 'DEBUG=1' in content


def test_setup_env_uses_dev_env_directory():
    """Ensure the script references the dev_env directory."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'dev_env' in content


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


def test_conda_lock_not_installed_in_base():
    """Ensure the script does not attempt to modify the base environment."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'conda install -y -n base -c conda-forge conda-lock' not in content


def test_conda_lock_installed_in_prefix():
    """Script should prefer installing conda-lock into the local prefix."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'conda run --prefix "./${LOCAL_ENV_DIR}" conda install -y -c conda-forge conda-lock' in content


def test_setup_env_exports_user_bin_for_conda_lock():
    """Script should add the pip user bin directory to PATH if needed."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'site --user-base' in content
    assert 'export PATH="${USER_BIN}:${PATH}"' in content
    assert 'hash -r' in content


def test_setup_env_exports_user_bin_after_pip_install():
    """USER_BIN export should occur after pip fallback and before hash."""
    with open('setup_env.sh') as f:
        content = f.read()
    pip_idx = content.index('python -m pip install --user conda-lock')
    export_idx = content.index('export PATH="${USER_BIN}:${PATH}"')
    hash_idx = content.index('hash -r')
    assert pip_idx < export_idx < hash_idx


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


def test_setup_env_checks_numpy_presence():
    """Script should verify numpy import after environment creation."""
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'conda run --prefix "./${LOCAL_ENV_DIR}" python -c "import numpy"' in content

    
def test_pre_commit_fallback_to_pip(tmp_path, monkeypatch):
    """If conda install fails, the script should attempt pip install."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    conda_base = tmp_path / "conda"
    (conda_base / "etc/profile.d").mkdir(parents=True)
    (conda_base / "etc/profile.d/conda.sh").write_text("")

    log_file = tmp_path / "conda_log"

    conda_script = bin_dir / "conda"
    conda_script.write_text(
        f"""#!/bin/bash
echo "$@" >> "{log_file}"
if [ "$1" = "info" ] && [ "$2" = "--base" ]; then
  echo "{conda_base}"
elif [ "$1" = "info" ] && [ "$2" = "--json" ]; then
  echo '{{"platform":"linux-64"}}'
elif [ "$1" = "env" ] && [ "$2" = "create" ] && [ "$3" = "--help" ]; then
  echo "--force"
  exit 0
elif [ "$1" = "env" ]; then
  exit 0
elif [ "$1" = "run" ]; then
  shift
  if [ "$1" = "--prefix" ]; then
    shift 2
  fi
  if [ "$1" = "pre-commit" ] && [ "$2" = "--version" ]; then
    exit 1
  elif [ "$1" = "conda" ] && [ "$2" = "install" ]; then
    exit 1
  fi
  exit 0
else
  exit 0
fi
"""
    )
    conda_script.chmod(0o755)

    pip_script = bin_dir / "pip"
    pip_script.write_text(
        f"""#!/bin/bash
echo "$@" >> "{log_file}"
exit 0
"""
    )
    pip_script.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", "./setup_env.sh", "--dev", "--skip-conda-lock", "--no-tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    executed = log_file.read_text()
    assert "pip install pre-commit" in executed


def test_check_not_in_active_env_function_present():
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'check_not_in_active_env()' in content
    assert "dev_env is currently active" in content


def test_check_not_in_active_env_called_before_creation():
    with open('setup_env.sh') as f:
        content = f.read()
    def_idx = content.index('check_not_in_active_env()')
    call_idx = content.index('check_not_in_active_env', def_idx + 1)
    env_idx = content.index('section "Setting up Conda environment"')
    assert call_idx < env_idx


def test_setup_aborts_if_env_active(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()

    conda_base = tmp_path / 'conda'
    (conda_base / 'etc/profile.d').mkdir(parents=True)
    (conda_base / 'etc/profile.d/conda.sh').write_text('')

    conda_script = bin_dir / 'conda'

def test_setup_env_uses_user_bin_conda_lock_when_not_in_path(tmp_path, monkeypatch):
    """Ensure setup succeeds when conda-lock exists only in the user bin."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    conda_base = tmp_path / "conda"
    (conda_base / "etc/profile.d").mkdir(parents=True)
    (conda_base / "etc/profile.d/conda.sh").write_text("")

    conda_script = bin_dir / "conda"

    conda_script.write_text(
        f"""#!/bin/bash
if [ \"$1\" = \"info\" ] && [ \"$2\" = \"--base\" ]; then
  echo \"{conda_base}\"
elif [ \"$1\" = \"info\" ] && [ \"$2\" = \"--json\" ]; then
  echo '{{"platform":"linux-64"}}'
elif [ \"$1\" = \"env\" ] && [ \"$2\" = \"create\" ] && [ \"$3\" = \"--help\" ]; then
  echo "--force"
  exit 0
elif [ \"$1\" = \"env\" ]; then
  exit 0
elif [ \"$1\" = \"env\" ]; then
  exit 0
elif [ \"$1\" = \"run\" ]; then
  exit 0
else
  exit 0
fi
"""
    )
    conda_script.chmod(0o755)

    monkeypatch.setenv('PATH', f"{bin_dir}:{os.environ['PATH']}")
    dev_env = Path("dev_env")
    dev_env.mkdir(exist_ok=True)
    monkeypatch.setenv("CONDA_PREFIX", str(dev_env.resolve()))

    result = subprocess.run(
        ['bash', './setup_env.sh', '--skip-conda-lock', '--no-tests'],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'dev_env is currently active' in result.stdout + result.stderr
    user_base = tmp_path / "user"
    user_bin = user_base / "bin"
    user_bin.mkdir(parents=True)

    conda_lock_script = user_bin / "conda-lock"
    conda_lock_script.write_text("#!/bin/bash\necho 'conda-lock 1.0.0'")
    conda_lock_script.chmod(0o755)

    python_script = bin_dir / "python"
    python_script.write_text(
        f"""#!/bin/bash
if [ \"$1\" = "-m" ] && [ \"$2\" = "site" ] && [ \"$3\" = "--user-base" ]; then
  echo '{user_base}'
elif [ \"$1\" = "-m" ] && [ \"$2\" = "pip" ] && [ \"$3\" = "install" ]; then
  exit 0
else
  /usr/bin/env python "$@"
fi
"""
    )
    python_script.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONUSERBASE", str(user_base))

    result = subprocess.run(
        ["bash", "./setup_env.sh", "--no-tests"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

def test_setup_env_exits_when_active(monkeypatch):
    """Script should fail if run inside an active environment."""
    monkeypatch.setenv("CONDA_PREFIX", os.path.abspath("dev_env"))
    result = subprocess.run([
        "bash",
        "./setup_env.sh",
        "--dev",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "deactivate" in result.stdout + result.stderr


def test_setup_env_invokes_nfs_cleanup():
    """Ensure cleanup function is defined and used after environment removal."""
    with open("setup_env.sh") as f:
        content = f.read()
    assert "cleanup_nfs_temp_files()" in content
    remove_indices = [i for i in range(len(content.splitlines())) if "conda env remove" in content.splitlines()[i]]
    cleanup_indices = [i for i in range(len(content.splitlines())) if "cleanup_nfs_temp_files" in content.splitlines()[i]]
    for idx in remove_indices:
        assert any(c > idx for c in cleanup_indices)
