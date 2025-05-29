import subprocess
import shutil

BASH = shutil.which("bash") or "/bin/bash"
import pytest

BASH = shutil.which("bash") or "/bin/bash"


@pytest.mark.slow
@pytest.mark.usefixtures("tmp_path")
def test_paths_script(tmp_path):
    if shutil.which("conda") is None:
        pytest.skip("conda not available")

    env_result = subprocess.run(
        [
            BASH,
            "./setup_env.sh",
            "--skip-conda-lock",
            "--no-tests",
        ],
        capture_output=True,
        text=True,
    )
    assert env_result.returncode == 0, env_result.stderr

    result = subprocess.run([
        BASH,
        "./paths.sh",
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "SyntaxError" not in result.stderr
