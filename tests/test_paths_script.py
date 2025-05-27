import subprocess
import shutil
import pytest


@pytest.mark.usefixtures("tmp_path")
def test_paths_script(tmp_path):
    if shutil.which("conda") is None:
        pytest.skip("conda not available")

    env_result = subprocess.run([
        "bash",
        "./setup_env.sh",
        "--dev",
    ], capture_output=True, text=True)
    assert env_result.returncode == 0, env_result.stderr

    result = subprocess.run([
        "bash",
        "./paths.sh",
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "SyntaxError" not in result.stderr
