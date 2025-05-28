import shutil
import subprocess
from pathlib import Path


def test_paths_sh_runs_without_syntaxerror(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    # copy required files
    shutil.copy(repo_root / "paths.sh", tmp_path / "paths.sh")
    shutil.copy(repo_root / "setup_utils.sh", tmp_path / "setup_utils.sh")

    configs_dir = _copy_required_files(
        tmp_path, "configs", repo_root, "project_paths.yaml.template"
    )
    _ = _copy_required_files(
        tmp_path, "scripts", repo_root, "make_paths_relative.py"
    )
    result = subprocess.run(
        ["bash", str(tmp_path / "paths.sh")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "SyntaxError" not in result.stderr
    assert (configs_dir / "project_paths.yaml").exists()


# Helper for copying files required by paths.sh
def _copy_required_files(tmp_path, arg1, repo_root, arg3):
    result = tmp_path / arg1
    result.mkdir()
    shutil.copy(repo_root / arg1 / arg3, result / arg3)

    return result
