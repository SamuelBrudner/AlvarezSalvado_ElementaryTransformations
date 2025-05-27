import shutil
import subprocess
from pathlib import Path


def test_paths_sh_runs_without_syntaxerror(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]

    # copy required files
    shutil.copy(repo_root / "paths.sh", tmp_path / "paths.sh")
    shutil.copy(repo_root / "setup_utils.sh", tmp_path / "setup_utils.sh")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    shutil.copy(
        repo_root / "configs" / "project_paths.yaml.template",
        configs_dir / "project_paths.yaml.template",
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(
        repo_root / "scripts" / "make_paths_relative.py",
        scripts_dir / "make_paths_relative.py",
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
