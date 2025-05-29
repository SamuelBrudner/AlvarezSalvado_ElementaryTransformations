import os
import shutil
import subprocess
from pathlib import Path

BASH = shutil.which("bash") or "/bin/bash"


def test_paths_sh_noninteractive_quiet(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copy(repo_root / "paths.sh", tmp_path / "paths.sh")
    shutil.copy(repo_root / "setup_utils.sh", tmp_path / "setup_utils.sh")

    (tmp_path / "configs").mkdir()
    shutil.copy(
        repo_root / "configs" / "project_paths.yaml.template",
        tmp_path / "configs" / "project_paths.yaml.template",
    )
    (tmp_path / "scripts").mkdir()
    shutil.copy(
        repo_root / "scripts" / "make_paths_relative.py",
        tmp_path / "scripts" / "make_paths_relative.py",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_matlab = bin_dir / "matlab"
    fake_matlab.write_text("#!/bin/sh\nexit 0\n")
    fake_matlab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PS4"] = "+"

    log_file = tmp_path / "log"
    result = subprocess.run(
        [BASH, "-c", f"source ./paths.sh > {log_file} 2>&1 && echo $MATLAB_EXEC"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_matlab)
    log_content = log_file.read_text()
    assert "[INFO]" not in log_content
    assert "[WARNING]" not in log_content
    assert "[ERROR]" not in log_content
    assert "[SUCCESS]" not in log_content
