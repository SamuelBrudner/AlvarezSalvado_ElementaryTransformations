import os
import shutil
import subprocess
from pathlib import Path

BASH = shutil.which("bash") or "/bin/bash"
SYSTEM_PYTHON = shutil.which("python3") or "/usr/bin/python3"


def test_project_root_absolute(tmp_path):
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
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        f"""#!/bin/sh
if echo \"$1\" | grep -q 'make_paths_relative.py'; then
    {SYSTEM_PYTHON} - \"$2\" \"$3\" <<'PY'
import sys
config, root = sys.argv[1:3]
text = open(config).read().replace(root + '/', '').replace(root, '.')
open(config, 'w').write(text)
PY
else
    exec {SYSTEM_PYTHON} "$@"
fi
"""
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [BASH, str(tmp_path / "paths.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr

    config_path = tmp_path / "configs" / "project_paths.yaml"
    content = config_path.read_text().splitlines()
    project_root_line = next((l for l in content if l.startswith("project_root:")), None)
    assert project_root_line is not None
    value = project_root_line.split(":", 1)[1].strip().strip('"')
    assert os.path.isabs(value)
    assert value == str(tmp_path)
