import os
import shutil
import subprocess
from pathlib import Path


def test_paths_sh_uses_module(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    # copy scripts to tmp dir
    shutil.copy(repo_root / "paths.sh", tmp_path / "paths.sh")
    shutil.copy(repo_root / "setup_utils.sh", tmp_path / "setup_utils.sh")

    # copy required config template and helper script
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

    # create fake module command
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for cmd in [
        "dirname",
        "mkdir",
        "date",
        "mv",
        "envsubst",
        "python3",
        "cat",
        "chmod",
        "rm",
        "sed",
    ]:
        target = Path("/usr/bin") / cmd
        if target.exists():
            (bin_dir / cmd).symlink_to(target)
    module_log = tmp_path / "module.log"
    fake_matlab = bin_dir / "matlab"
    module_script = bin_dir / "module"
    module_script.write_text(
        f"""#!/bin/sh
echo "$@" >> "{module_log}"
if [ "$1 $2" = 'load MATLAB/2023b' ]; then
  cat > "{fake_matlab}" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x "{fake_matlab}"
fi
"""
    )
    module_script.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)

    result = subprocess.run(
        ["/usr/bin/bash", "-c", "source ./paths.sh && echo $MATLAB_EXEC"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_matlab)
    assert module_log.read_text().strip() == "load MATLAB/2023b"
