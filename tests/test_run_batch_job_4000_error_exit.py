import subprocess
import shutil
from pathlib import Path

BASH = shutil.which("bash") or "/bin/bash"

def test_wrapper_exits_nonzero():
    script = Path(__file__).with_suffix('.sh')
    result = subprocess.run([BASH, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Wrapper exited with non-zero" in result.stdout
