import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.video_intensity import get_intensities_from_video_via_matlab


def test_get_intensities_from_video_via_matlab(monkeypatch, tmp_path):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = 'disp("hello")'

    mat_file = tmp_path / "out.mat"
    from scipy.io import savemat

    savemat(mat_file, {"all_intensities": np.array([1, 2, 3], dtype=np.float32)})

    stdout = f"some log\nTEMP_MAT_FILE_SUCCESS:{mat_file}\n"

    def fake_run(cmd, capture_output, text):
        assert cmd[0] == matlab_exec
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    created_files = []
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        tmp = orig_ntf(*args, **kwargs)
        created_files.append(tmp.name)
        return tmp

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)

    arr = get_intensities_from_video_via_matlab(
        script_content, matlab_exec, px_per_mm=0.5, frame_rate=60.0
    )
    assert np.array_equal(arr, np.array([1, 2, 3], dtype=np.float32))

    for f in created_files:
        assert not Path(f).exists(), f"temporary file {f} should be removed"
    assert not mat_file.exists(), "MAT-file should be cleaned up"
