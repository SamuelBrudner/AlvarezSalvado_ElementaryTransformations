import os
import stat
from pathlib import Path

import importlib



def test_find_matlab_executable_from_project_paths(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"
    config_file = config_dir / "project_paths.yaml"

    fake_matlab = tmp_path / "fake_matlab"
    fake_matlab.write_text("#!/bin/sh\necho MATLAB")
    fake_matlab.chmod(fake_matlab.stat().st_mode | stat.S_IXUSR)

    config_file.write_text(
        f"matlab:\n  executable: '{fake_matlab}'\n"
    )

    video_intensity = importlib.reload(importlib.import_module("Code.video_intensity"))

    try:
        path = video_intensity.find_matlab_executable()
        assert path == str(fake_matlab)
    finally:
        config_file.unlink()

