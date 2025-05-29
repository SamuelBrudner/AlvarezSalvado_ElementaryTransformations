import os
import stat
from pathlib import Path

import importlib
import sys
import types

import pytest



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


def test_find_matlab_executable_from_env(tmp_path, monkeypatch):
    fake_matlab = tmp_path / "fake_matlab_env"
    fake_matlab.write_text("#!/bin/sh\necho MATLAB")
    fake_matlab.chmod(fake_matlab.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("MATLAB_EXEC", str(fake_matlab))

    video_intensity = importlib.reload(importlib.import_module("Code.video_intensity"))

    assert video_intensity.find_matlab_executable() == str(fake_matlab)


def test_find_matlab_executable_missing(tmp_path, monkeypatch):
    """find_matlab_executable raises when no executable can be located."""
    repo_root = Path(__file__).resolve().parents[1]
    config_file = repo_root / "configs" / "project_paths.yaml"

    # Ensure no leftover configuration or environment variables interfere
    if config_file.exists():
        config_file.unlink()
    monkeypatch.delenv("MATLAB_EXEC", raising=False)

    # Provide stub modules so Code.video_intensity imports without real deps
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(asarray=lambda x: x),
    )
    fake_scipy_io = types.SimpleNamespace(loadmat=lambda p: {"all_intensities": []})
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(io=fake_scipy_io))
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)
    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=lambda *a, **k: None))
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        types.SimpleNamespace(safe_load=lambda *a, **k: {}, safe_dump=lambda *a, **k: None),
    )

    video_intensity = importlib.reload(importlib.import_module("Code.video_intensity"))

    non_existent = tmp_path / "missing_matlab"
    with pytest.raises(FileNotFoundError):
        video_intensity.find_matlab_executable(str(non_existent))

