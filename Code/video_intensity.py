"""Utilities for retrieving video intensities using MATLAB."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def get_intensities_from_video_via_matlab(script_contents: str, matlab_exec_path: str) -> np.ndarray:
    """Run a MATLAB script and return the extracted intensity vector."""
    script_file = None
    mat_path = None
    try:
        script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m")
        script_file.write(script_contents.encode())
        script_file.flush()
        matlab_cmd = [matlab_exec_path, "-batch", Path(script_file.name).stem]
        proc = subprocess.run(matlab_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MATLAB failed: {proc.stdout}\n{proc.stderr}")

        for line in proc.stdout.splitlines():
            if line.startswith("TEMP_MAT_FILE_SUCCESS:"):
                mat_path = line.split(":", 1)[1].strip()
                break
        if not mat_path or not os.path.exists(mat_path):
            raise RuntimeError("MATLAB did not report output MAT-file")

        data = loadmat(mat_path)
        if "all_intensities" not in data:
            raise KeyError("all_intensities not found in MAT-file")
        return np.asarray(data["all_intensities"]).flatten()
    finally:
        if script_file is not None:
            try:
                os.unlink(script_file.name)
            except FileNotFoundError:
                pass
        if mat_path is not None:
            try:
                os.unlink(mat_path)
            except FileNotFoundError:
                pass
