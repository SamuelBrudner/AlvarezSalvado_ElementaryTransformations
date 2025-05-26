"""Utilities for retrieving video intensities using MATLAB.

The helper function in this module writes a temporary MATLAB script to disk and
executes it using ``matlab -batch``.  If ``px_per_mm`` and ``frame_rate`` values
are supplied, they are inserted as variable assignments at the beginning of the
script so that MATLAB code can access them directly.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def get_intensities_from_video_via_matlab(
    script_contents: str,
    matlab_exec_path: str,
    px_per_mm: float | None = None,
    frame_rate: float | None = None,
) -> np.ndarray:
    """Run a MATLAB script and return the extracted intensity vector.

    Parameters
    ----------
    script_contents : str
        Contents of the MATLAB script to execute.
    matlab_exec_path : str
        Path to the MATLAB executable to run.
    px_per_mm : float, optional
        Pixel-to-millimetre conversion factor. When provided, ``px_per_mm`` is
        inserted at the top of the generated MATLAB script so downstream
        functions can access it as a workspace variable.
    frame_rate : float, optional
        Frame rate of the video in Hz. As with ``px_per_mm``, the value is
        embedded in the temporary MATLAB script for use by helper routines.

    Returns
    -------
    numpy.ndarray
        Flattened array of the intensity values extracted from the MAT-file.
    """
    script_file = None
    mat_path = None
    try:
        script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m")
        header_lines = []
        if px_per_mm is not None:
            header_lines.append(f"px_per_mm = {px_per_mm};")
        if frame_rate is not None:
            header_lines.append(f"frame_rate = {frame_rate};")
        full_contents = "\n".join(header_lines + [script_contents])
        script_file.write(full_contents.encode())
        script_file.flush()
        matlab_cmd = [matlab_exec_path, "-batch", f"run('{script_file.name}')"]
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
