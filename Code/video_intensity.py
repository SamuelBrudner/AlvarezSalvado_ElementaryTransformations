"""Utilities for retrieving video intensities using MATLAB.

The helper function in this module writes a temporary MATLAB script to disk and
executes it using ``matlab -batch``.  If ``px_per_mm`` and ``frame_rate`` values
are supplied, they are inserted as variable assignments at the beginning of the
script so that MATLAB code can access them directly.  When
``orig_script_path`` is provided, the variables ``orig_script_path`` and
``orig_script_dir`` are also defined, pointing to the path of the original
script and its directory, respectively.

Examples
--------
Create the development environment and run a short Python snippet inside it::

    ./setup_env.sh --dev
    conda run --prefix ./dev-env python - <<'PY'
    from Code.video_intensity import get_intensities_from_video_via_matlab
    arr = get_intensities_from_video_via_matlab('myscript.m', 'matlab')
    print(arr.shape)
    PY
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def find_matlab_executable(user_path: Optional[str] = None) -> str:
    """Find MATLAB executable in common locations.

    Args:
        user_path: Optional path to MATLAB executable provided by user.

    Returns:
        Path to MATLAB executable.

    Raises:
        FileNotFoundError: If MATLAB executable cannot be found.
    """
    # Check user-provided path first
    if user_path and os.path.isfile(user_path) and os.access(user_path, os.X_OK):
        return user_path

    # Check common MATLAB locations
    common_paths: List[str] = [
        "/usr/local/MATLAB/*/bin/matlab",
        "/opt/MATLAB/*/bin/matlab",
        "/Applications/MATLAB_*.app/bin/matlab",
        os.path.expanduser("~/bin/matlab"),
        "/usr/bin/matlab",
        "/usr/local/bin/matlab",
    ]

    for path_pattern in common_paths:
        for path in glob.glob(path_pattern):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

    # Check if matlab is in PATH
    if matlab_path := shutil.which("matlab"):
        return matlab_path

    raise FileNotFoundError(
        "MATLAB executable not found. Please specify the path to MATLAB using the "
        "MATLAB_EXEC environment variable or the --matlab_exec command line argument. "
        "Common locations were checked: " + ", ".join(common_paths)
    )


def get_intensities_from_video_via_matlab(
    script_contents: str,
    matlab_exec_path: Optional[str] = None,
    px_per_mm: Optional[float] = None,
    frame_rate: Optional[float] = None,
    work_dir: Optional[str] = None,
    orig_script_path: Optional[str] = None,
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
    work_dir : str, optional
        Directory MATLAB should change into before running the temporary script.
    orig_script_path : str, optional
        Original path of the MATLAB script. When provided, the generated
        temporary script defines ``orig_script_path`` and ``orig_script_dir`` so
        that downstream code can reference the original location.


    Notes
    -----
    The temporary script path is embedded in a ``run('...')`` command.
    Any single quotes in the path are escaped for MATLAB by doubling them so
    paths with spaces or quotes are handled correctly.

    Returns
    -------
    numpy.ndarray
        Flattened array of the intensity values extracted from the MAT-file.

    Examples
    --------
    >>> from Code.video_intensity import get_intensities_from_video_via_matlab
    >>> arr = get_intensities_from_video_via_matlab('myscript.m', 'matlab')
    >>> arr.size >= 0
    True
    """
    logger = logging.getLogger(__name__)
    script_file = None
    mat_path = None

    # Find MATLAB executable
    try:
        matlab_path = find_matlab_executable(matlab_exec_path)
        logger.info("Using MATLAB at: %s", matlab_path)
    except FileNotFoundError as e:
        logger.error("Failed to find MATLAB executable: %s", e)
        logger.info(
            "You can specify the MATLAB path using the MATLAB_EXEC environment variable"
        )
        logger.info("or the --matlab_exec command line argument.")
        raise

    try:
        script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m")
        header_lines = []
        # Add all optional parameters that are not None
        header_lines.extend(
            line
            for line in [
                f"cd('{work_dir}')" if work_dir is not None else None,
                f"px_per_mm = {px_per_mm};" if px_per_mm is not None else None,
                f"frame_rate = {frame_rate};" if frame_rate is not None else None,
                (
                    f"orig_script_path = '{orig_script_path}';"
                    if orig_script_path is not None
                    else None
                ),
                (
                    "orig_script_dir = fileparts(orig_script_path);"
                    if orig_script_path is not None
                    else None
                ),
            ]
            if line is not None
        )
        full_contents = "\n".join(header_lines + [script_contents])
        script_file.write(full_contents.encode())
        script_file.flush()
        safe_path = script_file.name.replace("'", "''")
        matlab_cmd = [matlab_exec_path, "-batch", f"run('{safe_path}')"]
        logger.info(
            "Running MATLAB script %s in %s",
            script_file.name,
            work_dir or os.getcwd(),
        )
        proc = subprocess.run(matlab_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            hint = "" if orig_script_path is None else f" (script: {orig_script_path})"
            raise RuntimeError(
                f"MATLAB failed{hint}: {proc.stderr.strip()}\nCheck that orig_script_dir is correct"
            )

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
            with contextlib.suppress(FileNotFoundError):
                os.unlink(script_file.name)
        if mat_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(mat_path)
