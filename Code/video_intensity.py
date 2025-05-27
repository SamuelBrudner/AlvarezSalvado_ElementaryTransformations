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

import logging
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def get_intensities_from_video_via_matlab(
    script_contents: str,
    matlab_exec_path: str,
    px_per_mm: float | None = None,
    frame_rate: float | None = None,
    work_dir: str | None = None,
    orig_script_path: str | None = None,
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
    try:
        script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m")
        header_lines = []
        if work_dir is not None:
            header_lines.append(f"cd('{work_dir}')")
        if px_per_mm is not None:
            header_lines.append(f"px_per_mm = {px_per_mm};")
        if frame_rate is not None:
            header_lines.append(f"frame_rate = {frame_rate};")
        if orig_script_path is not None:
            header_lines.append(f"orig_script_path = '{orig_script_path}';")

            header_lines.append("orig_script_dir = fileparts(orig_script_path);")
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
            try:
                os.unlink(script_file.name)
            except FileNotFoundError:
                pass
        if mat_path is not None:
            try:
                os.unlink(mat_path)
            except FileNotFoundError:
                pass
