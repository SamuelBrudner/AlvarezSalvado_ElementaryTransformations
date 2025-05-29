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
    conda run --prefix ./dev_env python - <<'PY'
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
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import yaml
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

    Notes
    -----
    If the environment variable ``MATLAB_EXEC`` points to an executable
    file, it is used ahead of any auto-detection logic.
    """
    # Check user-provided path first. If supplied, validate that it points to an
    # executable. Previously the path was returned even when invalid, which made
    # error reporting inconsistent.
    if user_path:
        if os.path.isfile(user_path) and os.access(user_path, os.X_OK):
            return user_path
        raise FileNotFoundError(
            f"Provided MATLAB executable not found or not executable: {user_path}"
        )

    # Environment variable override
    env_exec = os.environ.get("MATLAB_EXEC")
    if env_exec:
        if os.path.isfile(env_exec) and os.access(env_exec, os.X_OK):
            return env_exec
        logger.debug("MATLAB_EXEC set but is not executable: %s", env_exec)

    # Look for configs/project_paths.yaml relative to repo root
    project_yaml = (
        Path(__file__).resolve().parents[1] / "configs" / "project_paths.yaml"
    )
    if project_yaml.exists():
        try:
            with project_yaml.open("r") as fh:
                config = yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read project_paths.yaml: %s", exc)
        else:
            exec_path = (
                config.get("matlab", {}).get("executable")
                if isinstance(config, dict)
                else None
            )
            if (
                exec_path
                and os.path.isfile(exec_path)
                and os.access(exec_path, os.X_OK)
            ):
                return exec_path

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
    timeout: Optional[float] = None,
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
    timeout : float, optional
        Maximum time in seconds to allow MATLAB to run. If ``None``, no timeout
        is enforced.
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
        The file must contain a variable named ``all_intensities``. Files saved
        with MATLAB ``-v7.3`` (HDF5) are also supported.

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
        script_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
        header_lines = []
        if px_per_mm is not None:
            header_lines.append(f"px_per_mm = {px_per_mm};")
        if frame_rate is not None:
            header_lines.append(f"frame_rate = {frame_rate};")
        if orig_script_path is not None:
            # Escape single quotes in the path for MATLAB
            safe_path = orig_script_path.replace("'", "''")
            header_lines.extend(
                [
                    f"orig_script_path = '{safe_path}';",
                    "orig_script_dir = fileparts(orig_script_path);",
                ]
            )
        if work_dir is not None:
            safe_wd = work_dir.replace("'", "''")
            header_lines.append(f"cd('{safe_wd}')")
        # Write the header lines and script contents
        header = "\n".join(header_lines) + "\n\n" if header_lines else ""
        script_file.write((header + script_contents).encode())
        script_file.flush()

        # Create a safe path for MATLAB
        safe_path = script_file.name.replace("'", "''")
        matlab_cmd = [
            matlab_path,
            "-nosplash",
            "-nodesktop",
            "-noFigureWindows",
            "-batch",
            f"try, run('{safe_path}'), catch ME, disp(['MATLAB Error: ' getReport(ME, 'extended')]); exit(1); end",
        ]

        logger.info(
            "Running MATLAB script %s in %s",
            script_file.name,
            work_dir or os.getcwd(),
        )
        logger.debug(
            "MATLAB command: %s",
            " ".join(f'"{x}"' if " " in x else x for x in matlab_cmd),
        )

        # Run MATLAB
        try:
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            if work_dir is not None:
                kwargs["cwd"] = work_dir

            proc = subprocess.run(
                matlab_cmd,
                capture_output=True,
                text=True,
                **kwargs,
            )

            # Log MATLAB output for debugging
            if proc.stdout:
                logger.debug("MATLAB stdout:\n%s", proc.stdout)
            if proc.stderr:
                logger.error("MATLAB stderr:\n%s", proc.stderr)

            if proc.returncode != 0:
                if proc.stdout:
                    logger.warning("MATLAB stdout:\n%s", proc.stdout)
                if orig_script_path is not None:
                    orig_script_dir = os.path.dirname(orig_script_path)
                    hint = (
                        f" (orig_script_path: {orig_script_path}, orig_script_dir: {orig_script_dir})"
                    )
                else:
                    hint = ""
                error_msg = proc.stderr.strip() or "No error message from MATLAB"
                raise RuntimeError(
                    f"MATLAB failed with exit code {proc.returncode}{hint}: {error_msg}"
                )

        except subprocess.TimeoutExpired as exc:
            msg = (
                f"MATLAB script execution timed out after {timeout} seconds"
                if timeout is not None
                else "MATLAB script execution timed out"
            )
            raise RuntimeError(msg) from exc

        for line in proc.stdout.splitlines():
            if line.startswith("TEMP_MAT_FILE_SUCCESS:"):
                mat_path = line.split(":", 1)[1].strip()
                break
        if not mat_path or not os.path.exists(mat_path):
            raise RuntimeError("MATLAB did not report output MAT-file")

        try:
            data = loadmat(mat_path)
            arr = np.asarray(data["all_intensities"])  # type: ignore[index]
        except (NotImplementedError, ValueError):
            with h5py.File(mat_path, "r") as f:
                if "all_intensities" not in f:
                    raise KeyError("all_intensities not found in MAT-file")
                arr = np.asarray(f["all_intensities"][()])
        return arr.flatten()
    finally:
        if script_file is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(script_file.name)
        if mat_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(mat_path)
