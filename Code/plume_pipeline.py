"""Helpers to convert and transform plume videos.

This module provides a convenience function
:func:`video_to_scaled_rotated_h5` that converts an AVI file to HDF5,
scales the intensities to match the Crimaldi range, and rotates the
frames 90 degrees clockwise. All intermediate results are stored in
HDF5 files and registered via :func:`Code.plume_registry.update_plume_registry`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .plume_registry import update_plume_registry
from .plume_utils import rescale_to_crim_range
from .rotate_video import video_to_hdf5

__all__ = ["video_to_scaled_rotated_h5", "scale_hdf5_to_crim_range", "rotate_hdf5_clockwise"]


def _read_shape(dset: h5py.Dataset) -> tuple[int, int, int]:
    """Return (height, width, frames) from ``dset`` attributes."""
    height = int(dset.attrs["height"])
    width = int(dset.attrs["width"])
    frames = int(dset.attrs["frames"])
    return height, width, frames


def scale_hdf5_to_crim_range(input_path: str | Path, output_path: str | Path) -> None:
    """Scale ``input_path`` to the Crimaldi intensity range and write ``output_path``."""
    with h5py.File(input_path, "r") as f:
        dset = f["dataset1"]
        data = dset[()]
        height, width, frames = _read_shape(dset)

    scaled = rescale_to_crim_range(np.asarray(data, dtype=float))
    with h5py.File(output_path, "w") as f:
        out = f.create_dataset("dataset1", data=scaled)
        out.attrs.update(height=height, width=width, frames=frames)

    update_plume_registry(str(output_path), float(scaled.min()), float(scaled.max()))


def rotate_hdf5_clockwise(input_path: str | Path, output_path: str | Path) -> None:
    """Rotate plume data in ``input_path`` 90° clockwise and save to ``output_path``."""
    with h5py.File(input_path, "r") as f:
        dset = f["dataset1"]
        data = dset[()]
        height, width, frames = _read_shape(dset)

    frames_arr = np.reshape(data, (frames, height, width))
    rotated = np.array([np.rot90(frame, k=-1) for frame in frames_arr])
    with h5py.File(output_path, "w") as f:
        out = f.create_dataset("dataset1", data=rotated.reshape(-1))
        out.attrs.update(height=width, width=height, frames=frames)

    update_plume_registry(str(output_path), float(rotated.min()), float(rotated.max()))


def video_to_scaled_rotated_h5(
    avi_path: str | Path,
    raw_h5: str | Path,
    scaled_h5: str | Path,
    rotated_h5: str | Path,
) -> None:
    """Full pipeline: AVI → HDF5 → scaled → rotated."""
    video_to_hdf5(avi_path, raw_h5)

    with h5py.File(raw_h5, "r+") as f:
        dset = f["dataset1"]
        # determine shape if missing
        if not {"height", "width", "frames"} <= set(dset.attrs):
            raise ValueError("missing shape information in raw HDF5 file")
        height, width, frames = _read_shape(dset)
        data = dset[()]
    scaled = rescale_to_crim_range(np.asarray(data, dtype=float))
    with h5py.File(scaled_h5, "w") as f:
        out = f.create_dataset("dataset1", data=scaled)
        out.attrs.update(height=height, width=width, frames=frames)
    update_plume_registry(str(scaled_h5), float(scaled.min()), float(scaled.max()))

    frames_arr = np.reshape(scaled, (frames, height, width))
    rotated = np.array([np.rot90(frame, k=-1) for frame in frames_arr])
    with h5py.File(rotated_h5, "w") as f:
        out = f.create_dataset("dataset1", data=rotated.reshape(-1))
        out.attrs.update(height=width, width=height, frames=frames)
    update_plume_registry(str(rotated_h5), float(rotated.min()), float(rotated.max()))
