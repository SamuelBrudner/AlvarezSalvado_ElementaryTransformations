"""Video rotation utilities."""
from __future__ import annotations

import logging
from pathlib import Path

from .plume_registry import load_registry, update_plume_registry

try:
    import imageio.v3 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

logger = logging.getLogger(__name__)


def rotate_video_clockwise(
    input_path: str | Path,
    output_path: str | Path,
    *,
    codec: str = "libx264",
    fps: float | None = None,
) -> None:
    """Rotate ``input_path`` 90 degrees clockwise and write to ``output_path``.

    Parameters
    ----------
    input_path, output_path : str or Path
        Paths to the input and output video files.
    codec : str, optional
        Video codec passed to :func:`imageio.imwrite`.  ``libx264`` is used by
        default, which should work for most ``.avi`` files.
    fps : float, optional
        Frames per second for the output video.  If not provided, the frame rate
        is copied from ``input_path`` when available, otherwise ``30`` is used.

    Examples
    --------
    >>> rotate_video_clockwise('in.avi', 'out.avi')
    """
    if imageio is None:  # pragma: no cover - runtime environment may lack imageio
        raise ImportError("imageio is required for rotate_video_clockwise")

    frames = list(imageio.imread(input_path, plugin="pyav"))
    if not frames:
        raise ValueError(f"no frames found in {input_path}")

    registry = load_registry()
    entry = registry.get(str(input_path))
    if entry is None:
        min_val = float(frames[0].min())
        max_val = float(frames[0].max())
        for frame in frames[1:]:
            min_val = min(min_val, float(frame.min()))
            max_val = max(max_val, float(frame.max()))
        update_plume_registry(str(input_path), min_val, max_val)
    else:
        min_val = float(entry.get("min", 0.0))
        max_val = float(entry.get("max", 0.0))

    if fps is None:
        try:
            reader = imageio.get_reader(input_path)
            try:
                fps = reader.get_meta_data().get("fps", 30)
            finally:
                reader.close()
        except Exception:  # noqa: BLE001
            fps = 30

    rotated = [frame[::-1].transpose(1, 0, *range(2, frame.ndim)) for frame in frames]
    imageio.imwrite(output_path, rotated, plugin="pyav", fps=fps, codec=codec)
    logger.info("Saved rotated video to %s", output_path)
    update_plume_registry(str(output_path), min_val, max_val)


def video_to_hdf5(input_path: str | Path, output_path: str | Path) -> None:
    """Read ``input_path`` and store the pixel intensities in an HDF5 file.

    Frames from the input video are concatenated into a single one-dimensional
    array which is saved under the dataset name ``dataset1``.  If frames contain
    multiple colour channels, only the first channel is used.

    Parameters
    ----------
    input_path, output_path : str or Path
        Paths to the input video and output HDF5 file.

    Examples
    --------
    >>> video_to_hdf5('input.avi', 'output.h5')
    """

    if imageio is None:  # pragma: no cover - runtime environment may lack imageio
        raise ImportError("imageio is required for video_to_hdf5")

    registry = load_registry()
    entry = registry.get(str(input_path))
    frames = []
    min_val = None
    max_val = None
    for frame in imageio.imiter(input_path):
        arr = frame if frame.ndim == 2 else frame[..., 0]
        frames.append(arr.reshape(-1))
        if entry is None:
            vmin = float(arr.min())
            vmax = float(arr.max())
            min_val = vmin if min_val is None else min(min_val, vmin)
            max_val = vmax if max_val is None else max(max_val, vmax)

    if not frames:
        raise ValueError(f"no frames found in {input_path}")

    if entry is not None:
        min_val = float(entry.get("min", 0.0))
        max_val = float(entry.get("max", 0.0))
    else:
        update_plume_registry(str(input_path), float(min_val), float(max_val))

    import h5py  # imported lazily to avoid mandatory dependency
    import numpy as np

    data = np.concatenate(frames)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("dataset1", data=data)
    update_plume_registry(str(output_path), float(min_val), float(max_val))
