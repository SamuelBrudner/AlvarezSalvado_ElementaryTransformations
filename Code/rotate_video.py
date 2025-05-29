"""Video rotation utilities."""
from __future__ import annotations

import logging
from pathlib import Path

try:
    import imageio.v3 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

logger = logging.getLogger(__name__)


def rotate_video_clockwise(input_path: str | Path, output_path: str | Path) -> None:
    """Rotate ``input_path`` 90 degrees clockwise and write to ``output_path``.

    Parameters
    ----------
    input_path, output_path : str or Path
        Paths to the input and output video files.

    Examples
    --------
    >>> rotate_video_clockwise('in.avi', 'out.avi')
    """
    if imageio is None:  # pragma: no cover - runtime environment may lack imageio
        raise ImportError("imageio is required for rotate_video_clockwise")

    frames = list(imageio.imread(input_path, plugin="pyav"))
    if not frames:
        raise ValueError(f"no frames found in {input_path}")

    rotated = [frame[::-1].transpose(1, 0, *range(2, frame.ndim)) for frame in frames]
    imageio.imwrite(output_path, rotated, plugin="pyav")
    logger.info("Saved rotated video to %s", output_path)
