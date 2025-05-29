"""Video rotation utilities."""
from __future__ import annotations

import logging
from pathlib import Path

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
