import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v3")

from Code import video_intensity as vi


def test_extract_intensities_from_video(tmp_path):
    frames = np.array([
        [[0, 255], [128, 64]],
        [[255, 128], [64, 0]],
    ], dtype=np.uint8)
    video_path = tmp_path / "tiny.mp4"
    imageio.imwrite(video_path, frames, fps=1)

    expected = frames.reshape(-1) / 255.0
    result = vi.extract_intensities_from_video(str(video_path))
    assert np.allclose(result, expected)
