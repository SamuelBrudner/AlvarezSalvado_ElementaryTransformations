import logging
import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v3")

from Code import video_intensity as vi


def test_extract_intensities_logs_memory(tmp_path, caplog):
    frames = np.zeros((3, 2, 2), dtype=np.uint8)
    video_path = tmp_path / "small.mp4"
    imageio.imwrite(video_path, frames, fps=1, macro_block_size=None)

    with caplog.at_level(logging.INFO):
        vi.extract_intensities_from_video(str(video_path))

    assert any("MB" in rec.getMessage() for rec in caplog.records)
