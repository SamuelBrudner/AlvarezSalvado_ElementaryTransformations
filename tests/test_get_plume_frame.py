import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v2")
h5py = pytest.importorskip("h5py")

from Code import plume_utils


def test_get_plume_frame(tmp_path):
    frames = np.array([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
    ], dtype=np.uint8)
    video_path = tmp_path / "tiny.mp4"
    imageio.imwrite(video_path, frames, fps=1, macro_block_size=None)

    h5_path = tmp_path / "tiny.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("intensity", data=frames)

    cfg = tmp_path / "project_paths.yaml"
    cfg.write_text(
        f"data:\n  crimaldi: {h5_path}\n  video: {video_path}\n",
        encoding="utf-8",
    )

    h5_frame = plume_utils.get_plume_frame("crimaldi", 1, config_path=cfg)
    vid_frame = plume_utils.get_plume_frame("video", 1, config_path=cfg)

    np.testing.assert_array_equal(h5_frame, frames[1])
    np.testing.assert_array_equal(vid_frame, frames[1])
