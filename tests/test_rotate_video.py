import numpy as np
import pytest

from Code import rotate_video

def test_rotate_video_clockwise(tmp_path):
    # Create a dummy 2x3 video with two frames using numpy
    frames = [
        np.arange(6, dtype=np.uint8).reshape(2, 3),
        np.arange(6, 12, dtype=np.uint8).reshape(2, 3),
    ]
    input_video = tmp_path / "in.avi"
    output_video = tmp_path / "out.avi"

    imageio = pytest.importorskip("imageio.v3")
    imageio.imwrite(input_video, frames, plugin="pyav")

    rotate_video.rotate_video_clockwise(str(input_video), str(output_video))

    result = imageio.imread(output_video, plugin="pyav")
    assert result.shape == (2, 3, 2)
    # Rotated frames should have shape 3x2
    rotated_frames = list(np.moveaxis(result, -1, 0))
    assert rotated_frames[0].shape == (3, 2)
    assert rotated_frames[1].shape == (3, 2)


def test_rotate_video_clockwise_custom_fps(tmp_path):
    frames = [
        np.arange(6, dtype=np.uint8).reshape(2, 3),
        np.arange(6, 12, dtype=np.uint8).reshape(2, 3),
    ]
    input_video = tmp_path / "in.avi"
    output_video = tmp_path / "out.avi"

    imageio = pytest.importorskip("imageio.v3")
    imageio.imwrite(input_video, frames, plugin="pyav", fps=15)

    rotate_video.rotate_video_clockwise(
        str(input_video),
        str(output_video),
        fps=25,
    )

    meta = imageio.get_reader(output_video).get_meta_data()
    assert meta.get("fps") == 25
