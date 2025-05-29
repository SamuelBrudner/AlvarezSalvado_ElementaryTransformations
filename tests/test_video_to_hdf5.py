import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v3")
h5py = pytest.importorskip("h5py")

from Code import rotate_video


def test_video_to_hdf5(tmp_path):
    frames = np.stack([
        np.arange(6, dtype=np.uint8).reshape(2, 3),
        np.arange(6, 12, dtype=np.uint8).reshape(2, 3),
    ])
    input_path = tmp_path / "in.avi"
    output_path = tmp_path / "out.h5"
    imageio.imwrite(input_path, frames, plugin="pyav", fps=1)

    rotate_video.video_to_hdf5(str(input_path), str(output_path))

    with h5py.File(output_path, "r") as f:
        data = f["dataset1"][()]

    expected = frames.reshape(-1)
    assert np.array_equal(data, expected)

