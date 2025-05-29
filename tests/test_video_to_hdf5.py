import numpy as np
import pytest
from pathlib import Path

imageio = pytest.importorskip("imageio.v3")
h5py = pytest.importorskip("h5py")

from Code import rotate_video


def _simple_yaml(path: Path) -> dict:
    data: dict[str, dict[str, float]] = {}
    current = None
    with open(path, "r") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw.startswith("  "):
                current = stripped.rstrip(":")
                data[current] = {}
            else:
                k, v = stripped.split(":", 1)
                data[current][k.strip()] = float(v)
    return data

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

    registry_path = Path("configs") / "plume_registry.yaml"
    registry = _simple_yaml(registry_path)
    assert output_path.name in registry

