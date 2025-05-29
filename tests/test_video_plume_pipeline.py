import numpy as np
import pytest
from pathlib import Path

from Code import plume_pipeline, plume_utils

imageio = pytest.importorskip("imageio.v3")
h5py = pytest.importorskip("h5py")


def _simple_yaml(path: Path) -> dict:
    data: dict[str, dict[str, float]] = {}
    current = None
    with path.open("r") as f:
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


def test_video_plume_pipeline(tmp_path):
    frames = np.stack([
        np.arange(6, dtype=np.uint8).reshape(2, 3),
        np.arange(6, 12, dtype=np.uint8).reshape(2, 3),
    ])
    avi = tmp_path / "in.avi"
    imageio.imwrite(avi, frames, plugin="pyav", fps=1)

    raw_h5 = tmp_path / "raw.h5"
    scaled_h5 = tmp_path / "scaled.h5"
    rotated_h5 = tmp_path / "rotated.h5"

    plume_pipeline.video_to_scaled_rotated_h5(
        str(avi), str(raw_h5), str(scaled_h5), str(rotated_h5)
    )

    with h5py.File(raw_h5, "r") as f:
        raw = f["dataset1"][()]
        attrs = dict(f["dataset1"].attrs)
    assert np.array_equal(raw, frames.reshape(-1))
    assert attrs["height"] == 2
    assert attrs["width"] == 3
    assert attrs["frames"] == 2

    stats = plume_utils.get_intensity_stats()
    with h5py.File(scaled_h5, "r") as f:
        scaled = f["dataset1"][()]
    assert pytest.approx(stats["CRIM"]["min"], rel=1e-6) == float(scaled.min())
    assert pytest.approx(stats["CRIM"]["max"], rel=1e-6) == float(scaled.max())

    with h5py.File(rotated_h5, "r") as f:
        rotated = f["dataset1"][()]
        attrs_rot = dict(f["dataset1"].attrs)

    scaled_frames = plume_utils.rescale_to_crim_range(frames.reshape(-1)).reshape(frames.shape)
    expected_rot = np.stack([np.rot90(f, -1) for f in scaled_frames]).reshape(-1)
    assert np.array_equal(rotated, expected_rot)
    assert attrs_rot["height"] == 3
    assert attrs_rot["width"] == 2
    assert attrs_rot["frames"] == 2

    reg_path = Path("configs") / "plume_registry.yaml"
    registry = _simple_yaml(reg_path)
    assert raw_h5.name in registry
    assert scaled_h5.name in registry
    assert rotated_h5.name in registry

