import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
imageio = pytest.importorskip("imageio.v3")
h5py = pytest.importorskip("h5py")

from scripts import process_custom_plume


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
                if current is not None:
                    data[current][k.strip()] = float(v)
    return data


def test_process_custom_plume(tmp_path):
    frames = np.stack(
        [
            np.arange(6, dtype=np.uint8).reshape(2, 3),
            np.arange(6, 12, dtype=np.uint8).reshape(2, 3),
        ]
    )
    avi = tmp_path / "input.avi"
    imageio.imwrite(avi, frames, plugin="pyav", fps=1)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    process_custom_plume.main([str(avi), str(out_dir), "2", "1"])

    raw_h5 = out_dir / "input_raw.h5"
    scaled_h5 = out_dir / "input_scaled.h5"
    rotated_h5 = out_dir / "input_rotated.h5"
    meta = out_dir / "input_meta.yaml"

    assert raw_h5.is_file()
    assert scaled_h5.is_file()
    assert rotated_h5.is_file()
    assert meta.is_file()

    meta_data = _simple_yaml(meta)
    assert meta_data["output_directory"] == str(out_dir)
    assert meta_data["output_filename"] == rotated_h5.name
    assert meta_data["vid_mm_per_px"] == 0.5
    assert meta_data["fps"] == 1
    assert meta_data["scaled_to_crim"]

    reg_path = Path("configs") / "plume_registry.yaml"
    registry = _simple_yaml(reg_path)
    assert raw_h5.name in registry
    assert scaled_h5.name in registry
    assert rotated_h5.name in registry
