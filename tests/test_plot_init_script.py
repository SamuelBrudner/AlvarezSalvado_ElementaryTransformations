import json
import subprocess
from pathlib import Path
import numpy as np
import h5py


def create_config(path, x_range, y_range, success_radius, h5_path):
    cfg = {
        "plume_id": "test",
        "data_path": {"path": str(h5_path), "dataset_name": "/dataset2"},
        "spatial": {
            "resolution": {"width": 4, "height": 4},
            "mm_per_pixel": 0.1,
            "arena_bounds": {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 0}
        },
        "temporal": {"frame_rate": 10, "total_frames": 2},
        "model_params": {"tscale": 0.2, "pxscale": 0.1},
        "simulation": {
            "success_radius_cm": success_radius,
            "duration_seconds": 1,
            "agent_initialization": {
                "x_range_cm": x_range,
                "y_range_cm": y_range,
                "n_agents_per_job": 1
            },
            "source_position": {"x_cm": 0, "y_cm": 0}
        }
    }
    path.write_text(json.dumps(cfg))


def test_plot_init_reads_config(tmp_path):
    h5_file = tmp_path / "plume.h5"
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("dataset2", data=np.zeros((4, 4, 2)))

    x_range = [-9, 9]
    y_range = [-22, -18]
    radius = 3.3

    crim_cfg = tmp_path / "crim.json"
    smoke_cfg = tmp_path / "smoke.json"
    create_config(crim_cfg, x_range, y_range, radius, h5_file)
    create_config(smoke_cfg, x_range, y_range, radius, h5_file)

    root = Path(__file__).resolve().parents[1]
    cmd = f"plot_init_with_plumes('{crim_cfg}', '{smoke_cfg}');"
    result = subprocess.run([
        'bash', str(root / 'run_matlab_safe.sh')
    ], input=cmd, text=True, capture_output=True, cwd=root)

    assert str(x_range[0]) in result.stdout
    assert str(y_range[0]) in result.stdout
    assert str(radius) in result.stdout
