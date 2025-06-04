import os
import sys
import json
from pathlib import Path


def test_write_plume_config_creates_file(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from setup_plume_config import write_plume_config

    config_dir = tmp_path / "configs"
    path = write_plume_config("plume.hdf5", base_dir=config_dir)
    expected = config_dir / "navigation_model" / "navigation_model_default.json"

    assert Path(path) == expected
    with open(path) as fh:
        data = json.load(fh)
    assert data["plume_file"] == "plume.hdf5"
