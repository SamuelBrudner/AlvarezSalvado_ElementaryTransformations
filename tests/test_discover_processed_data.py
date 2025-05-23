import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.data_discovery import discover_processed_data


def test_discover_processed_data(tmp_path):
    base = tmp_path / "processed"
    run_dir = base / "gaussian_bilateral" / "agent_1" / "seed_0"
    run_dir.mkdir(parents=True)
    # create a dummy config_used.yaml in JSON format
    (run_dir / "config_used.yaml").write_text('{"frame_rate": 100, "px_per_mm": 5}')
    (run_dir / "results.mat").write_text("dummy")

    cfg_dict = {
        "data_paths": {"processed_base_dirs": [str(base)]},
        "metadata_extraction": {"directory_template": "{plume}_{mode}/agent_{agent_id}/seed_{seed}"},
        "load_run_config": True,
    }
    cfg_path = tmp_path / "analysis_config.yaml"
    cfg_path.write_text(json.dumps(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    records = list(discover_processed_data(cfg))
    assert len(records) == 1
    rec = records[0]
    assert rec["metadata"]["plume"] == "gaussian"
    assert rec["metadata"]["mode"] == "bilateral"
    assert rec["metadata"]["agent_id"] == "1"
    assert rec["metadata"]["seed"] == "0"
    assert rec["config"]["frame_rate"] == 100

