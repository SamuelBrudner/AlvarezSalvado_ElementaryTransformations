import json
import os
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.data_discovery import discover_processed_data, check_parameter_consistency


def test_use_config_used_for_dt(tmp_path):
    base = tmp_path / "processed"
    run_dir = base / "gaussian_bilateral" / "agent_1" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config_used.yaml").write_text("frame_rate: 20\n")
    (run_dir / "params.json").write_text(json.dumps({"beta": 0.01}))

    cfg_dict = {
        "data_paths": {"processed_base_dirs": [str(base)]},
        "metadata_extraction": {"directory_template": "{plume}_{mode}/agent_{agent_id}/seed_{seed}"},
        "parameter_usage": {
            "use_config_used_for_dt": True,
            "framerate_field_in_config_used": "frame_rate"
        }
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    records = list(discover_processed_data(cfg))
    assert len(records) == 1
    assert pytest.approx(records[0]["dt"], 1e-6) == 1/20


def test_check_model_parameter_consistency(tmp_path):
    base = tmp_path / "processed"
    run1 = base / "gaussian_bilateral" / "agent_1" / "seed_0"
    run2 = base / "gaussian_bilateral" / "agent_2" / "seed_1"
    for p in [run1, run2]:
        p.mkdir(parents=True)
    (run1 / "params.json").write_text(json.dumps({"beta": 0.01}))
    (run2 / "params.json").write_text(json.dumps({"beta": 0.01}))

    cfg_dict = {
        "data_paths": {"processed_base_dirs": [str(base)]},
        "metadata_extraction": {"directory_template": "{plume}_{mode}/agent_{agent_id}/seed_{seed}"},
        "parameter_usage": {
            "check_model_parameter_consistency": {
                "enabled": True,
                "parameters_to_check": ["beta"]
            }
        }
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    records = list(discover_processed_data(cfg))
    check_parameter_consistency(records, cfg)

    # Introduce mismatch
    (run2 / "params.json").write_text(json.dumps({"beta": 0.02}))
    records = list(discover_processed_data(cfg))
    with pytest.raises(ValueError):
        check_parameter_consistency(records, cfg)
