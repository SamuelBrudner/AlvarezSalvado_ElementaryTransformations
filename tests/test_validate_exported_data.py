import os
import sys
import json
import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.validate_exported_data import validate_exported_data


def make_run_dir(tmp_path, traj_content, summary_content, params_content="{}"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trajectories.csv").write_text(traj_content)
    (run_dir / "summary.json").write_text(summary_content)
    (run_dir / "params.json").write_text(params_content)
    return run_dir


def test_validate_missing_column(tmp_path):
    cfg_dict = {"trajectory_processing": {"required_columns": ["t", "x", "y", "theta", "turn"]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    traj_content = "t,x,y,turn\n0,0,0,0"
    summary_content = json.dumps({"successrate": 1.0, "latency": [1], "n_trials": 1, "timesteps": 1})
    run_dir = make_run_dir(tmp_path, traj_content, summary_content)

    try:
        validate_exported_data(run_dir, cfg)
    except Exception as e:
        assert "theta" in str(e)
    else:
        raise AssertionError("Validator did not detect missing column")


def test_validate_invalid_summary(tmp_path):
    cfg_dict = {"trajectory_processing": {"required_columns": ["t", "x", "y", "theta", "turn"]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    traj_content = "t,x,y,theta,turn\n0,0,0,0,0"
    summary_content = json.dumps({"successrate": "bad", "latency": [1], "n_trials": 1, "timesteps": 1})
    run_dir = make_run_dir(tmp_path, traj_content, summary_content)

    try:
        validate_exported_data(run_dir, cfg)
    except Exception as e:
        assert "successrate" in str(e)
    else:
        raise AssertionError("Validator did not detect invalid summary")


def test_validate_valid_run(tmp_path):
    cfg_dict = {"trajectory_processing": {"required_columns": ["t", "x", "y", "theta", "turn"]}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    traj_content = "t,x,y,theta,turn\n0,0,0,0,0"
    summary_content = json.dumps({"successrate": 1.0, "latency": [1.0], "n_trials": 1, "timesteps": 1})
    run_dir = make_run_dir(tmp_path, traj_content, summary_content)

    validate_exported_data(run_dir, cfg)
