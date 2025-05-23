import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.data_loading import load_trajectories


def test_load_trajectories_required_columns(tmp_path):
    csv_path = tmp_path / "trajectories.csv"
    csv_path.write_text(
        "t,trial,x,y,theta,odor,ON,OFF,turn\n"
        "0,0,1,2,3,0.1,0.1,0.1,0\n"
        "1,0,1.1,2.1,3.1,0.2,0.2,0.2,1\n"
    )

    cfg_dict = {
        "trajectory_processing": {
            "required_columns": ["t", "x", "y", "theta", "turn"]
        }
    }
    cfg_path = tmp_path / "analysis_config.yaml"
    cfg_path.write_text(json.dumps(cfg_dict))

    cfg = load_analysis_config(cfg_path)
    df = load_trajectories(csv_path, cfg)
    assert list(df.columns) == ["t", "x", "y", "theta", "turn"]
