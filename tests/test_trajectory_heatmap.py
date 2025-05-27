import os
import sys
import csv
import tempfile

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.comparative_analysis import generate_trajectory_heatmaps


def sample_records():
    return [
        {
            "metadata": {"outcome": "success"},
            "trajectories": [
                {"x": 0.2, "y": 0.2},
                {"x": 0.3, "y": 0.3},
            ],
        },
        {
            "metadata": {"outcome": "fail"},
            "trajectories": [
                {"x": 0.8, "y": 0.8},
            ],
        },
    ]


def sample_config(tmp_path):
    cfg_dict = {
        "heatmap_generation": [
            {
                "condition_key": "outcome",
                "condition_value": "success",
                "bins": [2, 2],
                "range": [[0.0, 1.0], [0.0, 1.0]],
                "output_filename": "success_heatmap.csv",
            }
        ],
        "output_paths": {"figures": str(tmp_path)},
    }
    path = tmp_path / "analysis_config.yaml"
    path.write_text(yaml.safe_dump(cfg_dict))
    return path


def read_csv(path):
    with open(path, newline="") as f:
        return [list(map(int, row)) for row in csv.reader(f)]


def test_generate_heatmap_counts(tmp_path):
    cfg_path = sample_config(tmp_path)
    cfg = load_analysis_config(cfg_path)
    data = sample_records()

    files = generate_trajectory_heatmaps(data, cfg)
    assert len(files) == 1
    heatmap = read_csv(files[0])
    assert heatmap == [[2, 0], [0, 0]]
