import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.generate_dashboard import generate_dashboard
from Code.load_analysis_config import load_analysis_config


def create_sample_data():
    return [
        {"plume_type": "Crimaldi_bilateral", "sensing_mode": "mode1", "success_rate": 0.8, "latency": 2.0, "path_length": 10},
        {"plume_type": "Crimaldi_bilateral", "sensing_mode": "mode2", "success_rate": 0.9, "latency": 2.5, "path_length": 12},
        {"plume_type": "other", "sensing_mode": "mode1", "success_rate": 0.5, "latency": 3.0, "path_length": 15},
        {"plume_type": "other", "sensing_mode": "mode2", "success_rate": 0.6, "latency": 3.5, "path_length": 16},
    ]


def dashboard_config(tmp_path):
    cfg_dict = {
        "dashboard_layout": {
            "output_filename": "dashboard.png",
            "subplots": [
                {
                    "metric": "success_rate",
                    "plot_type": "bar",
                    "group_by": "plume_type",
                    "title": "Success Rate by Plume",
                },
                {
                    "metric": "latency",
                    "plot_type": "box",
                    "group_by": "sensing_mode",
                    "title": "Latency by Sensing Mode",
                },
                {
                    "metric": "path_length",
                    "plot_type": "hist",
                    "filters": {"plume_type": "Crimaldi_bilateral"},
                    "title": "Path Length Distribution",
                },
            ],
        },
        "output_paths": {"figures": str(tmp_path)},
    }
    cfg_path = tmp_path / "analysis_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    return cfg_path


def test_generate_dashboard(tmp_path):
    cfg_path = dashboard_config(tmp_path)
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    fig_path = generate_dashboard(data, cfg)
    assert fig_path.exists()
