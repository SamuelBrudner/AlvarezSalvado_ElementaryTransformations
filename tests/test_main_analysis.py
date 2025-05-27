import os
import sys
import json
import csv

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.main_analysis import run_pipeline


def create_run(base, plume, mode, agent_id, seed, successrate=1.0, latency=2.0):
    run_dir = base / f"{plume}_{mode}" / f"agent_{agent_id}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text("fr: 2\n")
    (run_dir / "summary.json").write_text(json.dumps({"successrate": successrate, "latency": latency}))
    with open(run_dir / "trajectories.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y"])
        writer.writeheader()
        writer.writerow({"t": 0, "x": 0.0, "y": 0.0})
        writer.writerow({"t": 1, "x": 1.0, "y": 0.0})
        writer.writerow({"t": 2, "x": 2.0, "y": 1.0})
    return run_dir


def sample_config(tmp_path, base):
    cfg = {
        "data_paths": {"processed_base_dirs": [str(base)]},
        "metadata_extraction": {"directory_template": "{plume}_{mode}/agent_{agent_id}/seed_{seed}"},
        "data_loading_options": {
            "load_summary_json": True,
            "load_trajectories_csv": True,
            "load_config_used_yaml": True,
        },
        "metrics_to_compute": ["average_speed", "success_rate"],
        "metric_parameters": {"average_speed": {"dt_source": "from_config_used_yaml", "framerate_field_in_config_used": "fr"}},
        "aggregation_options": {"group_by_keys": ["plume_type", "sensing_mode"], "statistics_to_compute": ["mean"]},
        "table_generation": [
            {
                "metrics": ["success_rate"],
                "group_by_keys": ["plume_type", "sensing_mode"],
                "statistic_to_report": "mean",
                "output_filename": "summary.csv",
            }
        ],
        "plotting_tasks": [
            {
                "metric_name": "success_rate",
                "plot_type": "bar",
                "x_axis_grouping": "plume_type",
                "hue_grouping": "sensing_mode",
                "output_filename": "success.png",
            }
        ],
        "statistical_analysis": [
            {
                "test_type": "t_test_ind",
                "metric_name": "success_rate",
                "grouping_variable": "plume_type",
                "groups_to_compare": ["gaussian", "crimaldi"],
            }
        ],
        "output_paths": {
            "figures": str(tmp_path / "figures"),
            "tables": str(tmp_path / "tables"),
            "processed": str(tmp_path / "processed"),
        },
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_run_pipeline(tmp_path):
    base = tmp_path / "data" / "processed"
    create_run(base, "gaussian", "bilateral", 1, 0, successrate=0.8)
    create_run(base, "crimaldi", "unilateral", 1, 0, successrate=0.5)

    cfg_path = sample_config(tmp_path, base)
    cfg = load_analysis_config(cfg_path)

    run_pipeline(cfg)

    assert (tmp_path / "tables" / "summary.csv").exists()
    assert (tmp_path / "figures" / "success.png").exists()
