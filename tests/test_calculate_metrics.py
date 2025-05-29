import os
import sys
import math

import pytest

yaml = pytest.importorskip("yaml")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from Code.load_analysis_config import load_analysis_config
from Code.calculate_metrics import calculate_metrics, _compute_dt


def test_calculate_metrics(tmp_path):
    cfg_dict = {
        "metrics_to_compute": [
            "success_rate",
            "latency",
            "path_length",
            "average_speed",
            "net_upwind_displacement",
            "straightness",
            "turning_rate",
        ],
        "metric_parameters": {
            "average_speed": {"dt_source": "from_latency"},
            "net_upwind_displacement": {
                "upwind_axis": "y",
                "upwind_positive_direction": True,
            },
        },
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = load_analysis_config(cfg_path)

    trajectories = [
        {"t": 0, "x": 0.0, "y": 0.0, "turn": 0},
        {"t": 1, "x": 1.0, "y": 0.0, "turn": 1},
        {"t": 2, "x": 2.0, "y": 0.5, "turn": 0},
    ]
    record = {
        "trajectories": trajectories,
        "summary": {"successrate": 1.0, "latency": 2.0},
        "config": {"frame_rate": 2},
    }

    metrics = calculate_metrics(record, cfg)
    expected_path_length = math.dist((0, 0), (1, 0)) + math.dist((1, 0), (2, 0.5))
    assert metrics["path_length"] == pytest.approx(expected_path_length)
    assert metrics["average_speed"] == pytest.approx(expected_path_length / 2.0)
    assert metrics["net_upwind_displacement"] == pytest.approx(0.5)
    assert metrics["straightness"] == pytest.approx(math.dist((0, 0), (2, 0.5)) / expected_path_length)
    assert metrics["turning_rate"] == pytest.approx(1/3)


def test_compute_dt_from_latency_missing_latency():
    record = {"trajectories": [{"t": 0}, {"t": 1}]}
    params = {"average_speed": {"dt_source": "from_latency"}}
    with pytest.raises(ValueError):
        _compute_dt(record, params)


def test_compute_dt_from_config_missing_frame_rate():
    record = {"trajectories": [{"t": 0}, {"t": 1}], "config": {}}
    params = {"average_speed": {"dt_source": "from_config_used_yaml"}}
    with pytest.raises(ValueError):
        _compute_dt(record, params)


def test_compute_dt_fixed_value_missing_parameter():
    record = {"trajectories": [{"t": 0}]}
    params = {"average_speed": {"dt_source": "fixed_value"}}
    with pytest.raises(ValueError):
        _compute_dt(record, params)


def test_compute_dt_unknown_source():
    record = {"trajectories": [{"t": 0}]}
    params = {"average_speed": {"dt_source": "bogus"}}
    with pytest.raises(ValueError):
        _compute_dt(record, params)
