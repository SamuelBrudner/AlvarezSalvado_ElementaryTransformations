import os
import sys
import yaml
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config
from Code.comparative_analysis import generate_tables, run_statistical_tests, generate_plots


def create_sample_data():
    return [
        {"plume_type": "crimaldi", "sensing_mode": "bilateral", "success_rate": 0.8, "latency": 2.0},
        {"plume_type": "crimaldi", "sensing_mode": "bilateral", "success_rate": 0.9, "latency": 2.2},
        {"plume_type": "custom_video", "sensing_mode": "bilateral", "success_rate": 0.5, "latency": 3.0},
        {"plume_type": "custom_video", "sensing_mode": "bilateral", "success_rate": 0.6, "latency": 3.1},
    ]


def sample_config(tmp_path, stat="mean"):
    cfg_dict = {
        "plotting_tasks": [
            {
                "metric_name": "success_rate",
                "plot_type": "bar",
                "x_axis_grouping": "plume_type",
                "hue_grouping": "sensing_mode",
                "error_bar": "sem",
                "title": "Mean Success Rate Comparison",
                "y_label": "Success Rate",
                "output_filename": "success_rate.png",
            }
        ],
        "table_generation": [
            {
                "metrics": ["success_rate", "latency"],
                "group_by_keys": ["plume_type", "sensing_mode"],
                "statistic_to_report": stat,
                "output_filename": "summary.csv",
            }
        ],
        "statistical_analysis": [
            {
                "test_type": "t_test_ind",
                "metric_name": "success_rate",
                "grouping_variable": "plume_type",
                "groups_to_compare": ["crimaldi", "custom_video"],
                "alpha_level": 0.05,
            }
        ],
        "output_paths": {"figures": str(tmp_path), "tables": str(tmp_path), "processed": str(tmp_path)},
    }
    cfg_path = tmp_path / "analysis_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    return cfg_path


def test_generate_tables_and_stats(tmp_path):
    cfg_path = sample_config(tmp_path)
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    table_files = generate_tables(data, cfg)
    assert len(table_files) == 1
    table_path = table_files[0]
    assert table_path.exists()
    assert table_path.read_text().startswith("plume_type,sensing_mode")

    results = run_statistical_tests(data, cfg)
    assert len(results) == 1
    res = results[0]
    assert "t_stat" in res and "p_value" in res


def test_generate_plots_placeholder(tmp_path):
    cfg_path = sample_config(tmp_path)
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    plot_files = generate_plots(data, cfg)
    assert len(plot_files) == 1
    assert plot_files[0].exists()


def read_table(path):
    import csv
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def test_generate_tables_mean(tmp_path):
    cfg_path = sample_config(tmp_path, stat="mean")
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    table_files = generate_tables(data, cfg)
    rows = read_table(table_files[0])
    assert float(rows[1][2]) == pytest.approx(0.85, rel=1e-6)
    assert float(rows[1][3]) == pytest.approx(2.1, rel=1e-6)
    assert float(rows[2][2]) == pytest.approx(0.55, rel=1e-6)
    assert float(rows[2][3]) == pytest.approx(3.05, rel=1e-6)


def test_generate_tables_sem(tmp_path):
    cfg_path = sample_config(tmp_path, stat="sem")
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    table_files = generate_tables(data, cfg)
    rows = read_table(table_files[0])
    assert float(rows[1][2]) == pytest.approx(0.05, rel=1e-6)
    assert float(rows[1][3]) == pytest.approx(0.1, rel=1e-6)
    assert float(rows[2][2]) == pytest.approx(0.05, rel=1e-6)
    assert float(rows[2][3]) == pytest.approx(0.05, rel=1e-6)


def test_generate_tables_invalid_stat(tmp_path):
    cfg_path = sample_config(tmp_path, stat="median")
    cfg = load_analysis_config(cfg_path)
    data = create_sample_data()

    with pytest.raises(ValueError):
        generate_tables(data, cfg)

