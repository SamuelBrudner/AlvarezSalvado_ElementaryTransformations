import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.check_simulation_health import check_simulation_health


def test_check_simulation_health_flags_issues(tmp_path):
    base = tmp_path / "data" / "raw"
    run_ok = base / "gaussian_bilateral" / "agent_0" / "seed_0"
    run_nan = base / "gaussian_bilateral" / "agent_1" / "seed_0"
    run_mismatch = base / "gaussian_bilateral" / "agent_2" / "seed_0"

    for p in [run_ok, run_nan, run_mismatch]:
        p.mkdir(parents=True)
        (p / "config_used.yaml").write_text("ntrials: 100\n")

    json.dump({"successrate": 0.8, "n_trials": 100}, (run_ok / "summary.json").open("w"))
    json.dump({"successrate": float('nan'), "n_trials": 100}, (run_nan / "summary.json").open("w"))
    json.dump({"successrate": 0.7, "n_trials": 1}, (run_mismatch / "summary.json").open("w"))

    cfg = {
        "data_paths": {"processed_base_dirs": [str(base)]},
        "metadata_extraction": {"directory_template": "{plume}_{mode}/agent_{agent_id}/seed_{seed}"},
        "data_loading_options": {"load_summary_json": True, "load_config_used_yaml": True},
        "health_checks": {"reasonable_bounds": {"successrate": [0.0, 1.0]}}
    }

    issues = check_simulation_health(cfg)
    assert any("successrate is NaN" in issue["issue"] for issue in issues)
    assert any("n_trials" in issue["issue"] and "config" in issue["issue"] for issue in issues)
