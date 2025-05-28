"""Scan simulation runs for common anomalies."""

from __future__ import annotations

import math
from typing import List, Dict, Any

from Code.data_discovery import discover_processed_data


def check_simulation_health(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Check runs specified in the config and return detected issues."""
    opts = cfg.setdefault("data_loading_options", {})
    opts.setdefault("load_summary_json", True)
    opts.setdefault("load_config_used_yaml", True)
    opts.setdefault("load_params_json", False)
    opts.setdefault("load_trajectories_csv", False)

    bounds = cfg.get("health_checks", {}).get("reasonable_bounds", {})
    require_traj = cfg.get("health_checks", {}).get("require_trajectories", False)

    issues: List[Dict[str, str]] = []
    for rec in discover_processed_data(cfg):
        path = rec["path"]
        summary = rec.get("summary", {})
        config = rec.get("config", {})
        trajectories = rec.get("trajectories")

        sr = summary.get("successrate")
        if sr is None or (isinstance(sr, float) and math.isnan(sr)):
            issues.append({"path": path, "issue": "successrate is NaN"})
        else:
            b = bounds.get("successrate")
            if b and not (b[0] <= sr <= b[1]):
                issues.append({"path": path, "issue": f"successrate {sr} outside [{b[0]}, {b[1]}]"})

        if "n_trials" in summary and "ntrials" in config:
            if summary["n_trials"] != config["ntrials"]:
                issues.append({
                    "path": path,
                    "issue": f"n_trials {summary['n_trials']} != config ntrials {config['ntrials']}",
                })

        if require_traj:
            if trajectories is None:
                issues.append({"path": path, "issue": "trajectories missing"})
            elif len(trajectories) == 0:
                issues.append({"path": path, "issue": "trajectories empty"})

    return issues


def main(argv: List[str] | None = None) -> None:
    import argparse
    from Code.load_analysis_config import load_analysis_config

    parser = argparse.ArgumentParser(description="Check simulation directories for anomalies")
    parser.add_argument("config", help="Analysis config YAML file")
    args = parser.parse_args(argv)

    cfg = load_analysis_config(args.config)
    issues = check_simulation_health(cfg)
    if issues:
        for item in issues:
            print(f"{item['path']}: {item['issue']}")
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
