"""Utilities for comparative analysis and visualization."""

from __future__ import annotations

import json
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Type alias for clarity
Record = Dict[str, Any]


def _group_records(records: Iterable[Record], keys: List[str]) -> Dict[tuple, List[Record]]:
    groups: Dict[tuple, List[Record]] = {}
    for rec in records:
        key = tuple(rec[k] for k in keys)
        groups.setdefault(key, []).append(rec)
    return groups


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _sem(values: List[float]) -> float:
    if not values:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def generate_tables(records: List[Record], cfg: Dict[str, Any]) -> List[Path]:
    """Generate summary tables as specified in the config.

    Parameters
    ----------
    records : list of dict
        Input data records.
    cfg : dict
        Parsed analysis configuration.

    Returns
    -------
    list of Path
        Paths to generated table files.
    """
    tables_cfg = cfg.get("table_generation", [])
    output_dir = Path(cfg.get("output_paths", {}).get("tables", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: List[Path] = []
    for task in tables_cfg:
        metrics = task.get("metrics", [])
        groups = task.get("group_by_keys", [])
        stat = task.get("statistic_to_report", "mean")
        outfile = output_dir / task.get("output_filename", "table.csv")

        grouped = _group_records(records, groups)
        with outfile.open("w", newline="") as f:
            writer = csv.writer(f)
            header = groups + metrics
            writer.writerow(header)
            for gkey, recs in grouped.items():
                row = list(gkey)
                for m in metrics:
                    vals = [r[m] for r in recs if m in r]
                    if stat == "mean":
                        row.append(_mean(vals))
                    else:
                        row.append(_mean(vals))
                writer.writerow(row)
        output_files.append(outfile)
    return output_files


def _normal_p_value(t_stat: float) -> float:
    """Approximate two-tailed p-value using the normal distribution."""
    return 2 * (1 - statistics.NormalDist().cdf(abs(t_stat)))


def run_statistical_tests(records: List[Record], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run statistical tests specified in the config.

    Currently supports independent t-tests.
    """
    tests_cfg = cfg.get("statistical_analysis", [])
    results = []
    for task in tests_cfg:
        metric = task.get("metric_name")
        group_var = task.get("grouping_variable")
        groups = task.get("groups_to_compare", [])
        if len(groups) != 2:
            continue
        a = [rec[metric] for rec in records if rec.get(group_var) == groups[0]]
        b = [rec[metric] for rec in records if rec.get(group_var) == groups[1]]
        if not a or not b:
            continue
        mean_a, mean_b = _mean(a), _mean(b)
        var_a = statistics.pvariance(a) if len(a) > 1 else 0.0
        var_b = statistics.pvariance(b) if len(b) > 1 else 0.0
        se = math.sqrt(var_a / len(a) + var_b / len(b))
        t_stat = (mean_a - mean_b) / se if se != 0 else float("inf")
        p_val = _normal_p_value(t_stat)
        results.append({
            "metric": metric,
            "groups": groups,
            "t_stat": t_stat,
            "p_value": p_val,
        })
    return results


def generate_plots(records: List[Record], cfg: Dict[str, Any]) -> List[Path]:
    """Generate placeholder plot files as specified in the config."""
    plots_cfg = cfg.get("plotting_tasks", [])
    output_dir = Path(cfg.get("output_paths", {}).get("figures", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for task in plots_cfg:
        outfile = output_dir / task.get("output_filename", "plot.png")
        # Create a placeholder file since matplotlib is unavailable
        outfile.write_text(f"Plot placeholder for {task.get('metric_name')}")
        paths.append(outfile)
    return paths
