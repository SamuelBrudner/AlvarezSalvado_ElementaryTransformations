"""Compute behavioral metrics for a single simulation run.

Examples
--------
>>> from Code.calculate_metrics import calculate_metrics
>>> rec = {"trajectories": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "summary": {}}
>>> cfg = {"metrics_to_compute": ["path_length"]}
>>> calculate_metrics(rec, cfg)["path_length"]
1.4142135623730951
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

Trajectory = List[Dict[str, float]]


def _compute_dt(record: Dict[str, Any], metric_params: Dict[str, Any]) -> float:
    opts = metric_params.get("average_speed", {})
    src = opts.get("dt_source", "from_config_used_yaml")
    if src == "from_latency":
        latency = record.get("summary", {}).get("latency")
        traj = record.get("trajectories", [])
        if latency is None or len(traj) < 2:
            raise ValueError("Cannot derive dt from latency")
        return latency / float(len(traj) - 1)
    elif src == "from_config_used_yaml":
        fr_key = opts.get("framerate_field_in_config_used", "frame_rate")
        fr = record.get("config", {}).get(fr_key)
        if fr is None:
            raise ValueError("Frame rate missing for dt calculation")
        return 1.0 / float(fr)
    elif src == "fixed_value":
        val = opts.get("dt_fixed_value")
        if val is None:
            raise ValueError("dt_fixed_value required when dt_source is fixed_value")
        return float(val)
    else:
        raise ValueError(f"Unknown dt_source {src}")


def calculate_metrics(record: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Calculate metrics listed in ``cfg['metrics_to_compute']``.

    Parameters
    ----------
    record : dict
        Data for a single simulation run.
    cfg : dict
        Analysis configuration dictionary.

    Returns
    -------
    dict
        Mapping of metric names to computed values.
    """
    metrics: Dict[str, float] = {}
    metric_list = cfg.get("metrics_to_compute", [])
    params = cfg.get("metric_parameters", {})
    traj: Trajectory = record.get("trajectories", [])

    if not traj:
        raise ValueError("Trajectories data required for metric calculation")

    if "success_rate" in metric_list:
        metrics["success_rate"] = record.get("summary", {}).get("successrate")

    if "latency" in metric_list:
        metrics["latency"] = record.get("summary", {}).get("latency")

    if any(m in metric_list for m in ["path_length", "average_speed", "straightness"]):
        path_length = 0.0
        for p1, p2 in zip(traj[:-1], traj[1:]):
            path_length += math.dist((p1["x"], p1["y"]), (p2["x"], p2["y"]))
        metrics["path_length"] = path_length
    if "average_speed" in metric_list:
        dt = _compute_dt(record, params)
        total_time = dt * (len(traj) - 1)
        metrics["average_speed"] = metrics["path_length"] / total_time
    if "net_upwind_displacement" in metric_list:
        opts = params.get("net_upwind_displacement", {})
        axis = opts.get("upwind_axis", "y")
        pos_dir = opts.get("upwind_positive_direction", True)
        start = traj[0][axis]
        end = traj[-1][axis]
        disp = end - start
        metrics["net_upwind_displacement"] = disp if pos_dir else -disp
    if "straightness" in metric_list:
        net_disp = math.dist(
            (traj[0]["x"], traj[0]["y"]), (traj[-1]["x"], traj[-1]["y"])
        )
        path_len = metrics.get("path_length", 0.0)
        metrics["straightness"] = net_disp / path_len if path_len else float("nan")
    if "turning_rate" in metric_list:
        total_turns = sum(p.get("turn", 0) for p in traj)
        metrics["turning_rate"] = total_turns / float(len(traj))

    return metrics
