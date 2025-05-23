"""Aggregate agent metrics based on analysis configuration."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any, Tuple
import math
import statistics


def aggregate_metrics(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[Tuple, Dict[str, Dict[str, float]]]:
    """Aggregate metrics by groups defined in the config.

    Parameters
    ----------
    records : list of dict
        Each dictionary contains metadata and metric values.
    cfg : dict
        Analysis configuration containing ``aggregation_options`` with keys
        ``group_by_keys`` and ``statistics_to_compute``.

    Returns
    -------
    dict
        Nested dictionary keyed by the grouping tuple. Each metric has another
        dictionary with computed statistics.
    """
    opts = cfg.get("aggregation_options", {})
    group_keys = opts.get("group_by_keys", [])
    stats_to_compute = opts.get(
        "statistics_to_compute",
        ["mean", "std", "sem", "count", "median", "min", "max"],
    )

    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = tuple(rec.get(k) for k in group_keys)
        grouped[key].append(rec)

    summary: Dict[Tuple, Dict[str, Dict[str, float]]] = {}
    for key, items in grouped.items():
        metrics: Dict[str, Dict[str, float]] = {}
        # determine metric names (exclude group keys)
        metric_names = [m for m in items[0].keys() if m not in group_keys]
        for name in metric_names:
            values = [float(it[name]) for it in items if name in it]
            stats: Dict[str, float] = {}
            if not values:
                metrics[name] = stats
                continue
            if "mean" in stats_to_compute or "std" in stats_to_compute or "sem" in stats_to_compute:
                mean_val = sum(values) / len(values)
                if "mean" in stats_to_compute:
                    stats["mean"] = mean_val
            if "std" in stats_to_compute or "sem" in stats_to_compute:
                if len(values) > 1:
                    var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
                    std_val = math.sqrt(var)
                else:
                    std_val = float("nan")
                if "std" in stats_to_compute:
                    stats["std"] = std_val
                if "sem" in stats_to_compute:
                    stats["sem"] = std_val / math.sqrt(len(values)) if len(values) > 0 else float("nan")
            if "count" in stats_to_compute:
                stats["count"] = len(values)
            if "median" in stats_to_compute:
                stats["median"] = statistics.median(values)
            if "min" in stats_to_compute:
                stats["min"] = min(values)
            if "max" in stats_to_compute:
                stats["max"] = max(values)
            metrics[name] = stats
        summary[key] = metrics
    return summary
