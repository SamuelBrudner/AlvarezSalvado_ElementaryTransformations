"""Generate a dashboard figure with multiple subplots based on config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Record = Dict[str, Any]


def _apply_filters(records: List[Record], filters: Dict[str, Any]) -> List[Record]:
    if not filters:
        return records
    return [rec for rec in records if all(rec.get(k) == v for k, v in filters.items())]


def generate_dashboard(records: List[Record], cfg: Dict[str, Any]):
    layout = cfg.get("dashboard_layout", {})
    subplots = layout.get("subplots", [])
    if not subplots:
        return None
    output_dir = Path(cfg.get("output_paths", {}).get("figures", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = layout.get("output_filename", "dashboard.png")

    fig, axes = plt.subplots(1, len(subplots), figsize=(5 * len(subplots), 4))
    if len(subplots) == 1:
        axes = [axes]

    for ax, spec in zip(axes, subplots):
        metric = spec.get("metric")
        plot_type = spec.get("plot_type", "bar")
        group_by = spec.get("group_by")
        filters = spec.get("filters", {})

        subset = _apply_filters(records, filters)

        if group_by:
            grouped = {}
            for rec in subset:
                key = rec.get(group_by)
                grouped.setdefault(key, []).append(rec.get(metric))
            groups = list(grouped.keys())
            data = [grouped[g] for g in groups]
        else:
            data = [rec.get(metric) for rec in subset]
            groups = [metric] if plot_type in ("bar", "box") else None
            if plot_type in ("bar", "box"):
                data = [data]

        if plot_type == "bar":
            means = [sum(vals) / len(vals) for vals in data]
            ax.bar(groups, means)
        elif plot_type == "box":
            ax.boxplot(data, labels=groups)
        elif plot_type == "hist":
            values = data if groups is None else [v for vals in data for v in vals]
            ax.hist(values, bins=10)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        ax.set_title(spec.get("title", metric))
        ax.set_ylabel(metric)
        if group_by:
            ax.set_xlabel(group_by)

    fig.tight_layout()
    out_path = output_dir / fname
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
