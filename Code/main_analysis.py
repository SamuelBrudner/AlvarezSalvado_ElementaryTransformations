"""Master script for batch post-processing and analysis.

Examples
--------
>>> from Code.main_analysis import run_pipeline
>>> run_pipeline("configs/example_analysis.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from Code.calculate_metrics import calculate_metrics
from Code.comparative_analysis import (generate_plots, generate_tables,
                                       run_statistical_tests)
from Code.data_aggregation import aggregate_metrics
from Code.data_discovery import (check_parameter_consistency,
                                 discover_processed_data)
from Code.load_analysis_config import load_analysis_config

Record = dict[str, Any]


def run_pipeline(cfg_or_path: str | Path | Dict[str, Any]) -> None:
    """Execute the full analysis pipeline based on the config."""
    if isinstance(cfg_or_path, (str, Path)):
        cfg = load_analysis_config(cfg_or_path)
    else:
        cfg = cfg_or_path

    records: List[Record] = []
    discovered = list(discover_processed_data(cfg))
    check_parameter_consistency(discovered, cfg)

    for rec in discovered:
        metrics = calculate_metrics(rec, cfg)
        metadata = rec.get("metadata", {})
        metrics.update(
            {
                "plume_type": metadata.get("plume"),
                "sensing_mode": metadata.get("mode"),
            }
        )
        records.append(metrics)

    aggregate_metrics(records, cfg)
    generate_tables(records, cfg)
    generate_plots(records, cfg)
    run_statistical_tests(records, cfg)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run post-processing pipeline")
    parser.add_argument("config", help="Path to analysis_config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
