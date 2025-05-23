# Analysis Workflow Overview

This document describes a parameterized approach for processing simulation results. A master YAML file controls every step of the workflow so that analyses remain reproducible and easy to modify.

## 1. Analysis Configuration Setup
- **Master YAML**: The analysis is driven by a configuration file such as `configs/analysis_config.yaml`.
- **Sections** typically include:
  - `data_paths` – locations for raw and processed data.
  - `metadata_extraction` – rules for deriving metadata from folder names.
  - `metrics_calculation` – parameters for computing metrics.
  - `aggregation_groups` – how to group results before summarising.
  - `plotting_parameters` – figure options and metrics to display.
  - `statistical_tests` – tests to run and significance levels.
  - `output_paths` – directories for figures, tables and analysis outputs.

The analysis scripts load this YAML at startup using `load_analysis_config`.

## 2. Data Discovery and Organization
- Directories containing processed results are specified under `data_paths.processed_base_dirs`.
- The `metadata_extraction.directory_template` string describes how to parse metadata (plume, sensing mode, agent id, seed) from subdirectories.
- If `load_run_config` is enabled, each run's `config_used.yaml` is read to access run‑specific parameters.

## 3. Data Loading and Parsing
- Depending on the configuration, scripts may load summary JSON files, trajectories, and parameter snapshots.
- Flags in the YAML file determine which inputs are required for a given analysis pass.

## 4. Metric Computation and Aggregation
- Metric parameters such as thresholds or smoothing windows come directly from `metrics_calculation`.
- Results are aggregated according to the groups listed in `aggregation_groups` to produce summary statistics.

## 5. Plotting and Statistical Tests
- Plot appearance and output formats are controlled via `plotting_parameters`.
- Statistical tests defined in `statistical_tests` are executed on the aggregated data.

## 6. Output
- Tables, figures and processed analysis data are written to the paths given in `output_paths`.

With all parameters stored in a central YAML file, rerunning analyses with different settings becomes straightforward and transparent.

