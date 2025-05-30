# Simulation Analysis Workflow

This guide describes how to process results once your navigation simulations have finished. It assumes runs were produced by `run_batch_job_4000.sh` or a related wrapper.

## Step-by-Step

1. **Create the environment** (only the first time)
   ```bash
   ./setup_env.sh --dev
   ```
2. **Load project paths**
   ```bash
   source ./paths.sh
   ```
   This generates `configs/project_paths.yaml` if missing and detects MATLAB.
3. **Configure the analysis**
   Use `configs/analysis_config.yaml` as a template. Edit the `data_paths.processed_base_dirs` entry so it points to your processed results. Other sections control metrics, aggregation and plotting options (see [docs/analysis_plan.md](analysis_plan.md)).
4. **Run the pipeline**
   ```bash
   conda run --prefix ./dev_env python -m Code.main_analysis configs/analysis_config.yaml
   ```
5. **Inspect the output**
   Tables, figures and aggregated data are written to the directories defined under `output_paths` in the config file.

## Notes
- The pipeline loads all processed runs matching the directory template specified in `metadata_extraction.directory_template`.
- Statistical tests defined in the YAML configuration are executed automatically after metrics are aggregated.

For a detailed explanation of every configuration field, see [docs/analysis_plan.md](analysis_plan.md).
