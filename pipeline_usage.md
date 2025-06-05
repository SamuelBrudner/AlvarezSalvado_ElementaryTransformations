# Pipeline Workflow

`run_my_pipeline.sh` provides an end-to-end workflow for running navigation model simulations on a SLURM cluster. It automates configuration generation, job submission, result monitoring and basic analysis.

## Prerequisites

1. Create the development environment:
   ```bash
   bash setup_env.sh --dev
   ```
2. Run the unit tests:
   ```bash
   pytest -q
   ```
3. Ensure MATLAB and `sbatch` are available on your PATH.

## Usage

Execute the pipeline from the repository root:

```bash
bash ./run_my_pipeline.sh
```

The script will:

1. Generate clean configuration files with `generate_clean_configs.m`.
2. Submit short test jobs on Crimaldi and Smoke plumes.
3. Wait until both jobs leave the pending or running state.
4. Summarize results with `create_results_report.sh` and generate figures via `run_plot_results.sh`.

Logs are written to `slurm_logs/nav_crim/` and `slurm_logs/nav_smoke/`. Results appear in the `results/` directory.

## Output

A summary report named `pipeline_results_summary_<timestamp>.txt` is placed in the project root. Individual SLURM logs and MATLAB figures remain in their respective directories.

