# Running `run_batch_job_4000.sh`

`run_batch_job_4000.sh` is a SLURM wrapper for launching large batches of navigation simulations. It splits agent runs across multiple array jobs and ensures MATLAB results are exported to CSV and JSON. This page explains the variables you can override and how to submit the script on an HPC cluster.

## Prerequisites

1. **Create the development environment**
   ```bash
   ./setup_env.sh --dev
   ```
   If the script installs a new conda distribution it may require network access.
2. **Update project paths**
   ```bash
   source ./paths.sh
   ```
   This generates `configs/project_paths.yaml` if missing and attempts to locate MATLAB.

## Key Variables

The script accepts configuration via environment variables. Important ones include:

- `EXPERIMENT_NAME` – subdirectory under `data/raw` for results (`default_experiment` by default).
- `PLUME_TYPES` – space‑separated list such as `"crimaldi custom"`.
- `SENSING_MODES` – typically `"bilateral unilateral"`.
- `AGENTS_PER_CONDITION` – total agents to simulate for each plume/sensing pair.
- `AGENTS_PER_JOB` – number of agents each array task runs.
- `PLUME_CONFIG` – path to the YAML configuration file.
- `PLUME_VIDEO` – movie used by the simulator.
- `OUTPUT_BASE` – parent directory for raw outputs (`data/raw`).
- `MATLAB_VERSION` – module name used if MATLAB is not on `PATH` (`2023b`).
- `BYTES_PER_AGENT` – estimated disk usage per agent (defaults to `50000000`).

## Submitting the Job

`run_batch_job_4000.sh` is designed to be submitted as a SLURM array. A minimal submission might look like:

```bash
sbatch --array=0-399 --export=ALL,EXPERIMENT_NAME=myrun \
       run_batch_job_4000.sh
```

The array size should equal the total number of jobs computed from `AGENTS_PER_CONDITION`, `AGENTS_PER_JOB`, `PLUME_TYPES` and `SENSING_MODES`. The script checks available disk space before starting and loads the appropriate MATLAB module.

Use `run_full_batch.sh` for the standard four‑condition production run or `run_test_batch.sh` for a quick smoke test. Both wrappers export the necessary variables before calling `run_batch_job_4000.sh`.

## Output

Results are written under `${OUTPUT_BASE}/${EXPERIMENT_NAME}/<plume>_<mode>/<agent>_<seed>` as MATLAB `.mat` files. After MATLAB completes, the script converts each result into CSV and JSON using `export_results`. Processed data are placed under `data/processed` with the same folder hierarchy.

