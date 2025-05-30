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
- `PLUME_METADATA` – YAML file describing the HDF5 movie when `PLUME_VIDEO`
  points to an `.h5` file.
- `OUTPUT_BASE` – parent directory for raw outputs (`data/raw`).
- `MATLAB_VERSION` – module name used if MATLAB is not on `PATH` (`2023b`).
- `BYTES_PER_AGENT` – estimated disk usage per agent (defaults to `50000000`).

### Video and Metadata Configuration

`plume_video` and `plume_metadata` are mutually exclusive keys in a
plume configuration file. Use `plume_video` for raw movies (AVI, MP4,
etc.) together with `px_per_mm` and `frame_rate`. When a movie has been
converted to HDF5, reference the companion YAML under `plume_metadata`
instead.

Generate such a YAML with:

```bash
conda run --prefix ./dev_env python -m scripts.process_custom_plume \
    configs/my_plume.yaml
```

During batch runs you can override the default video by exporting
`PLUME_METADATA`:

```bash
sbatch --export=ALL,PLUME_METADATA=configs/my_plume_meta.yaml \
       run_batch_job_4000.sh
```
The script assigns `cfg.plume_metadata` when this variable is set
and falls back to `cfg.plume_video` otherwise (see lines 103–111 of
`run_batch_job_4000.sh`).

## Submitting the Job

`run_batch_job_4000.sh` is designed to be submitted as a SLURM array. A minimal submission might look like:

```bash
sbatch --array=0-399 --export=ALL,EXPERIMENT_NAME=myrun,\\
PLUME_VIDEO=data/smoke_plume.h5,PLUME_METADATA=data/smoke_plume_meta.yaml \
       run_batch_job_4000.sh
```

The array size should equal the total number of jobs computed from `AGENTS_PER_CONDITION`, `AGENTS_PER_JOB`, `PLUME_TYPES` and `SENSING_MODES`. The script checks available disk space before starting and loads the appropriate MATLAB module.

Use `run_full_batch.sh` for the standard four‑condition production run or `run_test_batch.sh` for a quick smoke test. Both wrappers export the necessary variables before calling `run_batch_job_4000.sh`.

When `PLUME_METADATA` is exported as shown above, `run_test_batch.sh` submits a short array job for debugging:

```bash
export PLUME_METADATA=configs/my_plume_meta.yaml
bash run_test_batch.sh
```

## Output

Results are written under `${OUTPUT_BASE}/${EXPERIMENT_NAME}/<plume>_<mode>/<agent>_<seed>` as MATLAB `.mat` files. After MATLAB completes, the script converts each result into CSV and JSON using `export_results`. Processed data are placed under `data/processed` with the same folder hierarchy.

