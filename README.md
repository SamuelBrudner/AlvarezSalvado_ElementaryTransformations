# AlvarezSalvado_ElementaryTransformations

This repository accompanies the paper **"Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies"** by Álvarez-Salvado *et al.* (eLife, 2018). It contains code, CAD drawings and LabVIEW programs used in the publication.

## Repository layout

- **Code/** – MATLAB scripts for analyzing behavior and running the navigation model. This folder also contains an `import functions feb2017` subfolder for reading binary tracking data produced with the Miniature Manhattan wind tunnels.
- **Arena 4f/** – Laser cut design files (AI format) and instructions describing how to assemble the behavioral arena.
- **WalkingArenaLoopLongerIRtime2/** – Arduino sketch for controlling arena lighting, odor valves and camera triggers.
- **FlyTracker 3.6.vi**, **Get Frame Timestamp.vi**, **SelectBackground.vi** – LabVIEW VIs used for data acquisition and preprocessing.

## Quick start

The MATLAB code was written for MATLAB R2017b and depends on functions in the `Code/` and `Code/import functions feb2017/` directories. To simulate the navigation model, open MATLAB in the repository root and run:

```matlab
addpath(genpath('Code'));
out = Elifenavmodel_bilateral(5000, 'Crimaldi', 1, 1); % example simulation
```

Data collected with the Miniature Manhattan wind tunnels can be imported with the `importer` function in `Code/import functions feb2017`.

## Development environment

If your Codex environment includes a `setup_env.sh` script, run it with the `--dev` flag to create a development conda environment before executing the MATLAB code or unit tests:

```bash
bash setup_env.sh --dev
```

This repository includes a few automated tests under the `tests/` directory. If your Codex environment provides a `setup_env.sh` script, you can use it to create a conda environment with required dependencies before running the tests or MATLAB code.

## Running simulations on SLURM

Use `slurm_submit.sh` to generate a batch file from `slurm_job_template.slurm`. The
script reads several environment variables, all of which have sensible defaults:

- `AGENTS_PER_CONDITION` (default `1000`)
- `AGENTS_PER_JOB` (default `10`)
- `TRIAL_LENGTH` (default `5000`)
- `ENVIRONMENT` (default `Crimaldi`)
- `OUTPUT_DIR` (defaults to the current directory)
- `PARTITION` (default `day`)
- `TIME_LIMIT` (default `6:00:00`)
- `MEM_PER_TASK` (default `64G`)
- `MAX_CONCURRENT` (default `100`)
- `EXP_NAME` (default `crimaldi`)

If present, you may optionally run `bash setup_env.sh --dev` before submitting
jobs to set up a conda environment.

Example workflow:

```bash
bash slurm_submit.sh job.slurm
sbatch job.slurm
```

## License

This project is released under the MIT License (see [LICENSE](LICENSE)).
