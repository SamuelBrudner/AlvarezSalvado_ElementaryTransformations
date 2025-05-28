# AlvarezSalvado Elementary Transformations

This repository hosts MATLAB and Python code for the navigation model from Álvarez-Salvado *et al.* (2018) and related analysis tools.

For detailed instructions on environment setup and intensity comparison workflows, see [docs/intensity_comparison.md](docs/intensity_comparison.md).

- [Quick Start](#quick-start)
- [Step-by-Step](#step-by-step)
- [Overview](#overview)
- [Requirements](#requirements)
- [Development Environment Setup](#development-environment-setup)
- [Configuration](#configuration)
- [Data Organization](#data-organization)
- [Data Dictionary](#data-dictionary)
- [Exporting Results to Open Formats](#exporting-results-to-open-formats)
- [Running Simulations](#running-simulations)
- [Analysis Configuration](#analysis-configuration)
- [Plume Intensity Utilities](#plume-intensity-utilities)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [Roadmap](#roadmap)

## Quick Start

Check if the conda environment exists, if not, create it.

 - Check: `./dev_env` exists
 - If not, run: `./setup_env.sh --dev`

Check if pre commit hooks are installed, if not, install them.

 - Check: `./dev_env/.git/hooks/pre-commit` exists
 - If not, run: `conda run --prefix dev_env pre-commit install`

Sourcing ``paths.sh`` creates ``configs/project_paths.yaml`` from the template
if it does not already exist.

`paths.sh` attempts to locate MATLAB automatically and sets `MATLAB_EXEC`.
Set this variable yourself or use the `--matlab_exec` option to override the
auto-detected path.
If MATLAB is not found, `paths.sh` tries to load a module named
`MATLAB/$MATLAB_VERSION` (or `MATLAB_MODULE` if set). Set these variables
before sourcing to override the default.
Python utilities such as `Code.video_intensity` also honour `MATLAB_EXEC` when
it points to a valid executable.
Use `--allow-mismatch` with `Code.compare_intensity_stats` if your datasets have
different lengths.

See [docs/intensity_comparison.md](docs/intensity_comparison.md#initial-setup)
for a detailed explanation of the path setup process.

With the environment active you can run MATLAB and Python scripts from the `Code` directory using the module syntax:

```bash
conda run --prefix ./dev-env python -m Code.some_script
```

## Step-by-Step

1 and 2 are only *as required* -- check if there's evidence they've already run

1. Run `./setup_env.sh --dev` to create `./dev-env`.
2. Source `./paths.sh` to generate `configs/project_paths.yaml` and detect MATLAB. `paths.sh` uses this file and falls back to default paths when `yq` is missing.
3. Execute `conda run --prefix ./dev-env python -m Code.compare_intensity_stats`.

## Directory Overview

- `Code/` – MATLAB and Python modules
- `configs/` – configuration templates
- `data/` – raw and processed datasets
- `docs/` – project documentation

For citations and metadata, see `CITATION.cff` and `codemeta.json`.
