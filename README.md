# AlvarezSalvado Elementary Transformations

This repository hosts MATLAB and Python code for the navigation model from Álvarez-Salvado *et al.* (2018) and related analysis tools.

For detailed instructions on environment setup and intensity comparison workflows, see [docs/intensity_comparison.md](docs/intensity_comparison.md).

- [Quick Start](#quick-start)
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

```bash
./setup_env.sh --dev
source ./paths.sh
```

With the environment active you can run MATLAB and Python scripts from the `Code` directory using the module syntax:

```bash
conda run --prefix ./dev-env python -m Code.some_script
```

## Directory Overview

- `Code/` – MATLAB and Python modules
- `configs/` – configuration templates
- `data/` – raw and processed datasets
- `docs/` – project documentation

For citations and metadata, see `CITATION.cff` and `codemeta.json`.
