# AlvarezSalvado Elementary Transformations

This repository hosts MATLAB and Python code for the navigation model from Álvarez-Salvado *et al.* (2018) and related analysis tools.

For detailed instructions on environment setup and intensity comparison workflows, see [docs/intensity_comparison.md](docs/intensity_comparison.md).

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
