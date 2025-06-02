# AlvarezSalvado_ElementaryTransformations

This repository accompanies the paper **"Elementary sensory-motor transformations underlying olfactory navigation in walking fruit flies"** by Álvarez-Salvado *et al.* (eLife, 2018). It contains code, CAD drawings and LabVIEW programs used in the publication.

## Repository layout

- **Code/** – MATLAB scripts for analyzing behavior and running the navigation model. This folder also contains an `import functions feb2017` subfolder for reading binary tracking data produced with the Miniature Manhattan wind tunnels.
- **Arena 4f/** – Laser cut design files (AI format) and instructions describing how to assemble the behavioral arena.
- **WalkingArenaLoopLongerIRtime2/** – Arduino sketch for controlling arena lighting, odor valves and camera triggers.
- **FlyTracker 3.6.vi**, **Get Frame Timestamp.vi**, **SelectBackground.vi** – LabVIEW VIs used for dSMOKE:
  mean: 0.359
  median: 0.498
  p95: 0.502
  p99: 0.502
  min: 0.047
  max: 0.506
  count: 6768230400
CRIM:
  mean: 0.059
  median: 0.038
  p95: 0.190
  p99: 0.325
  min: -0.020
  max: 1.291
  count: 315705600ata acquisition and preprocessing.

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

This repository now includes a small pytest suite. After creating the development environment you can run the tests with:

```bash
pytest -q
```

The `slurm_submit.sh` helper script supports a `-h`/`--help` option that prints
usage instructions and also logs the calculated array size and selected paths
when invoked.


## License

This project is released under the MIT License (see [LICENSE](LICENSE)).
