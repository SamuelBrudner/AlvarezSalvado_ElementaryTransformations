# Intensity Dataset Comparison

This page describes how to characterise the intensity of individual odour plumes and how to compare multiple intensity datasets.

## Characterising a Single Plume

To obtain intensity statistics for a single plume, use the `analyse_crimaldi_data.py` script. The command prints summary statistics such as the peak intensity and integrated intensity over time.

```bash
conda run --prefix ./dev-env python Code/analyse_crimaldi_data.py data/raw/plume1.hdf5
```

Expected output:

```
Peak intensity: 3.2
Mean intensity: 1.4
```

## Comparing Multiple Datasets

Use the `compare_intensity_datasets.py` script with multiple input files. The script computes cross‑dataset statistics and produces a plot showing the distribution of intensities.

```bash
conda run --prefix ./dev-env python Code/compare_intensity_datasets.py data/raw/plume1.hdf5 data/raw/plume2.hdf5
```

Typical output:

```
Plume1 peak: 3.2
Plume2 peak: 2.8
Difference (mean): 0.4
Figure saved to figures/intensity_comparison.png
```

## Notes

- All commands assume the development environment created via `setup_env.sh --dev`.
- Output paths are controlled by `configs/paths.yaml`.
