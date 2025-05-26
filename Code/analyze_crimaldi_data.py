"""Utility to compute statistics from the Crimaldi plume dataset.

This module provides a function :func:`analyze_crimaldi_data` that loads the
``10302017_10cms_bounded.hdf5`` file and returns summary statistics of the
``/dataset_1`` dataset. It can also be executed as a script to print the
statistics in a human‑readable form.
"""

from __future__ import annotations

import argparse
from typing import Dict
import os

try:
    import numpy as np
    import h5py
except ImportError as exc:  # pragma: no cover - dependencies missing
    raise ImportError(
        "analyze_crimaldi_data requires numpy and h5py to be installed"
    ) from exc


def analyze_crimaldi_data(path: str) -> Dict[str, float | dict]:
    """Return basic statistics for the ``/dataset_1`` array.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing ``/dataset_1``.

    Returns
    -------
    dict
        Dictionary with ``min``, ``max``, ``mean``, ``std``, and
        ``percentiles`` keys.
    """
    with h5py.File(path, "r") as f:
        data = f["/dataset_1"][:]

    stats = {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "percentiles": {
            1: float(np.percentile(data, 1)),
            5: float(np.percentile(data, 5)),
            95: float(np.percentile(data, 95)),
            99: float(np.percentile(data, 99)),
        },
    }
    return stats


def get_intensities_from_crimaldi(
    hdf5_path: str, dataset_name: str = "dataset_1"
) -> np.ndarray:
    """Return a flat array of intensity values from the Crimaldi plume file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file to read.
    dataset_name : str, optional
        Name of the dataset within the file. Defaults to ``dataset_1``.

    Returns
    -------
    numpy.ndarray
        Flattened 1-D array of the intensity values.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If ``dataset_name`` is not found in the file.
    """

    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in {hdf5_path}"
            )
        data = f[dataset_name][:]

    return data.flatten()


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description="Analyze Crimaldi data range")
    parser.add_argument(
        "path",
        help="Path to 10302017_10cms_bounded.hdf5",
    )
    args = parser.parse_args()
    stats = analyze_crimaldi_data(args.path)
    print("Min:", stats["min"])
    print("Max:", stats["max"])
    print("Mean:", stats["mean"])
    print("Std:", stats["std"])
    for p, v in stats["percentiles"].items():
        print(f"{p}th percentile: {v}")


if __name__ == "__main__":  # pragma: no cover
    main()
