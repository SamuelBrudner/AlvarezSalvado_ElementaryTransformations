#!/usr/bin/env python3
"""Utility to inspect the structure of an HDF5 file."""

import sys

import h5py


def print_h5_structure(filename):
    """Print the structure of an HDF5 file."""
    try:
        with h5py.File(filename, "r") as f:
            print(f"File: {filename}")
            print("Datasets and groups:")

            def print_attrs(name, obj):
                print(f"  {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Shape: {obj.shape}")
                    print(f"    Dtype: {obj.dtype}")
                if obj.attrs:
                    print("    Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"      {key}: {val}")

            f.visititems(print_attrs)

    except Exception as e:
        print(f"Error reading {filename}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <hdf5_file>")
        sys.exit(1)

    print_h5_structure(sys.argv[1])
