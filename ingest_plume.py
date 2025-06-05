#!/usr/bin/env python3
"""Add a new plume configuration and update pipeline configuration."""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover - environment without h5py
    h5py = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "/dataset2"


def build_config(hdf5_path: Path, mm_per_pixel: float, fps: int, dataset: str):
    """Return configuration dictionary for the plume."""
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files")
    with h5py.File(hdf5_path, "r") as h5:
        dset = h5[dataset]
        width, height, frames = dset.shape
    return {
        "plume_id": hdf5_path.stem,
        "data_path": {
            "path": str(hdf5_path),
            "dataset_name": dataset,
        },
        "spatial": {
            "resolution": {"width": int(width), "height": int(height)},
            "mm_per_pixel": mm_per_pixel,
        },
        "temporal": {"frame_rate": fps, "total_frames": int(frames)},
    }


def update_paths(paths_file: Path, hdf5_path: Path, config_path: Path) -> None:
    """Update plume paths in the given paths.json file."""
    if not paths_file.exists():
        logger.warning("paths file %s not found", paths_file)
        return
    data = json.loads(paths_file.read_text())
    data["plume_file"] = str(hdf5_path)
    data["plume_config"] = str(config_path)
    data["paths_generated"] = datetime.utcnow().isoformat() + "Z"
    paths_file.write_text(json.dumps(data, indent=2))
    logger.info("Updated %s", paths_file)


def update_pipeline_config(pipeline_file: Path, plume_id: str) -> None:
    """Append plume_id to pipeline configuration if not already present."""
    if not pipeline_file.exists():
        data = {"plumes": []}
    else:
        data = json.loads(pipeline_file.read_text())
        if "plumes" not in data:
            data["plumes"] = []
    if plume_id not in data["plumes"]:
        data["plumes"].append(plume_id)
        pipeline_file.parent.mkdir(parents=True, exist_ok=True)
        pipeline_file.write_text(json.dumps(data, indent=2))
        logger.info("Updated %s", pipeline_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a new plume dataset")
    parser.add_argument("plume_id", help="Identifier for the plume")
    parser.add_argument("hdf5_file", help="Path to plume HDF5 file")
    parser.add_argument("--mm-per-pixel", type=float, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-dir", default="configs/plumes")
    parser.add_argument("--paths-file", default="configs/paths.json")
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline/pipeline_plumes.json",
        help="Pipeline configuration file to update",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_file)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(hdf5_path, args.mm_per_pixel, args.fps, args.dataset)
    config["plume_id"] = args.plume_id
    output_path = config_dir / f"{args.plume_id}.json"
    output_path.write_text(json.dumps(config, indent=2))
    logger.info("Wrote %s", output_path)

    update_paths(Path(args.paths_file), hdf5_path, output_path)
    update_pipeline_config(Path(args.pipeline_config), args.plume_id)


if __name__ == "__main__":
    main()
