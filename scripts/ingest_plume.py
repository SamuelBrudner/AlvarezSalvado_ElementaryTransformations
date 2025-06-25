#!/usr/bin/env python3
"""Add a new plume configuration and update pipeline configuration."""
import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover - environment without h5py
    h5py = None

# Enhanced logging with loguru per Section 3.2 requirements
from loguru import logger

# Configure structured logging to logs/ directory with timestamp-based files
# Remove default handler and add custom handler for centralized logging
logger.remove()
logger.add(
    "../logs/ingest_plume_{time:YYYY-MM-DD_HH-mm-ss}.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    enqueue=True,  # Thread-safe logging
)

# Also keep console output for immediate feedback
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="{message}",
    colorize=True,
)

DEFAULT_DATASET = "/dataset2"


def build_config(hdf5_path: Path, mm_per_pixel: float, fps: int, dataset: str):
    """Return configuration dictionary for the plume."""
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 files")
    
    logger.info("Reading HDF5 file: {}", hdf5_path)
    logger.debug("Dataset: {}, mm_per_pixel: {}, fps: {}", dataset, mm_per_pixel, fps)
    
    try:
        with h5py.File(hdf5_path, "r") as h5:
            if dataset not in h5:
                logger.error("Dataset '{}' not found in HDF5 file. Available datasets: {}", 
                           dataset, list(h5.keys()))
                raise KeyError(f"Dataset '{dataset}' not found in HDF5 file")
            
            dset = h5[dataset]
            width, height, frames = dset.shape
            logger.debug("HDF5 dimensions: width={}, height={}, frames={}", width, height, frames)
            
    except Exception as e:
        logger.error("Failed to read HDF5 file {}: {}", hdf5_path, str(e))
        raise
    
    config = {
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
    
    logger.success("Built configuration for plume '{}' with {} frames", hdf5_path.stem, frames)
    return config


def update_paths(paths_file: Path, hdf5_path: Path, config_path: Path) -> None:
    """Update plume paths in the given paths.json file."""
    logger.info("Updating paths file: {}", paths_file)
    
    if not paths_file.exists():
        logger.warning("Paths file {} not found, skipping paths update", paths_file)
        return
    
    try:
        data = json.loads(paths_file.read_text())
        logger.debug("Loaded existing paths configuration with {} keys", len(data))
        
        data["plume_file"] = str(hdf5_path)
        data["plume_config"] = str(config_path)
        data["paths_generated"] = datetime.utcnow().isoformat() + "Z"
        
        paths_file.write_text(json.dumps(data, indent=2))
        logger.success("Updated paths file: {}", paths_file)
        
    except Exception as e:
        logger.error("Failed to update paths file {}: {}", paths_file, str(e))
        raise


def update_pipeline_config(pipeline_file: Path, plume_id: str) -> None:
    """Append plume_id to pipeline configuration if not already present."""
    logger.info("Updating pipeline configuration: {}", pipeline_file)
    
    try:
        if not pipeline_file.exists():
            logger.info("Pipeline configuration file does not exist, creating new one")
            data = {"plumes": []}
        else:
            data = json.loads(pipeline_file.read_text())
            logger.debug("Loaded existing pipeline configuration with {} plumes", 
                        len(data.get("plumes", [])))
            
            if "plumes" not in data:
                data["plumes"] = []
        
        if plume_id not in data["plumes"]:
            data["plumes"].append(plume_id)
            pipeline_file.parent.mkdir(parents=True, exist_ok=True)
            pipeline_file.write_text(json.dumps(data, indent=2))
            logger.success("Added plume '{}' to pipeline configuration: {}", plume_id, pipeline_file)
        else:
            logger.info("Plume '{}' already exists in pipeline configuration", plume_id)
            
    except Exception as e:
        logger.error("Failed to update pipeline configuration {}: {}", pipeline_file, str(e))
        raise


def main() -> None:
    """Main entry point for plume ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest a new plume dataset")
    parser.add_argument("plume_id", help="Identifier for the plume")
    parser.add_argument("hdf5_file", help="Path to plume HDF5 file")
    parser.add_argument("--mm-per-pixel", type=float, required=True,
                       help="Spatial resolution in millimeters per pixel")
    parser.add_argument("--fps", type=int, required=True,
                       help="Frame rate in frames per second")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                       help=f"HDF5 dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--config-dir", default="../configs/plumes",
                       help="Directory for plume configuration files")
    parser.add_argument("--paths-file", default="../configs/paths.json",
                       help="Paths configuration file to update")
    parser.add_argument(
        "--pipeline-config",
        default="../configs/pipeline/pipeline_plumes.json",
        help="Pipeline configuration file to update",
    )
    args = parser.parse_args()

    logger.info("Starting plume ingestion for plume_id: {}", args.plume_id)
    logger.info("Input HDF5 file: {}", args.hdf5_file)
    logger.info("Configuration: mm_per_pixel={}, fps={}, dataset={}", 
               args.mm_per_pixel, args.fps, args.dataset)

    try:
        # Validate input file exists
        hdf5_path = Path(args.hdf5_file)
        if not hdf5_path.exists():
            logger.error("HDF5 file does not exist: {}", hdf5_path)
            raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")
        
        # Create output directory
        config_dir = Path(args.config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using configuration directory: {}", config_dir.resolve())

        # Build and write plume configuration
        config = build_config(hdf5_path, args.mm_per_pixel, args.fps, args.dataset)
        config["plume_id"] = args.plume_id
        
        output_path = config_dir / f"{args.plume_id}.json"
        output_path.write_text(json.dumps(config, indent=2))
        logger.success("Wrote plume configuration to: {}", output_path.resolve())

        # Update related configuration files
        update_paths(Path(args.paths_file), hdf5_path, output_path)
        update_pipeline_config(Path(args.pipeline_config), args.plume_id)
        
        logger.success("Successfully completed plume ingestion for '{}'", args.plume_id)
        
    except Exception as e:
        logger.error("Plume ingestion failed: {}", str(e))
        logger.debug("Full error details", exc_info=True)
        raise


if __name__ == "__main__":
    main()