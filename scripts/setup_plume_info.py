#!/usr/bin/env python3
"""Interactive setup for plume configuration files.

This script requests the user to provide the file paths for the smoke and
Crimaldi plume datasets. Configuration files are then generated in the
specified directory (default: configs/plumes) with hard coded frame rates
and pixel sizes.

Enhanced with structured logging using loguru, with output routed to the
logs/ directory for centralized monitoring and debugging support.
"""
import argparse
import json
import sys
from pathlib import Path

# Import loguru for structured logging per Section 3.2 requirements
from loguru import logger


SMOKE_FPS = 60
SMOKE_MM_PER_PIXEL = 0.153
CRIMALDI_FPS = 15
CRIMALDI_MM_PER_PIXEL = 0.74
DATASET_NAME = "/dataset2"


def setup_logging() -> None:
    """Configure structured logging to logs/ directory with timestamp-based filenames.
    
    Per Section 7.2 verbose logging requirements, structured logging is routed
    to logs/ directory with timestamp-based log files for centralized monitoring.
    Maintains Python 3.6+ compatibility as specified in Section 3.2.
    """
    # Remove default handler to avoid duplicate output
    logger.remove()
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler with timestamp-based filename and structured format
    logger.add(
        logs_dir / "setup_plume_info_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="gz"
    )
    
    # Add console handler for immediate feedback (INFO level and above)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    logger.info("Logging system initialized - outputs routed to logs/ directory")


def ask_path(prompt: str) -> str:
    """Prompt the user for a path with enhanced logging.
    
    Args:
        prompt: The prompt message to display to the user
        
    Returns:
        User-provided path string, or empty string on EOF
    """
    logger.debug(f"Prompting user for input: {prompt}")
    try:
        path = input(prompt).strip()
        logger.debug(f"User input received: {'<empty>' if not path else path}")
        return path
    except EOFError:
        logger.warning("EOF received during user input - returning empty string")
        return ""


def write_config(path: Path, plume_file: str, mm_per_pixel: float, fps: int) -> None:
    """Write a simple JSON configuration with comprehensive logging.
    
    Args:
        path: Output path for the configuration file
        plume_file: Path to the plume HDF5 data file
        mm_per_pixel: Millimeters per pixel for the plume dataset
        fps: Frame rate for the plume dataset
    """
    logger.info(f"Writing configuration file: {path}")
    logger.debug(f"Configuration parameters - plume_file: {plume_file}, mm_per_pixel: {mm_per_pixel}, fps: {fps}")
    
    data = {
        "plume_file": plume_file,
        "dataset_name": DATASET_NAME,
        "mm_per_pixel": mm_per_pixel,
        "frame_rate": fps,
    }
    
    try:
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Successfully wrote configuration to {path}")
        logger.debug(f"Configuration content: {json.dumps(data, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to write configuration file {path}: {e}")
        raise


def main() -> None:
    """Main execution function with enhanced error handling and logging."""
    # Initialize structured logging before any other operations
    setup_logging()
    
    logger.info("Starting plume configuration setup")
    logger.debug("Setting up argument parser")
    
    parser = argparse.ArgumentParser(description="Generate plume configuration files")
    parser.add_argument("--smoke-file", help="Path to smoke plume HDF5 file")
    parser.add_argument("--crimaldi-file", help="Path to crimaldi plume HDF5 file")
    parser.add_argument("--config-dir", default="configs/plumes", help="Directory for output configs")
    args = parser.parse_args()

    logger.info(f"Command-line arguments parsed - config_dir: {args.config_dir}")
    logger.debug(f"All arguments: smoke_file={args.smoke_file}, crimaldi_file={args.crimaldi_file}, config_dir={args.config_dir}")

    config_dir = Path(args.config_dir)
    logger.info(f"Creating config directory: {config_dir}")
    
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Config directory created successfully: {config_dir}")
    except Exception as e:
        logger.error(f"Failed to create config directory {config_dir}: {e}")
        sys.exit(1)

    # Get plume file paths from arguments or user input
    logger.info("Collecting plume file paths")
    smoke_file = args.smoke_file or ask_path("Path to smoke plume file: ")
    crimaldi_file = args.crimaldi_file or ask_path("Path to crimaldi plume file: ")

    logger.info(f"Smoke file: {smoke_file}")
    logger.info(f"Crimaldi file: {crimaldi_file}")
    
    if not smoke_file:
        logger.warning("No smoke plume file provided")
    if not crimaldi_file:
        logger.warning("No crimaldi plume file provided")

    # Write configuration files with enhanced logging
    logger.info("Writing plume configuration files")
    
    try:
        if smoke_file:
            smoke_config_path = config_dir / "plumes_smoke_info.json"
            write_config(smoke_config_path, smoke_file, SMOKE_MM_PER_PIXEL, SMOKE_FPS)
        else:
            logger.warning("Skipping smoke configuration - no file path provided")
        
        if crimaldi_file:
            crimaldi_config_path = config_dir / "plumes_crimaldi_info.json"
            write_config(crimaldi_config_path, crimaldi_file, CRIMALDI_MM_PER_PIXEL, CRIMALDI_FPS)
        else:
            logger.warning("Skipping crimaldi configuration - no file path provided")
            
        logger.success("Plume configuration setup completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()