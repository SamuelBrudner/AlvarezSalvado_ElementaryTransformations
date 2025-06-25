import os
import json
from loguru import logger

# Configure loguru logging to route to logs/ directory with timestamp-based files
# Remove default handler to avoid duplicate console output
logger.remove()

# Add structured logging to logs/ directory with rotation and retention
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logger.add(
    os.path.join(log_dir, "plume_config_{time:YYYY-MM-DD}.log"),
    rotation="daily",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level="DEBUG"
)

# Add console handler for interactive usage
logger.add(
    lambda msg: print(msg, end=''),
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    level="INFO"
)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'configs',
    'navigation_model',
    'navigation_model_default.json',
)


def get_config_path() -> str:
    """Return the path to the configuration file."""
    return os.environ.get('PLUME_CONFIG', DEFAULT_CONFIG_PATH)


def get_plume_file() -> str:
    """Return the full path to the plume file specified in the configuration."""
    config_path = get_config_path()
    logger.debug("Loading config from {}", config_path)
    if not os.path.exists(config_path):
        logger.warning("Config file {} not found. Using default filename.", config_path)
        return '10302017_10cms_bounded.hdf5'
    try:
        with open(config_path) as f:
            data = json.load(f)
        plume_file = data.get('plume_file', '10302017_10cms_bounded.hdf5')
        plume_path = data.get('plume_path', '')
        if plume_path:
            full_path = os.path.join(plume_path, plume_file)
            logger.debug("Resolved plume file path: {}", full_path)
            return full_path
        logger.debug("Using plume file without path: {}", plume_file)
        return plume_file
    except Exception as exc:
        logger.error("Invalid config file {}: {}", config_path, exc)
        return '10302017_10cms_bounded.hdf5'