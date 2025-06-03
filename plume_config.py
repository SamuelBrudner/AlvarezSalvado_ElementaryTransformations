import os
import json
import logging

logger = logging.getLogger(__name__)

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
    logger.debug("Loading config from %s", config_path)
    if not os.path.exists(config_path):
        logger.warning("Config file %s not found. Using default filename.", config_path)
        return '10302017_10cms_bounded.hdf5'
    try:
        with open(config_path) as f:
            data = json.load(f)
        plume_file = data.get('plume_file', '10302017_10cms_bounded.hdf5')
        plume_path = data.get('plume_path', '')
        if plume_path:
            full_path = os.path.join(plume_path, plume_file)
            logger.debug("Resolved plume file path: %s", full_path)
            return full_path
        logger.debug("Using plume file without path: %s", plume_file)
        return plume_file
    except Exception as exc:
        logger.error("Invalid config file %s: %s", config_path, exc)
        return '10302017_10cms_bounded.hdf5'
