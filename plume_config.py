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
DEFAULT_PLUME_FILE = '10302017_10cms_bounded.hdf5'


def get_config_path() -> str:
    """Return the path to the configuration file."""
    path = os.environ.get('PLUME_CONFIG', DEFAULT_CONFIG_PATH)
    if path == DEFAULT_CONFIG_PATH:
        logger.debug('PLUME_CONFIG not set; using default %s', path)
    else:
        logger.debug('Using config path from PLUME_CONFIG: %s', path)
    return path


def get_plume_file() -> str:
    """Return the HDF5 plume filename from the configuration."""
    config_path = get_config_path()
    logger.debug('Loading config from %s', config_path)
    if not os.path.exists(config_path):
        logger.warning('Config file %s not found. Using default filename.', config_path)
        logger.debug('Default plume file: %s', DEFAULT_PLUME_FILE)
        return DEFAULT_PLUME_FILE
    try:
        with open(config_path) as f:
            data = json.load(f)
        plume_file = data.get('plume_file', DEFAULT_PLUME_FILE)
        logger.debug('Plume file from config: %s', plume_file)
        return plume_file
    except Exception as exc:
        logger.error('Invalid config file %s: %s', config_path, exc)
        logger.debug('Default plume file: %s', DEFAULT_PLUME_FILE)
        return DEFAULT_PLUME_FILE


def load_config() -> dict:
    """Return configuration dict with defaults."""
    config_path = get_config_path()
    logger.debug('Loading full config from %s', config_path)
    if not os.path.exists(config_path):
        logger.warning('Config file %s not found. Using defaults.', config_path)
        return {
            'plume_file': DEFAULT_PLUME_FILE,
            'mm_per_pixel': 0.74,
            'frame_rate_hz': 15,
        }
    try:
        with open(config_path) as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - parse errors
        logger.error('Invalid config %s: %s', config_path, exc)
        return {
            'plume_file': DEFAULT_PLUME_FILE,
            'mm_per_pixel': 0.74,
            'frame_rate_hz': 15,
        }

    return {
        'plume_file': data.get('plume_file', DEFAULT_PLUME_FILE),
        'mm_per_pixel': float(data.get('mm_per_pixel', 0.74)),
        'frame_rate_hz': float(data.get('frame_rate_hz', 15)),
    }
