import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_BASE = Path('configs')
SUBDIR = 'navigation_model'
CONFIG_NAME = 'navigation_model_default.json'


def write_plume_config(
    plume_file: str,
    base_dir: Path = CONFIG_BASE,
    subdir: str = SUBDIR,
    config_name: str = CONFIG_NAME,
) -> str:
    """Write plume file path to JSON configuration and return path."""
    config_dir = Path(base_dir) / subdir
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_name
    data = {"plume_file": plume_file}
    with open(config_path, 'w') as fh:
        json.dump(data, fh, indent=2)
    logger.info('Wrote plume configuration to %s', config_path)
    return str(config_path)


def main() -> None:
    """Interactively create a plume configuration file."""
    logging.basicConfig(level=logging.INFO)
    default_dir = Path('data/plumes')
    directory = input(f'Plume directory [{default_dir}]: ').strip() or str(default_dir)
    plume_dir = Path(directory)
    files = sorted(plume_dir.glob('*.hdf5'))
    if files:
        for idx, file in enumerate(files):
            print(f'[{idx}] {file}')
        choice = input('Select plume file by number: ').strip()
        try:
            index = int(choice)
            plume_file = str(files[index])
        except Exception as exc:  # pragma: no cover - user input errors
            logger.error('Invalid selection: %s (%s)', choice, exc)
            return
    else:
        plume_file = input('Enter path to plume file: ').strip()
    write_plume_config(plume_file)


if __name__ == '__main__':  # pragma: no cover - manual execution only
    main()
