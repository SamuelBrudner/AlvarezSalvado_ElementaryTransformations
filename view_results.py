import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PLUME_FILE = '10302017_10cms_bounded.hdf5'


def analyze_results(out: Any) -> float | None:
    """Return success rate from results and log it."""
    success = getattr(out, 'successrate', None)
    if success is None:
        logger.info('No success rate available')
        return None

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy may not be installed
        np = None  # type: ignore

    if np is not None and isinstance(success, np.ndarray):
        rate = float(success.mean()) if success.size > 1 else float(success.item())
    elif not isinstance(success, (float, int)) and hasattr(success, '__iter__'):
        rate = float(list(success)[0])
    else:
        rate = float(success)

    logger.info('Success rate: %.1f%%', rate * 100)
    return rate


if __name__ == '__main__':  # pragma: no cover - manual use
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description='Analyze navigation results')
    parser.add_argument('success', type=float, nargs='?', help='Success rate value')
    args = parser.parse_args()
    dummy = type('Out', (), {'successrate': args.success})
    analyze_results(dummy)
