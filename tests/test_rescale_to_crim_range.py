import pytest

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - skip tests without numpy
    np = None


@pytest.mark.skipif(np is None, reason="numpy not available")
def test_rescale_to_crim_range_basic():
    import Code.plume_utils as pu
    arr = np.array([0.0, 1.0], dtype=float)
    scaled = pu.rescale_to_crim_range(arr)
    stats = pu.get_intensity_stats()
    assert pytest.approx(stats["CRIM"]["min"], rel=1e-6) == float(scaled.min())
    assert pytest.approx(stats["CRIM"]["max"], rel=1e-6) == float(scaled.max())
