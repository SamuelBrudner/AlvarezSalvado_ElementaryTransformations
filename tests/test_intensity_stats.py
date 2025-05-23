import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.intensity_stats import calculate_intensity_stats_dict


def test_intensity_stats_basic():
    intensities = np.array([0, 0.02, 0.5, 1.0, 2.0, 0.005])
    stats = calculate_intensity_stats_dict(intensities, min_threshold=0.01)
    filtered = intensities[intensities >= 0.01]

    assert stats["num_pixels_total"] == len(intensities)
    assert stats["num_pixels_analyzed_post_threshold"] == len(filtered)
    assert stats["mean"] == pytest.approx(filtered.mean())
    assert stats["median"] == pytest.approx(np.median(filtered))
    assert stats["std"] == pytest.approx(filtered.std())
    assert stats["min"] == pytest.approx(filtered.min())
    assert stats["max"] == pytest.approx(filtered.max())
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
        assert stats["percentiles"][p] == pytest.approx(np.percentile(filtered, p))


def test_intensity_stats_empty():
    intensities = np.array([0.001, 0.005])
    stats = calculate_intensity_stats_dict(intensities, min_threshold=0.01)

    assert stats["num_pixels_total"] == len(intensities)
    assert stats["num_pixels_analyzed_post_threshold"] == 0
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["median"])
    assert np.isnan(stats["std"])
    assert np.isnan(stats["min"])
    assert np.isnan(stats["max"])
    assert stats["percentiles"] == {}
