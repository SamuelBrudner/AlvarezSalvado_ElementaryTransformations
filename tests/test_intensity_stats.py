import os
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.intensity_stats import calculate_intensity_stats_dict, main


def test_calculate_intensity_stats_dict():
    intensities = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    stats = calculate_intensity_stats_dict(intensities)
    assert stats["mean"] == pytest.approx(np.mean(intensities))
    assert stats["median"] == pytest.approx(np.median(intensities))
    assert stats["p95"] == pytest.approx(np.percentile(intensities, 95))
    assert stats["p99"] == pytest.approx(np.percentile(intensities, 99))
    assert stats["min"] == pytest.approx(intensities.min())
    assert stats["max"] == pytest.approx(intensities.max())
    assert stats["count"] == intensities.size


def test_main_prints_stats(tmp_path, capsys):
    data = np.arange(10)
    f = tmp_path / "data.txt"
    np.savetxt(f, data)
    main(["plumeA", str(f)])
    out = capsys.readouterr().out
    assert "Plume: plumeA" in out
    assert f"File: {f}" in out
    assert "mean:" in out


def test_main_plot_histogram(monkeypatch, tmp_path):
    data = np.linspace(0, 1, 5)
    f = tmp_path / "data.txt"
    np.savetxt(f, data)
    dummy = types.SimpleNamespace(hist=lambda *a, **k: None,
                                  title=lambda *a, **k: None,
                                  xlabel=lambda *a, **k: None,
                                  ylabel=lambda *a, **k: None,
                                  show=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', dummy)
    main(["plumeB", str(f), "--plot_histogram"])


def test_empty_intensity_stats_returns_nans():
    stats = calculate_intensity_stats_dict(np.array([]))
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["median"])
    assert np.isnan(stats["p95"])
    assert np.isnan(stats["p99"])
    assert np.isnan(stats["min"])
    assert np.isnan(stats["max"])
    assert stats["count"] == 0
