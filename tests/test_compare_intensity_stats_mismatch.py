"""Test case for intensity vector length validation in compare_intensity_stats."""

import importlib
import os
import sys
import types
from typing import Any

import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeArray(list):
    """Minimal numpy array-like for testing."""
    def __init__(self, data, dtype=None):
        super().__init__(float(x) for x in data)

    @property
    def size(self) -> int:
        return len(self)

    def mean(self) -> float:
        return sum(self) / len(self) if self else float("nan")

    def min(self) -> float:
        return min(self) if self else float("nan")
        
    def max(self) -> float:
        return max(self) if self else float("nan")


@pytest.fixture()
def cis(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fixture to provide a test environment for compare_intensity_stats.
    
    Mocks numpy, h5py, and scipy to avoid actual file I/O during testing.
    """
    # Create fake numpy module
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.asarray = lambda x, dtype=None: FakeArray(x)
    fake_numpy.array = lambda x, dtype=None: FakeArray(x)
    fake_numpy.median = lambda x: sorted(x)[len(x)//2] if x else float("nan")
    fake_numpy.percentile = lambda x, q: sorted(x)[int(q/100 * (len(x)-1))] if x else float("nan")
    fake_numpy.isscalar = lambda x: isinstance(x, (int, float))

    # Create minimal mock modules for other imports
    fake_h5py = types.ModuleType("h5py")
    fake_scipy = types.ModuleType("scipy")
    fake_scipy_io = types.ModuleType("scipy.io")
    fake_scipy_io.loadmat = lambda _: {"all_intensities": [1.0, 2.0]}
    fake_scipy.io = fake_scipy_io

    # Apply monkey patches
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)

    # Import and return the module under test
    module = importlib.import_module("Code.compare_intensity_stats")
    importlib.reload(module)
    return module


def test_mismatched_intensity_lengths_raise(cis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ValueError is raised when intensity vectors have different lengths."""
    # Test data with different lengths
    arr_a = [1.0, 2.0, 3.0]  # length 3
    arr_b = [4.0, 5.0]       # length 2

    # Mock load_intensities to return our test data
    def fake_load(path: str, *_: Any, **__: Any) -> FakeArray:
        return FakeArray(arr_a if path == "a" else arr_b)

    monkeypatch.setattr(cis, "load_intensities", fake_load)

    # Mock calculate_intensity_stats_dict to return simple stats
    def simple_stats(intensities: FakeArray) -> dict[str, float]:
        data = [float(x) for x in intensities]
        return {
            "mean": sum(data) / len(data) if data else float("nan"),
            "median": sorted(data)[len(data)//2] if data else float("nan"),
            "p95": sorted(data)[int(0.95 * (len(data)-1))] if data else float("nan"),
            "p99": sorted(data)[int(0.99 * (len(data)-1))] if data else float("nan"),
            "min": min(data) if data else float("nan"),
            "max": max(data) if data else float("nan"),
            "count": len(data)
        }

    monkeypatch.setattr(cis, "calculate_intensity_stats_dict", simple_stats)

    # Test that ValueError is raised with the correct message
    with pytest.raises(ValueError, match="length mismatch:"):
        cis.compare_intensity_stats([
            ("TestA", "a", None),
            ("TestB", "b", None)
        ])


def test_matching_lengths_work(cis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that matching length vectors are processed without errors."""
    # Test data with same lengths
    arr = [1.0, 2.0, 3.0]

    def fake_load(path: str, *_: Any, **__: Any) -> FakeArray:
        return FakeArray(arr)

    monkeypatch.setattr(cis, "load_intensities", fake_load)
    
    # This should not raise an exception
    results = cis.compare_intensity_stats([
        ("Test1", "path1", None),
        ("Test2", "path2", None)
    ])
    
    # Verify we got results for both inputs
    assert len(results) == 2
    assert results[0][0] == "Test1"
    assert results[1][0] == "Test2"
