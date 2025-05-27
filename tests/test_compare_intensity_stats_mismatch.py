import os
import sys
import pytest

np = pytest.importorskip("numpy")

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code import compare_intensity_stats as cis


def test_length_mismatch_raises(monkeypatch):
    """Test that ValueError is raised when intensity vectors have different lengths."""
    arr_a = np.array([1.0, 2.0, 3.0])
    arr_b = np.array([4.0, 5.0])

    def fake_load(path, *_, **__):
        return arr_a if path == "path_a" else arr_b

    monkeypatch.setattr(cis, "load_intensities", fake_load)

    with pytest.raises(ValueError, match=r"Expected intensities of length 3, got 2"):
        cis.compare_intensity_stats([
            ("A", "path_a", None),
            ("B", "path_b", None),
        ])


def test_matching_lengths_work(monkeypatch):
    """Test that matching length vectors are processed without errors."""
    arr = np.array([1.0, 2.0, 3.0])

    def fake_load(path, *_, **__):
        return arr

    monkeypatch.setattr(cis, "load_intensities", fake_load)

    results = cis.compare_intensity_stats([
        ("A", "path_a", None),
        ("B", "path_b", None),
    ])

    assert len(results) == 2
    assert results[0][0] == "A"
    assert results[1][0] == "B"
