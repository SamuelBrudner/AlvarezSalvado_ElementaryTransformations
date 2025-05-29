import importlib
import os
import sys
import types
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeArray(list):
    def __init__(self, data, dtype=None):
        super().__init__(float(x) for x in data)

    @property
    def size(self):
        return len(self)

    def mean(self):
        return sum(self) / len(self) if self else float("nan")

    def min(self):
        return min(self) if self else float("nan")

    def max(self):
        return max(self) if self else float("nan")


def asarray(data, dtype=None):
    return FakeArray(data)


def median(arr):
    arr = sorted(arr)
    n = len(arr)
    if n == 0:
        return float("nan")
    if n % 2:
        return arr[n // 2]
    return (arr[n // 2 - 1] + arr[n // 2]) / 2


def percentile(arr, q):
    arr = sorted(arr)
    if not arr:
        return float("nan")
    idx = int(round(q / 100 * (len(arr) - 1)))
    return arr[idx]


@pytest.fixture()
def cis(monkeypatch):
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.asarray = asarray
    fake_numpy.array = asarray
    fake_numpy.median = median
    fake_numpy.percentile = percentile
    fake_numpy.isscalar = lambda x: isinstance(x, (int, float))

    fake_h5py = types.ModuleType("h5py")
    fake_scipy = types.ModuleType("scipy")
    fake_scipy_io = types.ModuleType("scipy.io")
    fake_scipy_io.loadmat = lambda *_: {}
    fake_scipy.io = fake_scipy_io
    fake_loguru = types.ModuleType("loguru")
    fake_loguru.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda *a, **k: {}

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)
    monkeypatch.setitem(sys.modules, "loguru", fake_loguru)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    module = importlib.import_module("Code.compare_intensity_stats")
    importlib.reload(module)
    return module


def test_length_mismatch_processed(cis, monkeypatch):
    """Intensity vectors of different lengths should still be processed."""
    arr_a = [1.0, 2.0, 3.0]
    arr_b = [4.0, 5.0]

    def fake_load(path, *_, **__):
        return arr_a if path == "path_a" else arr_b

    monkeypatch.setattr(cis, "load_intensities", fake_load)

    results = cis.compare_intensity_stats([
        ("A", "path_a", None),
        ("B", "path_b", None),
    ])
    assert len(results) == 2
    assert results[0][0] == "A"
    assert results[1][0] == "B"


def test_matching_lengths_work(cis, monkeypatch):
    """Matching length vectors should work as before."""
    arr = [1.0, 2.0, 3.0]

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


def test_cli_mismatch_no_flag(cis, monkeypatch, capsys):
    """CLI should succeed even without --allow-mismatch."""
    arr_a = [1.0, 2.0, 3.0]
    arr_b = [4.0, 5.0]

    def fake_load(path, *_, **__):
        return arr_a if path == "path_a" else arr_b

    monkeypatch.setattr(cis, "load_intensities", fake_load)

    cis.main([
        "A",
        "path_a",
        "B",
        "path_b",
    ])

    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[0].startswith("identifier")
    assert out_lines[1].split("\t")[0] == "A"
    assert out_lines[2].split("\t")[0] == "B"
