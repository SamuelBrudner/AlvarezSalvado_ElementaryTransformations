import importlib
import os
import sys
import types

import pytest
import csv

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
    fake_scipy_io.loadmat = lambda _: {"all_intensities": [1.0, 2.0]}
    fake_scipy.io = fake_scipy_io

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", fake_scipy_io)

    module = importlib.import_module("Code.compare_intensity_stats")
    importlib.reload(module)
    return module


def simple_stats(intensities):
    data = list(float(x) for x in intensities)
    return {
        "mean": sum(data) / len(data),
        "median": median(data),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "min": min(data),
        "max": max(data),
        "count": len(data),
    }


def test_compute_differences(cis):
    results = [
        (
            "A",
            {
                "mean": 1.0,
                "median": 2.0,
                "p95": 3.0,
                "p99": 4.0,
                "min": 0.0,
                "max": 5.0,
                "count": 3,
            },
        ),
        (
            "B",
            {
                "mean": 3.0,
                "median": 5.0,
                "p95": 6.0,
                "p99": 7.0,
                "min": -2.0,
                "max": 8.0,
                "count": 1,
            },
        ),
    ]
    diff = cis.compute_differences(results)
    assert diff["delta_mean"] == pytest.approx(-2.0)
    assert diff["delta_median"] == pytest.approx(-3.0)
    assert diff["delta_p95"] == pytest.approx(-3.0)
    assert diff["delta_p99"] == pytest.approx(-3.0)
    assert diff["delta_min"] == pytest.approx(2.0)
    assert diff["delta_max"] == pytest.approx(-3.0)
    assert diff["delta_count"] == pytest.approx(2)


def test_diff_option_prints_table(cis, monkeypatch, capsys):
    monkeypatch.setattr(cis, "load_intensities", lambda *a, **k: [1.0, 2.0])
    monkeypatch.setattr(cis, "calculate_intensity_stats_dict", simple_stats)
    cis.main(["A", "path1", "B", "path2", "--diff"])
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[-1].startswith('DIFF')

def test_diff_option_writes_csv(cis, monkeypatch, tmp_path):
    arr_a = [1.0, 2.0]
    arr_b = [3.0, 5.0]

    def fake_load(path, *_, **__):
        return arr_a if path == 'path1' else arr_b

    monkeypatch.setattr(cis, 'load_intensities', fake_load)
    monkeypatch.setattr(cis, 'calculate_intensity_stats_dict', simple_stats)

    csv_path = tmp_path / 'stats.csv'
    cis.main(['A', 'path1', 'B', 'path2', '--diff', '--csv', str(csv_path)])

    with csv_path.open(newline='') as f:
        rows = list(csv.reader(f))

    assert rows[-1][0] == 'DIFF'
    assert float(rows[-1][1]) == pytest.approx(-2.5)
    assert float(rows[-1][2]) == pytest.approx(-2.5)
    diff_row = out_lines[-1].split("\t")
    assert diff_row[0] == "DIFF"
    assert len(diff_row) == 8
    # Ensure all delta fields are printed
    assert all(cell for cell in diff_row[1:])
