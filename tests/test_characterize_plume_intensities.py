# ruff: noqa: E402
import json
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def simple_stats(values):
    values = sorted(values)
    n = len(values)

    def pct(p):
        k = (n - 1) * p / 100
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(values[f])
        return float(values[f] * (c - k) + values[c] * (k - f))

    mean = sum(values) / n if n else float("nan")
    if n % 2 == 1:
        median = float(values[n // 2])
    else:
        median = (values[n // 2 - 1] + values[n // 2]) / 2
    stats = {
        "mean": float(mean),
        "median": float(median),
        "p95": pct(95),
        "p99": pct(99),
        "min": float(values[0]),
        "max": float(values[-1]),
        "count": n,
    }
    return stats


sys.modules["Code.intensity_stats"] = types.SimpleNamespace(
    calculate_intensity_stats_dict=simple_stats
)

from Code.characterize_plume_intensities import process_plume


def test_json_creation_and_update(tmp_path):
    output = tmp_path / "stats.json"

    # first plume
    intensities1 = [1, 2, 3]
    process_plume("p1", intensities1, output)
    data = json.loads(output.read_text())
    assert len(data) == 1
    assert data[0]["plume_id"] == "p1"
    expected1 = simple_stats(intensities1)
    assert data[0]["statistics"] == expected1

    # second plume
    process_plume("p2", [4, 5], output)
    data = json.loads(output.read_text())
    assert len(data) == 2
    ids = {d["plume_id"] for d in data}
    assert ids == {"p1", "p2"}

    # update first plume
    intensities_update = [7]
    process_plume("p1", intensities_update, output)
    data = json.loads(output.read_text())
    assert len(data) == 2
    entry = next(d for d in data if d["plume_id"] == "p1")
    expected_update = simple_stats(intensities_update)
    assert entry["statistics"] == expected_update


def test_corrupted_or_empty_file(tmp_path):
    output = tmp_path / "stats.json"
    output.write_text("{ not valid json }")

    process_plume("p1", [1], output)
    data = json.loads(output.read_text())
    assert len(data) == 1
    assert data[0]["plume_id"] == "p1"


def test_creates_parent_directory(tmp_path):
    output = tmp_path / "missing" / "stats.json"

    process_plume("p1", [1, 2], output)
    assert output.exists()
    data = json.loads(output.read_text())
    assert len(data) == 1
    expected = simple_stats([1, 2])
    assert data[0]["plume_id"] == "p1"
    assert data[0]["statistics"] == expected
