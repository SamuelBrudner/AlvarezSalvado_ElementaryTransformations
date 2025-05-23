import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.characterize_plume_intensities import process_plume


def test_json_creation_and_update(tmp_path):
    output = tmp_path / "stats.json"

    # first plume
    process_plume("p1", [1, 2, 3], output)
    data = json.loads(output.read_text())
    assert len(data) == 1
    assert data[0]["plume_id"] == "p1"

    # second plume
    process_plume("p2", [4, 5], output)
    data = json.loads(output.read_text())
    assert len(data) == 2
    ids = {d["plume_id"] for d in data}
    assert ids == {"p1", "p2"}

    # update first plume
    process_plume("p1", [7], output)
    data = json.loads(output.read_text())
    assert len(data) == 2
    entry = next(d for d in data if d["plume_id"] == "p1")
    assert entry["statistics"]["mean"] == 7


def test_corrupted_or_empty_file(tmp_path):
    output = tmp_path / "stats.json"
    output.write_text("{ not valid json }")

    process_plume("p1", [1], output)
    data = json.loads(output.read_text())
    assert len(data) == 1
    assert data[0]["plume_id"] == "p1"
