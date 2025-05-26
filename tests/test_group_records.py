import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code.comparative_analysis import _group_records


def test_group_records_missing_key():
    records = [
        {"a": 1, "b": 2},
        {"a": 1},
    ]
    grouped = _group_records(records, ["a", "b"])
    assert grouped[(1, 2)][0] == records[0]
    assert grouped[(1, None)][0] == records[1]
