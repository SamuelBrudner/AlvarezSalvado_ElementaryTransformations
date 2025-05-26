import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.data_aggregation import aggregate_metrics


def test_simple_aggregation():
    records = [
        {'plume_type': 'gaussian', 'sensing_mode': 'bilateral', 'metric': 1.0},
        {'plume_type': 'gaussian', 'sensing_mode': 'bilateral', 'metric': 3.0},
        {'plume_type': 'smoke', 'sensing_mode': 'unilateral', 'metric': 2.0},
    ]
    cfg = {
        'aggregation_options': {
            'group_by_keys': ['plume_type', 'sensing_mode'],
            'statistics_to_compute': ['mean', 'std', 'sem', 'count', 'median', 'min', 'max'],
        }
    }

    result = aggregate_metrics(records, cfg)
    gb = result[('gaussian', 'bilateral')]['metric']
    assert gb['count'] == 2
    assert gb['mean'] == pytest.approx(2.0)
    assert gb['min'] == 1.0
    assert gb['max'] == 3.0
