import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import types
import logging
from view_results import analyze_results


def test_analyze_results_handles_iterable(caplog):
    out = types.SimpleNamespace(successrate=[0.5])
    with caplog.at_level(logging.INFO):
        rate = analyze_results(out)
    assert rate == 0.5
    assert any('Success rate:' in rec.getMessage() for rec in caplog.records)
