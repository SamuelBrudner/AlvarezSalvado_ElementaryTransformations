import inspect
import re

from Code.comparative_analysis import run_statistical_tests


def test_run_statistical_tests_docstring_details():
    doc = inspect.getdoc(run_statistical_tests)
    assert doc is not None
    doc = doc.lower()
    assert "cfg['statistical_analysis']" in doc
    assert 'metric_name' in doc
    assert 'groups_to_compare' in doc
    expected = (
        'returns a list of result dictionaries containing '
        'metrics, groups, t-statistics and p-values.'
    )
    normalized = re.sub(' +', ' ', doc.replace('\n', ' '))
    assert expected in normalized
