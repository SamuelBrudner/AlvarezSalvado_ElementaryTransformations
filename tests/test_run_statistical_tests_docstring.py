import inspect

from Code.comparative_analysis import run_statistical_tests


def test_run_statistical_tests_docstring_details():
    doc = inspect.getdoc(run_statistical_tests)
    assert doc is not None
    doc = doc.lower()
    assert "cfg['statistical_analysis']" in doc
    assert 'metric_name' in doc
    assert 'groups_to_compare' in doc
    assert 'list of result dictionaries' in doc
