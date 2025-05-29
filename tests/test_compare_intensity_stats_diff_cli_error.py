import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_compare_intensity_stats_diff import cis as cis_fixture


def test_diff_with_three_datasets_errors(cis_fixture, monkeypatch, capsys):
    cis = cis_fixture
    monkeypatch.setattr(cis, "load_intensities", lambda *a, **k: [1.0])
    with pytest.raises(SystemExit) as excinfo:
        cis.main([
            "A",
            "path1",
            "B",
            "path2",
            "C",
            "path3",
            "--diff",
        ])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err.lower()
    assert "two datasets" in err
