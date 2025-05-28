import importlib.util
import sys
from pathlib import Path
import types
import pytest


@pytest.fixture()
def module(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    fake_cis = types.SimpleNamespace(
        compare_intensity_stats=lambda *a, **k: [("X", {"mean": 0.0})],
        format_table=lambda *a, **k: "table",
    )
    fake_code = types.SimpleNamespace(compare_intensity_stats=fake_cis)
    monkeypatch.setitem(sys.modules, "Code", fake_code)

    spec = importlib.util.spec_from_file_location(
        "run_intensity_batch", repo_root / "scripts" / "run_intensity_batch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    yield mod
    sys.path.remove(str(repo_root))


def test_matlab_exec_from_yaml(tmp_path, monkeypatch, module):
    cfg = tmp_path / "project_paths.yaml"
    cfg.write_text("matlab:\n  executable: /opt/matlab\n")

    captured = {}

    def fake_compare(sources, matlab_exec_path="matlab"):
        captured["exec"] = matlab_exec_path
        return [("A", {"mean": 1.0})]

    monkeypatch.setattr(module.cis, "compare_intensity_stats", fake_compare)
    monkeypatch.setattr(module.cis, "format_table", lambda r, diff=None: "table")

    module.main([str(tmp_path / "a.h5"), str(tmp_path / "b.m"), "--config", str(cfg)])

    assert captured["exec"] == "/opt/matlab"


def test_command_line_overrides_yaml(tmp_path, monkeypatch, module):
    cfg = tmp_path / "project_paths.yaml"
    cfg.write_text("matlab:\n  executable: /opt/matlab\n")

    captured = {}

    def fake_compare(sources, matlab_exec_path="matlab"):
        captured["exec"] = matlab_exec_path
        return [("A", {"mean": 1.0})]

    monkeypatch.setattr(module.cis, "compare_intensity_stats", fake_compare)
    monkeypatch.setattr(module.cis, "format_table", lambda r, diff=None: "table")

    module.main([
        str(tmp_path / "a.h5"),
        str(tmp_path / "b.m"),
        "--config",
        str(cfg),
        "--matlab_exec",
        "/custom/matlab",
    ])

    assert captured["exec"] == "/custom/matlab"
