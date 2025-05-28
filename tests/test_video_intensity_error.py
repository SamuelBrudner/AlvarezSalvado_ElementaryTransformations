import os
import subprocess
import tempfile
import sys
import importlib
import types
from pathlib import Path
import builtins
import glob
import shutil

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def test_error_message_contains_hint(monkeypatch, tmp_path):
    # Provide fake numpy/scipy so the module can be imported
    monkeypatch.setitem(sys.modules, 'numpy', types.SimpleNamespace(asarray=lambda x: x))
    fake_scipy_io = types.SimpleNamespace(loadmat=lambda p: {'all_intensities': []})
    monkeypatch.setitem(sys.modules, 'scipy', types.SimpleNamespace(io=fake_scipy_io))
    monkeypatch.setitem(sys.modules, 'scipy.io', fake_scipy_io)

    vi = importlib.reload(importlib.import_module('Code.video_intensity'))
    func = vi.get_intensities_from_video_via_matlab

    captured = {}
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault('delete', False)
        fh = orig_ntf(*args, **kwargs)
        captured['path'] = fh.name
        return fh

    def fake_run(cmd, capture_output, text):
        with open(captured['path']) as fh:
            captured['contents'] = fh.read()
        return subprocess.CompletedProcess(cmd, 1, stdout='', stderr='Configuration file not found')

    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', fake_ntf)
    monkeypatch.setattr(subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError) as exc:
        func(
            'disp("hi")',
            'matlab',
            work_dir='.',
            orig_script_path='/path/to/script.m',
        )

    msg = str(exc.value)
    assert '/path/to/script.m' in msg
    assert 'orig_script_dir' in msg

    assert "orig_script_path = '/path/to/script.m';" in captured['contents']
    assert 'orig_script_dir = fileparts(orig_script_path);' in captured['contents']


def test_matlab_exec_from_project_paths(tmp_path, monkeypatch):
    yaml_path = tmp_path / "project_paths.yaml"
    yaml_path.write_text('matlab:\n  executable: "/tmp/fake_matlab"\n')

    orig_open = builtins.open

    def fake_open(path, *args, **kwargs):
        path_str = os.fspath(path)
        if path_str.endswith("project_paths.yaml"):
            return orig_open(yaml_path, *args, **kwargs)
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    orig_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **kw: yaml_path.read_text()
        if str(self).endswith("project_paths.yaml")
        else orig_read_text(self, *a, **kw),
    )

    monkeypatch.setattr(glob, "glob", lambda pattern: [])
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.setattr(os.path, "isfile", lambda p: p == "/tmp/fake_matlab")
    monkeypatch.setattr(os, "access", lambda p, m: p == "/tmp/fake_matlab")

    vi = importlib.reload(importlib.import_module("Code.video_intensity"))
    assert vi.find_matlab_executable() == "/tmp/fake_matlab"

