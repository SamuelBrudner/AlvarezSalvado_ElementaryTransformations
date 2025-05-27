import os
import subprocess
import tempfile
import sys
import importlib
import types

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
