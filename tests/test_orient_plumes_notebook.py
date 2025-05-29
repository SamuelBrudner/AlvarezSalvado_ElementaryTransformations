import json
import sys
import types
import io
from pathlib import Path

NOTEBOOK_PATH = Path('notebooks/orient_plumes.ipynb')


def test_notebook_exists():
    assert NOTEBOOK_PATH.exists(), 'Notebook not found'


def test_notebook_imports():
    data = json.loads(NOTEBOOK_PATH.read_text())
    src = ''.join(''.join(cell.get('source', [])) for cell in data.get('cells', []))
    assert 'h5py' in src, 'Notebook must import h5py'
    assert 'imageio' in src, 'Notebook must import imageio'
    assert 'yaml' in src, 'Notebook must import yaml'


def test_yaml_cell_exists():
    data = json.loads(NOTEBOOK_PATH.read_text())
    assert len(data.get('cells', [])) >= 4, 'Expected at least four cells'


def test_notebook_uses_yaml_paths(tmp_path, monkeypatch):
    nb = json.loads(NOTEBOOK_PATH.read_text())
    cell_lines = nb['cells'][2]['source']

    # Build code up to the first file access
    prefix = []
    for line in cell_lines:
        if line.lstrip().startswith('with h5py.File'):
            break
        prefix.append(line)
    code = ''.join(prefix)

    expected = {'data': {'crimaldi': 'expected.h5', 'video': 'expected.avi'}}

    dummy_yaml = types.SimpleNamespace(safe_load=lambda fh: expected)
    monkeypatch.setitem(sys.modules, 'yaml', dummy_yaml)

    # Provide a fake project_paths.yaml
    yaml_text = "data:\n  crimaldi: expected.h5\n  video: expected.avi\n"
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: io.StringIO(yaml_text))

    # Stubs for modules imported in the notebook
    monkeypatch.setitem(sys.modules, 'h5py', types.ModuleType('h5py'))
    monkeypatch.setitem(sys.modules, 'imageio', types.ModuleType('imageio'))
    monkeypatch.setitem(sys.modules, 'random', types.ModuleType('random'))
    monkeypatch.setitem(sys.modules, 'numpy', types.ModuleType('numpy'))
    monkeypatch.setitem(sys.modules, 'matplotlib', types.ModuleType('matplotlib'))
    sys.modules['imageio'].get_reader = lambda p: []
    sys.modules['h5py'].File = lambda *a, **k: None
    sys.modules['random'].randrange = lambda x: 0
    sys.modules['matplotlib'].pyplot = types.ModuleType('pyplot')
    sys.modules['matplotlib'].pyplot.subplots = lambda *a, **k: (None, [types.SimpleNamespace(), types.SimpleNamespace()])

    env = {}
    exec(code, env)

    assert env['h5_path'] == expected['data']['crimaldi']
    assert env['video_path'] == expected['data']['video']
