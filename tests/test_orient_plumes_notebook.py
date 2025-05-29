import json
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
