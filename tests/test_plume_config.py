import os
import importlib
import sys

import pytest




def test_get_plume_file_reads_json(tmp_path, monkeypatch):
    """When PLUME_CONFIG points to a JSON file the path should be used."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    temp_config = tmp_path / 'config.json'
    temp_config.write_text('{"plume_file": "custom.hdf5"}')
    monkeypatch.setenv('PLUME_CONFIG', str(temp_config))

    plume_config = importlib.import_module('plume_config')
    path = plume_config.get_plume_file()
    assert path == "custom.hdf5"


def test_get_plume_file_with_path(tmp_path, monkeypatch):
    """plume_path is prepended when present."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    temp_config = tmp_path / 'config.json'
    temp_config.write_text(
        '{"plume_file": "custom.hdf5", "plume_path": "' + str(tmp_path) + '"}'
    )
    monkeypatch.setenv('PLUME_CONFIG', str(temp_config))

    plume_config = importlib.import_module('plume_config')
    importlib.reload(plume_config)
    expected = os.path.join(str(tmp_path), "custom.hdf5")
    assert plume_config.get_plume_file() == expected
