import os
import importlib
import sys

import pytest




def test_get_plume_file_reads_json(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    # Write a temporary JSON config file
    temp_config = tmp_path / 'config.json'
    temp_config.write_text('{"plume_file": "custom.hdf5"}')

    # Monkeypatch environment variable to point to temp config
    monkeypatch.setenv('PLUME_CONFIG', str(temp_config))

    plume_config = importlib.import_module('plume_config')
    path = plume_config.get_plume_file()
    assert path == 'custom.hdf5'
