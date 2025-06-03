import os
import importlib
import sys
import logging

import pytest


def test_get_plume_file_reads_json(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    temp_config = tmp_path / 'config.json'
    temp_config.write_text('{"plume_file": "custom.hdf5"}')
    monkeypatch.setenv('PLUME_CONFIG', str(temp_config))

    plume_config = importlib.import_module('plume_config')
    path = plume_config.get_plume_file()
    assert path == 'custom.hdf5'


def test_get_plume_file_missing_config(tmp_path, monkeypatch, caplog):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    missing = tmp_path / 'missing.json'
    monkeypatch.setenv('PLUME_CONFIG', str(missing))
    plume_config = importlib.import_module('plume_config')
    with caplog.at_level(logging.DEBUG):
        path = plume_config.get_plume_file()
    assert path == '10302017_10cms_bounded.hdf5'
    assert any('not found' in rec.getMessage() for rec in caplog.records)
