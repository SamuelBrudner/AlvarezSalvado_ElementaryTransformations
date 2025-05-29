import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.load_analysis_config import load_analysis_config


def test_load_analysis_config():
    cfg = load_analysis_config(os.path.join('configs', 'analysis_config.yaml'))
    assert cfg['data_paths']['raw_data_dir'] == 'data/raw'
    assert 'metrics_calculation' in cfg
    assert 'output_paths' in cfg
    assert 'statistical_analysis' in cfg
    assert 'statistical_tests' not in cfg


def test_load_analysis_config_missing_file():
    """load_analysis_config should raise when file is absent."""
    nonexistent = os.path.join('configs', 'i_do_not_exist.yaml')
    with pytest.raises(FileNotFoundError):
        load_analysis_config(nonexistent)
