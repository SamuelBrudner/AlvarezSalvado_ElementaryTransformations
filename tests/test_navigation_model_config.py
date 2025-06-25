import json
import os
import subprocess
from pathlib import Path


def run_model(env):
    root = Path(__file__).resolve().parents[1]
    script = "out = navigation_model_vec(10, 'Crimaldi', 0, 1);"
    result = subprocess.run([
        'bash', str(root / 'scripts/run_matlab_safe.sh')
    ], input=script, text=True, capture_output=True, cwd=root, env=env)
    return result


def test_navigation_model_default_crimaldi():
    env = os.environ.copy()
    env.pop('MATLAB_PLUME_FILE', None)
    result = run_model(env)
    assert result.returncode == 0
    assert 'Loaded config:' in result.stdout
    assert '15.0' in result.stdout


def test_navigation_model_smoke_config():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs' / 'plumes' / 'smoke_1a_backgroundsubtracted.json').read_text())
    env = os.environ.copy()
    env['MATLAB_PLUME_FILE'] = cfg['data_path']['path']
    result = run_model(env)
    assert result.returncode == 0
    assert 'Loaded config:' in result.stdout
    assert '60.0' in result.stdout
