import json
import subprocess
from pathlib import Path


def test_setup_plume_info_creates_configs(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/setup_plume_info.py'
    smoke_file = tmp_path / 'smoke.h5'
    crimaldi_file = tmp_path / 'crimaldi.h5'
    smoke_file.write_text('dummy')
    crimaldi_file.write_text('dummy')
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()

    subprocess.run([
        'python', str(script),
        '--smoke-file', str(smoke_file),
        '--crimaldi-file', str(crimaldi_file),
        '--config-dir', str(config_dir)
    ], check=True)

    smoke_config = json.loads((config_dir / 'plumes_smoke_info.json').read_text())
    crimaldi_config = json.loads((config_dir / 'plumes_crimaldi_info.json').read_text())

    assert smoke_config['plume_file'] == str(smoke_file)
    assert smoke_config['mm_per_pixel'] == 0.153
    assert smoke_config['frame_rate'] == 60

    assert crimaldi_config['plume_file'] == str(crimaldi_file)
    assert crimaldi_config['mm_per_pixel'] == 0.74
    assert crimaldi_config['frame_rate'] == 15
