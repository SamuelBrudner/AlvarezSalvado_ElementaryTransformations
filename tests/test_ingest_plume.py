import json
import subprocess
from pathlib import Path
import h5py
import numpy as np


def test_ingest_plume_creates_config_and_updates_paths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts' / 'ingest_plume.py'

    # create dummy hdf5 file
    h5_file = tmp_path / 'plume.h5'
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('dataset2', data=np.zeros((4, 5, 6)))

    config_dir = tmp_path / 'configs' / 'plumes'
    config_dir.mkdir(parents=True)
    paths_file = tmp_path / 'configs' / 'paths.json'
    paths_file.parent.mkdir(parents=True, exist_ok=True)
    paths_file.write_text(json.dumps({
        "project_root": str(tmp_path),
        "code_dir": str(tmp_path),
        "data_dir": str(tmp_path),
        "config_dir": str(tmp_path / 'configs'),
        "plume_file": "old.h5",
        "plume_config": "old.json"
    }))

    subprocess.run([
        'python', str(script),
        'new_plume', str(h5_file),
        '--mm-per-pixel', '0.2',
        '--fps', '30',
        '--config-dir', str(config_dir),
        '--paths-file', str(paths_file)
    ], check=True)

    cfg_path = config_dir / 'new_plume.json'
    assert cfg_path.is_file()
    cfg = json.loads(cfg_path.read_text())
    assert cfg['plume_id'] == 'new_plume'
    assert cfg['data_path']['path'] == str(h5_file)
    assert cfg['temporal']['frame_rate'] == 30

    paths = json.loads(paths_file.read_text())
    assert paths['plume_file'] == str(h5_file)
    assert paths['plume_config'] == str(cfg_path)


def test_ingest_plume_updates_pipeline_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts' / 'ingest_plume.py'

    h5_file = tmp_path / 'plume.h5'
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('dataset2', data=np.zeros((2, 3, 4)))

    config_dir = tmp_path / 'configs' / 'plumes'
    config_dir.mkdir(parents=True)
    pipeline_file = tmp_path / 'configs' / 'pipeline' / 'pipeline_plumes.json'
    pipeline_file.parent.mkdir(parents=True)
    pipeline_file.write_text(json.dumps({"plumes": ["existing"]}))

    subprocess.run([
        'python', str(script),
        'new_plume', str(h5_file),
        '--mm-per-pixel', '0.1',
        '--fps', '25',
        '--config-dir', str(config_dir),
        '--pipeline-config', str(pipeline_file)
    ], check=True)

    data = json.loads(pipeline_file.read_text())
    assert 'new_plume' in data['plumes']
