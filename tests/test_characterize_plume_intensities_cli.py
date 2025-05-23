import os
import subprocess


def run_script(args):
    return subprocess.run(['python', 'Code/characterize_plume_intensities.py'] + args, capture_output=True, text=True)


def test_script_exists():
    assert os.path.isfile('Code/characterize_plume_intensities.py'), 'Script does not exist'


def test_video_requires_px_per_mm_and_frame_rate():
    result = run_script(['--plume_type', 'video', '--file_path', 'path', '--output_json', 'out.json', '--plume_id', 'pid'])
    assert result.returncode != 0


def test_video_valid_arguments():
    result = run_script([
        '--plume_type', 'video',
        '--file_path', 'path',
        '--output_json', 'out.json',
        '--plume_id', 'pid',
        '--px_per_mm', '10.0',
        '--frame_rate', '25.0'
    ])
    assert result.returncode == 0
    assert "plume_type='video'" in result.stdout


def test_crimaldi_valid_arguments():
    result = run_script([
        '--plume_type', 'crimaldi',
        '--file_path', 'path',
        '--output_json', 'out.json',
        '--plume_id', 'pid'
    ])
    assert result.returncode == 0
    assert "plume_type='crimaldi'" in result.stdout
