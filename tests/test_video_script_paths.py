import pathlib


def test_video_script_uses_orig_script_dir():
    content = pathlib.Path('video_script.m').read_text()
    assert "if exist('orig_script_dir', 'var')" in content
    assert "scriptDir = orig_script_dir;" in content
    assert "fileparts(mfilename('fullpath'))" in content
    assert "fprintf('Script directory:" in content
