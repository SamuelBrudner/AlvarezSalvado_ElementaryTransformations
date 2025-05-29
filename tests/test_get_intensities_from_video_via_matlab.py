import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Code.video_intensity import get_intensities_from_video_via_matlab  # noqa: E402


def test_get_intensities_from_video_via_matlab(monkeypatch, tmp_path):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = 'disp("hello")'

    mat_file = tmp_path / "out.mat"
    from scipy.io import savemat

    savemat(mat_file, {"all_intensities": np.array([1, 2, 3], dtype=np.float32)})

    stdout = f"some log\nTEMP_MAT_FILE_SUCCESS:{mat_file}\n"

    captured = {}

    created_files = []
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        tmp = orig_ntf(*args, **kwargs)
        created_files.append(tmp.name)
        captured["script_path"] = tmp.name
        return tmp

    def fake_run(cmd, capture_output, text, **kwargs):
        assert cmd[0] == matlab_exec
        assert cmd[1] == "-batch"
        assert cmd[2] == f"run('{captured['script_path']}')"
        with open(captured["script_path"]) as fh:
            captured["script_contents"] = fh.read()
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)

    arr = get_intensities_from_video_via_matlab(
        script_content, matlab_exec, px_per_mm=0.5, frame_rate=60.0
    )
    assert np.array_equal(arr, np.array([1, 2, 3], dtype=np.float32))

    assert "px_per_mm = 0.5" in captured["script_contents"]
    assert "frame_rate = 60.0" in captured["script_contents"]

    for f in created_files:
        assert not Path(f).exists(), f"temporary file {f} should be removed"
    assert not mat_file.exists(), "MAT-file should be cleaned up"


def test_path_with_spaces_and_quotes(monkeypatch, tmp_path):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = 'disp("hello")'

    mat_file = tmp_path / "out.mat"
    from scipy.io import savemat

    savemat(mat_file, {"all_intensities": np.array([1], dtype=np.float32)})

    stdout = f"ok\nTEMP_MAT_FILE_SUCCESS:{mat_file}\n"

    captured = {}

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        file_path = tmp_path / "with space's" / "temp.m"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(file_path, "w+b")
        captured["script_path"] = str(file_path)
        return fh

    def fake_run(cmd, capture_output, text, **kwargs):
        assert cmd[0] == matlab_exec
        expected = captured["script_path"].replace("'", "''")
        assert cmd[2] == f"run('{expected}')"
        with open(captured["script_path"]) as fh:
            captured["script_contents"] = fh.read()
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    monkeypatch.setattr(subprocess, "run", fake_run)

    arr = get_intensities_from_video_via_matlab(script_content, matlab_exec)
    assert np.array_equal(arr, np.array([1], dtype=np.float32))

    assert "disp(" in captured["script_contents"]
    assert not Path(captured["script_path"]).exists()
    assert not mat_file.exists()


def test_error_message_includes_path_and_workdir(monkeypatch, tmp_path, caplog):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = "disp('fail')"

    def fake_run(cmd, capture_output, text, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="oops", stderr="bad")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError) as exc:
        get_intensities_from_video_via_matlab(
            script_content,
            matlab_exec,
            work_dir=str(tmp_path),
            orig_script_path="orig.m",
        )

    msg = exc.value.args[0]
    assert "orig.m" in msg
    assert str(tmp_path) in msg
    assert "orig_script_dir" in msg
    assert any("MATLAB failed while running" in r.message for r in caplog.records)


def test_workdir_with_single_quote(monkeypatch, tmp_path):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = 'disp("hi")'

    mat_file = tmp_path / "out.mat"
    from scipy.io import savemat

    savemat(mat_file, {"all_intensities": np.array([7], dtype=np.float32)})

    stdout = f"ok\nTEMP_MAT_FILE_SUCCESS:{mat_file}\n"

    captured = {}
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        tmp = orig_ntf(*args, **kwargs)
        captured["script_path"] = tmp.name
        return tmp

    def fake_run(cmd, capture_output, text, **kwargs):
        assert kwargs.get("cwd") == work_dir
        expected = captured["script_path"].replace("'", "''")
        assert cmd[0] == matlab_exec
        assert cmd[2] == f"run('{expected}')"
        with open(captured["script_path"]) as fh:
            captured["script_contents"] = fh.read()
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    monkeypatch.setattr(subprocess, "run", fake_run)

    work_dir = str(tmp_path / "with'quote")
    Path(work_dir).mkdir()

    arr = get_intensities_from_video_via_matlab(
        script_content, matlab_exec, work_dir=work_dir
    )
    assert np.array_equal(arr, np.array([7], dtype=np.float32))

    escaped = work_dir.replace("'", "''")
    expected_cd = f"cd('{escaped}')"
    assert expected_cd in captured["script_contents"]
    assert not Path(captured["script_path"]).exists()
    assert not mat_file.exists()


def test_logs_stdout_on_failure(monkeypatch, caplog):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = "disp('fail')"

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="hello", stderr="bad")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        get_intensities_from_video_via_matlab(script_content, matlab_exec)

    assert any("hello" in r.message for r in caplog.records)


def test_matlab_batch_command_uses_brackets(monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        get_intensities_from_video_via_matlab('disp("fail")', "matlab")

    batch_idx = captured["cmd"].index("-batch") + 1
    batch_arg = captured["cmd"][batch_idx]
    assert "disp(['MATLAB Error: ' getReport(ME, 'extended')])" in batch_arg
    assert "+ getReport" not in batch_arg


def test_reads_v7_3_mat_file(monkeypatch, tmp_path):
    matlab_exec = "/usr/local/MATLAB/R2023b/bin/matlab"
    script_content = 'disp("hi")'

    mat_file = tmp_path / "out.mat"
    import h5py

    np = pytest.importorskip("numpy")
    with h5py.File(mat_file, "w") as f:
        f.create_dataset("all_intensities", data=np.array([9, 8, 7], dtype=np.float32))

    stdout = f"ok\nTEMP_MAT_FILE_SUCCESS:{mat_file}\n"

    captured = {}
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault("delete", False)
        tmp = orig_ntf(*args, **kwargs)
        captured["script_path"] = tmp.name
        return tmp

    def fake_run(cmd, capture_output, text, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    monkeypatch.setattr(subprocess, "run", fake_run)

    arr = get_intensities_from_video_via_matlab(script_content, matlab_exec)
    assert np.array_equal(arr, np.array([9, 8, 7], dtype=np.float32))
    assert not Path(captured["script_path"]).exists()
    assert not mat_file.exists()


def test_command_uses_given_matlab_executable(monkeypatch):
    passed = "/usr/bin/custom"

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        get_intensities_from_video_via_matlab("disp('fail')", passed)

    assert captured["cmd"][0] == passed
