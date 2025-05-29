import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def simple_stats(values):
    values = sorted(values)
    n = len(values)

    def pct(p):
        k = (n - 1) * p / 100
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return float(values[f])
        return float(values[f] * (c - k) + values[c] * (k - f))

    mean = sum(values) / n if n else float("nan")
    if n % 2 == 1:
        median = float(values[n // 2])
    else:
        median = (values[n // 2 - 1] + values[n // 2]) / 2
    return {
        "mean": float(mean),
        "median": float(median),
        "p95": pct(95),
        "p99": pct(99),
        "min": float(values[0]),
        "max": float(values[-1]),
        "count": n,
    }


@pytest.fixture(autouse=True)
def _stub_intensity_stats(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "Code.intensity_stats",
        types.SimpleNamespace(calculate_intensity_stats_dict=simple_stats),
    )


from Code import characterize_plume_intensities as cpi


def test_video_requires_px_per_mm_and_frame_rate(monkeypatch, tmp_path):
    script = tmp_path / "script.m"
    script.write_text("disp('hi')")
    out_json = tmp_path / "out.json"
    fake_crim = types.SimpleNamespace(get_intensities_from_crimaldi=lambda p: [1])
    fake_vid = types.SimpleNamespace(

        get_intensities_from_video_via_matlab=lambda s, m, px_per_mm, frame_rate, orig_script_path=None, **kwargs: [
            1
        ]
    )
    monkeypatch.setitem(sys.modules, "Code.analyze_crimaldi_data", fake_crim)
    monkeypatch.setitem(sys.modules, "Code.video_intensity", fake_vid)
    with pytest.raises(SystemExit):
        cpi.main(
            [
                "--plume_type",
                "video",
                "--file_path",
                str(script),
                "--output_json",
                str(out_json),
                "--plume_id",
                "pid",
            ]
        )


def test_video_valid_arguments(monkeypatch, tmp_path):
    script = tmp_path / "script.m"
    script.write_text("disp('hi')")
    out_json = tmp_path / "out.json"
    fake_crim = types.SimpleNamespace(get_intensities_from_crimaldi=lambda p: [1])
    captured = {}

    def fake_vid_func(s, m, px_per_mm, frame_rate, orig_script_path=None, **kwargs):
        captured["px_per_mm"] = px_per_mm
        captured["frame_rate"] = frame_rate
        captured["matlab_exec"] = m
        captured["script"] = s
        return [1.0, 2.0, 3.0]

    fake_vid = types.SimpleNamespace(
        get_intensities_from_video_via_matlab=fake_vid_func
    )
    monkeypatch.setitem(sys.modules, "Code.analyze_crimaldi_data", fake_crim)
    monkeypatch.setitem(sys.modules, "Code.video_intensity", fake_vid)

    cpi.main(
        [
            "--plume_type",
            "video",
            "--file_path",
            str(script),
            "--output_json",
            str(out_json),
            "--plume_id",
            "pid",
            "--px_per_mm",
            "10",
            "--frame_rate",
            "25",
        ]
    )

    data = json.loads(out_json.read_text())
    assert data[0]["plume_id"] == "pid"
    assert data[0]["statistics"]["count"] == 3
    assert captured["px_per_mm"] == 10
    assert captured["matlab_exec"] == "matlab"
    assert captured["frame_rate"] == 25
    assert captured["script"] == script.read_text()


def test_crimaldi_valid_arguments(monkeypatch, tmp_path):
    hdf5_file = tmp_path / "sample.hdf5"
    hdf5_file.write_text("dummy")
    fake_crim = types.SimpleNamespace(
        get_intensities_from_crimaldi=lambda path: [4.0, 5.0]
    )
    fake_vid = types.SimpleNamespace(

        get_intensities_from_video_via_matlab=lambda s, m, px_per_mm, frame_rate, orig_script_path=None, **kwargs: [
            1
        ]
    )
    monkeypatch.setitem(sys.modules, "Code.analyze_crimaldi_data", fake_crim)
    monkeypatch.setitem(sys.modules, "Code.video_intensity", fake_vid)
    out_json = tmp_path / "out.json"

    cpi.main(
        [
            "--plume_type",
            "crimaldi",
            "--file_path",
            str(hdf5_file),
            "--output_json",
            str(out_json),
            "--plume_id",
            "pid",
        ]
    )

    data = json.loads(out_json.read_text())
    assert data[0]["plume_id"] == "pid"
    assert data[0]["statistics"]["count"] == 2


def test_matlab_exec_option(monkeypatch, tmp_path):
    script = tmp_path / "script.m"
    script.write_text("disp('hi')")
    out_json = tmp_path / "out.json"
    fake_crim = types.SimpleNamespace(get_intensities_from_crimaldi=lambda p: [1])
    captured = {}


    def fake_vid_func(s, m, px_per_mm, frame_rate, orig_script_path=None, **kwargs):
        captured["matlab_exec"] = m
        return [1]

    fake_vid = types.SimpleNamespace(
        get_intensities_from_video_via_matlab=fake_vid_func
    )
    monkeypatch.setitem(sys.modules, "Code.analyze_crimaldi_data", fake_crim)
    monkeypatch.setitem(sys.modules, "Code.video_intensity", fake_vid)

    cpi.main(
        [
            "--plume_type",
            "video",
            "--file_path",
            str(script),
            "--output_json",
            str(out_json),
            "--plume_id",
            "pid",
            "--px_per_mm",
            "10",
            "--frame_rate",
            "25",
            "--matlab_exec",
            "/custom/matlab",
        ]
    )
    assert captured["matlab_exec"] == "/custom/matlab"
