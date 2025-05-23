import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.video_intensity import get_intensities_from_video_via_matlab


def test_script_generation():
    script = get_intensities_from_video_via_matlab(
        video_file_path=r"C:\path\to\video.avi",
        px_per_mm=20,
        frame_rate=50,
        temp_out_file=r"C:\tmp\out.mat",
    )
    assert "load_plume_video" in script
    assert "all_intensities" in script
    assert "C:/tmp/out.mat" in script.replace('\\', '/')
