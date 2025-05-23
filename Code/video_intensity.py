"""Helper to extract video plume intensities using MATLAB."""

from __future__ import annotations

from pathlib import Path

def get_intensities_from_video_via_matlab(
    video_file_path: str | Path,
    px_per_mm: float,
    frame_rate: float,
    temp_out_file: str | Path,
) -> str:
    """Return MATLAB script to extract intensities from a video file.

    Parameters
    ----------
    video_file_path : str or Path
        Path to the plume video file.
    px_per_mm : float
        Pixels per millimeter scaling for the video.
    frame_rate : float
        Frame rate of the video in Hz.
    temp_out_file : str or Path
        Path to a temporary ``.mat`` file to store intensities.

    Returns
    -------
    str
        MATLAB script that can be executed to save ``all_intensities`` to
        ``temp_out_file``.
    """
    video_file = Path(video_file_path)
    out_file = Path(temp_out_file)

    template = r"""
try
    addpath('{code_dir}')
    plume = load_plume_video('{video}', {px_per_mm}, {frame_rate});
    all_intensities = plume.data(:);
    save('{out_file}', 'all_intensities');
    fprintf('INTENSITY_EXTRACTION_SUCCESS:%s\n', '{out_file}');
catch ME
    disp(getReport(ME));
    exit(1);
end
"""

    script = template.format(
        code_dir=str(Path('Code').resolve()).replace('\\', '/'),
        video=str(video_file).replace('\\', '/'),
        px_per_mm=px_per_mm,
        frame_rate=frame_rate,
        out_file=str(out_file).replace('\\', '/'),
    )
    return script
