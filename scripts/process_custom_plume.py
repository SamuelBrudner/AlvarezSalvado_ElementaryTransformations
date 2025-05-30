#!/usr/bin/env python3
"""Process a custom plume video into rotated, scaled HDF5 with metadata."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Code.plume_pipeline import video_to_scaled_rotated_h5  # noqa: E402

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - minimal YAML writer
    yaml = None


def _write_meta(
    path: Path, directory: Path, rotated_h5: str, px_per_mm: float, fps: float
) -> None:
    mm_per_px = 1 / px_per_mm
    info = {
        "output_directory": str(directory),
        "output_filename": Path(rotated_h5).name,
        "vid_mm_per_px": mm_per_px,
        "fps": fps,
        "scaled_to_crim": True,
    }
    if yaml is not None:
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(info, fh)
    else:  # pragma: no cover - fallback
        with path.open("w", encoding="utf-8") as fh:
            for key, value in info.items():
                fh.write(f"{key}: {value}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Process custom plume video")
    parser.add_argument("input_video", help="Input AVI video")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("px_per_mm", type=float, help="Pixels per millimetre")
    parser.add_argument("frame_rate", type=float, help="Frame rate of video")
    ns = parser.parse_args(argv)

    in_path = Path(ns.input_video)
    out_dir = Path(ns.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_path.stem

    raw_h5 = out_dir / f"{base}_raw.h5"
    scaled_h5 = out_dir / f"{base}_scaled.h5"
    rotated_h5 = out_dir / f"{base}_rotated.h5"

    video_to_scaled_rotated_h5(in_path, raw_h5, scaled_h5, rotated_h5)

    meta = out_dir / f"{base}_meta.yaml"
    _write_meta(meta, out_dir, rotated_h5.name, ns.px_per_mm, ns.frame_rate)


if __name__ == "__main__":  # pragma: no cover
    main()
