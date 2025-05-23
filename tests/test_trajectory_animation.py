import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from Code.trajectory_animation import animate_trajectories


def test_animation_output_file(tmp_path):
    csv_path = tmp_path / "trajectories.csv"
    csv_path.write_text("t,x,y\n0,0,0\n1,1,1\n")
    output_path = tmp_path / "anim.mp4"
    animate_trajectories(csv_path, output_path=output_path)
    assert output_path.exists()
