"""Animation utilities for agent trajectories.

Examples
--------
>>> from Code.trajectory_animation import animate_trajectories
>>> animate_trajectories('trajectories.csv', output_path='anim.mp4')
PosixPath('anim.mp4')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def animate_trajectories(
    trajectories_path: str | Path,
    plume: Dict[str, Any] | None = None,
    *,
    output_path: str | Path = "trajectory_animation.mp4",
    config: Dict[str, Any] | None = None,
) -> Path:
    """Generate an animation of agent trajectories.

    Parameters
    ----------
    trajectories_path : str or Path
        Location of the ``trajectories.csv`` file.
    plume : dict, optional
        Static plume definition. Currently unused.
    output_path : str or Path, optional
        File path for the resulting animation (GIF or MP4).
    config : dict, optional
        Dictionary of animation parameters.

    Returns
    -------
    pathlib.Path
        Path to the saved animation file.
    """

    df = pd.read_csv(trajectories_path)
    out_path = Path(output_path)

    fps = 10
    if config is not None:
        fps = config.get("fps", fps)

    fig, ax = plt.subplots()
    ax.set_xlim(df["x"].min() - 5, df["x"].max() + 5)
    ax.set_ylim(df["y"].min() - 5, df["y"].max() + 5)
    (point,) = ax.plot([], [], "o")

    def init() -> Iterable:
        point.set_data([], [])
        return (point,)

    def update(frame: int) -> Iterable:
        row = df.iloc[frame]
        point.set_data(row["x"], row["y"])
        return (point,)

    anim = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    anim.save(out_path, fps=fps)
    plt.close(fig)

    return out_path
