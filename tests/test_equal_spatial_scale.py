import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backend.examples.analysis_visualization import set_equal_spatial_scale


def test_set_equal_spatial_scale():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-1, 1)
    ax2.set_xlim(-5, 10)
    ax2.set_ylim(-2, 2)

    set_equal_spatial_scale([ax1, ax2])

    assert ax1.get_xlim() == ax2.get_xlim()
    assert ax1.get_ylim() == ax2.get_ylim()
