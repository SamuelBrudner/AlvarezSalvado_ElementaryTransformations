import os
import sys
import types
import yaml
import builtins

# Pre-insert dummy matplotlib before importing the module under test
class DummyAx:
    def bar(self, *a, **k):
        pass
    def boxplot(self, *a, **k):
        pass
    def hist(self, *a, **k):
        pass
    def set_title(self, *a, **k):
        pass
    def set_ylabel(self, *a, **k):
        pass
    def set_xlabel(self, *a, **k):
        pass

class DummyFig:
    def __init__(self, axes):
        self.axes = axes
    def tight_layout(self):
        pass
    def savefig(self, path):
        with open(path, 'w') as _:
            pass


def dummy_subplots(nrows, ncols, figsize=None):
    if ncols == 0:
        raise ValueError("Number of columns must be > 0")
    axes = [DummyAx() for _ in range(ncols)]
    fig = DummyFig(axes)
    if ncols == 1:
        return fig, axes[0]
    return fig, axes

def dummy_close(fig):
    pass

mpl = types.ModuleType('matplotlib')
mpl.use = lambda *a, **k: None
plt = types.ModuleType('matplotlib.pyplot')
plt.subplots = dummy_subplots
plt.close = dummy_close
plt.bar = DummyAx().bar
plt.boxplot = DummyAx().boxplot
plt.hist = DummyAx().hist

sys.modules.setdefault('matplotlib', mpl)
sys.modules.setdefault('matplotlib.pyplot', plt)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.generate_dashboard import generate_dashboard
from Code.load_analysis_config import load_analysis_config


SAMPLE_DATA = [
    {"metric": 1, "value": 10},
    {"metric": 2, "value": 20},
]


def test_bar_without_group(tmp_path):
    cfg = {
        "dashboard_layout": {
            "subplots": [
                {"metric": "value", "plot_type": "bar"}
            ],
            "output_filename": "dash.png",
        },
        "output_paths": {"figures": str(tmp_path)},
    }
    path = generate_dashboard(SAMPLE_DATA, cfg)
    assert path and path.exists()


def test_no_subplots_returns_none(tmp_path):
    cfg = {
        "dashboard_layout": {"subplots": []},
        "output_paths": {"figures": str(tmp_path)},
    }
    path = generate_dashboard(SAMPLE_DATA, cfg)
    assert path is None
