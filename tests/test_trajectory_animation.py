import importlib
import os
import sys
import types

import pytest
from _pytest.monkeypatch import MonkeyPatch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DummySeries(list):
    def min(self):
        return min(self)

    def max(self):
        return max(self)


class DummyDataFrame:
    def __init__(self, data):
        self._data = {k: DummySeries(v) for k, v in data.items()}

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(next(iter(self._data.values())))

    class _ILoc:
        def __init__(self, parent):
            self._parent = parent

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self._parent._data.items()}

    @property
    def iloc(self):
        return DummyDataFrame._ILoc(self)


class StubPoint:
    def __init__(self):
        self.data = None

    def set_data(self, x, y):
        self.data = (x, y)


class StubAxis:
    def __init__(self):
        self.point = StubPoint()
        self.limits = []

    def set_xlim(self, *_, **__):
        self.limits.append("x")

    def set_ylim(self, *_, **__):
        self.limits.append("y")

    def plot(self, *_, **__):
        return (self.point,)


def _subplots():
    return object(), StubAxis()


class DummyAnim:
    """Capture animation save calls."""

    last_instance = None

    def __init__(self, *_args, **_kwargs):
        DummyAnim.last_instance = self
        self.saved = None

    def save(self, *a, **kw):
        self.saved = (a, kw)
        return None


@pytest.fixture(scope="module")
def anim_module():
    """Provide ``Code.trajectory_animation`` with stubbed dependencies."""

    def _read_csv(path):
        data = {"x": [], "y": []}
        with open(path) as f:
            next(f)  # header
            for line in f:
                cols = line.strip().split(",")
                data["x"].append(float(cols[1]))
                data["y"].append(float(cols[2]))
        return DummyDataFrame(data)

    # pandas stub
    mp = MonkeyPatch()

    pd_mod = types.ModuleType("pandas")
    pd_mod.read_csv = _read_csv
    pd_mod.DataFrame = DummyDataFrame
    mp.setitem(sys.modules, "pandas", pd_mod)

    # matplotlib stubs
    plt_mod = types.ModuleType("matplotlib.pyplot")
    plt_mod.subplots = _subplots
    plt_mod.close = lambda *_: None

    anim_mod = types.ModuleType("matplotlib.animation")
    anim_mod.FuncAnimation = DummyAnim

    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.pyplot = plt_mod
    matplotlib_mod.animation = anim_mod

    mp.setitem(sys.modules, "matplotlib", matplotlib_mod)
    mp.setitem(sys.modules, "matplotlib.pyplot", plt_mod)
    mp.setitem(sys.modules, "matplotlib.animation", anim_mod)

    module = importlib.import_module("Code.trajectory_animation")
    yield module
    mp.undo()


def test_animation_output_file(tmp_path, anim_module):
    csv_path = tmp_path / "trajectories.csv"
    csv_path.write_text("t,x,y\n0,0,0\n1,1,1\n")
    output_path = tmp_path / "anim.mp4"
    anim_module.animate_trajectories(csv_path, output_path=output_path)
    assert DummyAnim.last_instance.saved[0][0] == output_path


def test_update_uses_sequence_points(tmp_path, anim_module, monkeypatch):
    """Verify the update function sets data using sequence inputs."""

    df = DummyDataFrame({"x": [1, 2], "y": [3, 4]})
    monkeypatch.setattr(anim_module.pd, "read_csv", lambda *_: df)

    fig = object()
    dummy_axis = StubAxis()

    def dummy_subplots():
        return fig, dummy_axis

    monkeypatch.setattr(anim_module.plt, "subplots", dummy_subplots)

    captured = {}

    class LocalAnim:
        def __init__(self, *args, **_kwargs):
            captured["update_func"] = args[1]

        def save(self, *_, **__):
            return None

    monkeypatch.setattr(anim_module, "FuncAnimation", LocalAnim)

    anim_module.animate_trajectories("dummy.csv", output_path=tmp_path / "out.mp4")

    update = captured["update_func"]
    update(0)
    assert dummy_axis.point.data == ([1], [3])
