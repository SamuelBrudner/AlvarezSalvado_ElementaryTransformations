import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import types

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


pd = types.SimpleNamespace(DataFrame=DummyDataFrame)

class StubPoint:
    def set_data(self, x, y):
        pass


class StubAxis:
    def __init__(self):
        self.point = StubPoint()

    def set_xlim(self, *_, **__):
        pass

    def set_ylim(self, *_, **__):
        pass

    def plot(self, *_, **__):
        return (self.point,)


def _subplots():
    return object(), StubAxis()

plt = types.SimpleNamespace(subplots=_subplots, close=lambda *_: None)

class DummyAnim:
    last_instance = None

    def __init__(self, *_args, **_kwargs):
        DummyAnim.last_instance = self
        self.saved = None

    def save(self, *_a, **_kw):
        self.saved = (_a, _kw)
        return None

animation = types.SimpleNamespace(FuncAnimation=DummyAnim)

sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules["matplotlib"].pyplot = plt
sys.modules["matplotlib"].animation = animation
sys.modules["matplotlib.pyplot"] = plt
sys.modules["matplotlib.animation"] = animation
sys.modules["pandas"] = types.ModuleType("pandas")
sys.modules["pandas"].DataFrame = DummyDataFrame

def _read_csv(path):
    data = {"x": [], "y": []}
    with open(path) as f:
        header = next(f)
        for line in f:
            _, x, y = line.strip().split(",")
            data["x"].append(float(x))
            data["y"].append(float(y))
    return DummyDataFrame(data)

sys.modules["pandas"].read_csv = _read_csv

from Code.trajectory_animation import animate_trajectories


def test_animation_output_file(tmp_path):
    csv_path = tmp_path / "trajectories.csv"
    csv_path.write_text("t,x,y\n0,0,0\n1,1,1\n")
    output_path = tmp_path / "anim.mp4"
    animate_trajectories(csv_path, output_path=output_path)
    assert DummyAnim.last_instance.saved[0][0] == output_path


def test_update_uses_sequence_points(tmp_path, monkeypatch):
    """Verify the update function sets data using sequence inputs."""

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    monkeypatch.setattr("Code.trajectory_animation.pd.read_csv", lambda *_: df)

    fig = object()

    class DummyAxis:
        def __init__(self):
            self.limits = []
            self.point = DummyPoint()

        def set_xlim(self, *_):
            self.limits.append("x")

        def set_ylim(self, *_):
            self.limits.append("y")

        def plot(self, *_, **__):
            return (self.point,)


    class DummyPoint:
        def __init__(self):
            self.data = None

        def set_data(self, x, y):
            self.data = (x, y)


    dummy_axis = DummyAxis()

    def dummy_subplots():
        return fig, dummy_axis

    monkeypatch.setattr("Code.trajectory_animation.plt.subplots", dummy_subplots)

    captured = {}

    class DummyAnim:
        def __init__(self, *_args, **_kwargs):
            captured["update_func"] = _args[1]

        def save(self, *_, **__):
            return None

    monkeypatch.setattr("Code.trajectory_animation.FuncAnimation", DummyAnim)

    animate_trajectories("dummy.csv", output_path=tmp_path / "out.mp4")

    update = captured["update_func"]
    update(0)

    assert dummy_axis.point.data == ([1], [3])
