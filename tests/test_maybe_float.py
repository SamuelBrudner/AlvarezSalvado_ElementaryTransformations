import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code.data_discovery import _maybe_float


def test_maybe_float_non_string():
    assert _maybe_float(5) == 5
    assert _maybe_float(None) is None


def test_maybe_float_strings():
    assert _maybe_float("3.5") == 3.5
    assert math.isnan(_maybe_float("nan"))
