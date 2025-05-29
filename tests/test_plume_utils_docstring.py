import ast
from pathlib import Path

MODULE_PATH = Path("Code/plume_utils.py")


def test_plume_utils_docstring_has_example():
    module = ast.parse(MODULE_PATH.read_text())
    doc = ast.get_docstring(module)
    assert doc is not None
    lowered = doc.lower()
    assert "rescale_to_crim_range" in lowered
    assert "configs/plume_intensity_stats.yaml" in lowered
    assert ">>>" in doc
