import importlib


def test_rotate_video_clockwise_in_all():
    package = importlib.import_module("Code")
    assert hasattr(package, "__all__"), "package should define __all__"
    assert "rotate_video_clockwise" in package.__all__, "rotate_video_clockwise should be re-exported"
    assert "video_to_scaled_rotated_h5" in package.__all__, "video_to_scaled_rotated_h5 should be re-exported"
