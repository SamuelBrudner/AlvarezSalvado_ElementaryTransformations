def test_run_test_batch_exports_plume_metadata():
    with open("run_test_batch.sh") as f:
        content = f.read()
    assert "PLUME_METADATA=$PLUME_METADATA" in content


def test_run_test_batch_help_echoes_plume_file():
    with open("run_test_batch.sh") as f:
        content = f.read()
    assert "$DESC              : $PLUME_PATH" in content
