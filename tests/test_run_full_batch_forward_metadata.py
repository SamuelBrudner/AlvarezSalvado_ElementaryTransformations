def test_run_full_batch_exports_plume_metadata():
    with open("run_full_batch.sh") as f:
        content = f.read()
    assert "export PLUME_METADATA" in content


def test_run_full_batch_echoes_plume_file():
    with open("run_full_batch.sh") as f:
        content = f.read()
    assert ("Using plume metadata" in content) or ("Using plume movie" in content)
