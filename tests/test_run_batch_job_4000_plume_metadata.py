def test_plume_metadata_variable_present():
    with open("run_batch_job_4000.sh") as f:
        content = f.read()
    assert "cfg.plume_metadata = '$PLUME_METADATA';" in content
