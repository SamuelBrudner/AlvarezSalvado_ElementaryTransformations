import os

def test_logs_output_directory():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    assert "fprintf('Saving results to %s\\n', cfg.outputDir);" in content, (
        'run_batch_job_4000.sh should log output directory in MATLAB script'
    )
