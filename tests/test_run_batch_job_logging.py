import os


def test_logging_directory_created():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert 'mkdir -p logs' in content, 'run_batch_job.sh should create logs directory'


def test_log_file_variable():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert 'JOB_LOG=' in content, 'run_batch_job.sh should define JOB_LOG variable'
