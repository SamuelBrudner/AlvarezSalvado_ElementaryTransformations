import re

def test_progress_logging_present():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert re.search(r'Progress:', content), 'run_batch_job.sh should log progress percentage'

def test_progress_logging_4000_present():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    assert re.search(r'Progress:', content), 'run_batch_job_4000.sh should log progress percentage'
