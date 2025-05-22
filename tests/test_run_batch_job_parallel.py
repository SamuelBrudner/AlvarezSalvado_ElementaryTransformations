import os

def test_run_batch_job_uses_parfor():
    filepath = os.path.join('Code', 'run_batch_job.m')
    assert os.path.isfile(filepath), 'run_batch_job.m should exist'
    with open(filepath) as f:
        content = f.read()
    assert 'parfor' in content.lower(), 'run_batch_job.m should use parfor for parallel execution'
