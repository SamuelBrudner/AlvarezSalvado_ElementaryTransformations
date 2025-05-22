import os

def test_no_variable_array_directive():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert '$((TOTAL_JOBS - 1))' not in content, 'array directive should not use shell arithmetic'
