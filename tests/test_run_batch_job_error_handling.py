import os


def test_cleanup_trap_present():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert 'cleanup()' in content, 'cleanup function should be defined'
    assert 'trap cleanup' in content, 'cleanup function should be trapped'
