import os


def test_default_matlab_version():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert 'MATLAB_VERSION=${MATLAB_VERSION:-R2023b}' in content, \
        'Default MATLAB version should be R2023b'
