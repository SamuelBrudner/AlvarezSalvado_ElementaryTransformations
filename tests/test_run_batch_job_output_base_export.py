import re

def test_find_uses_output_base():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    assert re.search(r'find \"\$OUTPUT_BASE\" -name result.mat', content), (
        'run_batch_job_4000.sh should search OUTPUT_BASE for result.mat')
