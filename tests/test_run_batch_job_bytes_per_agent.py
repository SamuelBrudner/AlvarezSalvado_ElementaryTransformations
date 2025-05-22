import re

def test_bytes_per_agent_increased():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    match = re.search(r'BYTES_PER_AGENT=(\d+)', content)
    assert match, 'BYTES_PER_AGENT should be defined'
    value = int(match.group(1))
    assert value >= 5000000, 'BYTES_PER_AGENT should account for raw .mat file size'
