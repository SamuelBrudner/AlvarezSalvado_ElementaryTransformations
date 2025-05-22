import re

def test_agent_env_override():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert re.search(r': \${AGENTS_PER_CONDITION:=', content), 'AGENTS_PER_CONDITION should be environment overridable'
    assert re.search(r': \${AGENTS_PER_JOB:=', content), 'AGENTS_PER_JOB should be environment overridable'

