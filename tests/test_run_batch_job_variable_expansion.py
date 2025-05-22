import os
import re


def test_variables_expanded_in_loop():
    with open('run_batch_job.sh') as f:
        content = f.read()
    # Expect a loop that sets SEED and AGENT_DIR for each agent
    loop_pattern = re.compile(r'for .*in.*RANDOM_SEEDS')
    assert loop_pattern.search(content), 'run_batch_job.sh should loop over RANDOM_SEEDS'
    assert 'AGENT_DIR=' in content, 'AGENT_DIR should be defined within the loop'
    assert 'SEED=' in content, 'SEED should be defined within the loop'
