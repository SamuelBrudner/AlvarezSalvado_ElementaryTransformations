import re


def test_output_directory_pattern():
    with open('run_batch_job.sh') as f:
        content = f.read()
    pattern = r'AGENT_DIR="\${OUTPUT_BASE}/\${CONDITION_NAME}/\${AGENT_INDEX}_\${SEED}"'
    assert re.search(pattern, content), 'Output directory pattern not updated'
