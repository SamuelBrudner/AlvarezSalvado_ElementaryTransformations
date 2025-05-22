import re

def test_output_path_includes_experiment_name():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    pattern = r'OUT_DIR="\\${OUTPUT_BASE}/\\${EXPERIMENT_NAME}/\\${PLUME_NAME}_\\${SENSING_NAME}/\\${AGENT_ID}_\\${SEED}"'
    assert re.search(pattern, content), 'Output path should include EXPERIMENT_NAME'
