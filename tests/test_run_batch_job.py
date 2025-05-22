import os
import re

def test_run_batch_job_exists():
    assert os.path.isfile('run_batch_job.sh'), 'run_batch_job.sh does not exist'


def test_run_batch_job_contents():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert '#SBATCH --partition=' in content
    assert ': ${PLUME_CONFIG:="' in content
    assert ': ${OUTPUT_BASE:="' in content
    assert 'AGENT_DIR="${OUTPUT_BASE}/${CONDITION_NAME}/${AGENT_INDEX}_${SEED}"' in content

