import re

def test_main_cleanup_cleared_before_exit():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    clear_match = re.search(r'echo\s+"clear cleanupObj;"\s*>>\"\$MATLAB_SCRIPT\"', content)
    exit_match = re.search(r'echo\s+"exit"\s*>>\"\$MATLAB_SCRIPT\"', content)
    assert clear_match is not None, 'clear cleanupObj line missing before exit in main script'
    assert exit_match is not None, 'exit line missing in main script'
    assert clear_match.start() < exit_match.start(), 'clear cleanupObj must precede exit'
