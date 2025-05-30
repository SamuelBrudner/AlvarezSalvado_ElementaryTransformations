import re

def test_cleanup_cleared_before_exit():
    with open('run_batch_job_4000.sh') as f:
        content = f.read()
    # find indexes of clear cleanupObj and exit lines
    clear_match = re.search(r'echo\s+"clear cleanupObj;"\s*>>\"\$EXPORT_SCRIPT\"', content)
    exit_match = re.search(r'echo\s+"exit"\s*>>\"\$EXPORT_SCRIPT\"', content)
    assert clear_match is not None, 'clear cleanupObj line missing'
    assert exit_match is not None, 'exit line missing'
    assert clear_match.start() < exit_match.start(), 'clear cleanupObj must come before exit'
