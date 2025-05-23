import os


def test_setup_env_script_has_log_function():
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'log()' in content, 'setup_env.sh should define a log function'
    assert 'INFO' in content
    assert 'SUCCESS' in content
    assert 'WARNING' in content
    assert 'ERROR' in content
