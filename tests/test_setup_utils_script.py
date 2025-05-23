import os
import subprocess

def test_setup_utils_exists():
    assert os.path.isfile('setup_utils.sh'), 'setup_utils.sh does not exist'


def test_setup_utils_functions_defined():
    with open('setup_utils.sh') as f:
        content = f.read()
    assert 'section()' in content
    assert 'error()' in content
    assert 'run_command_verbose()' in content
    assert 'log()' in content


def test_setup_env_sources_utils():
    with open('setup_env.sh') as f:
        content = f.read()
    assert 'setup_utils.sh' in content
    assert 'source' in content

