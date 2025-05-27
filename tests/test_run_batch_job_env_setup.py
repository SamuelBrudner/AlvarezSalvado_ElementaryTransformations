import os


def test_env_setup_uses_bash():
    with open('run_batch_job.sh') as f:
        content = f.read()
    assert 'bash ./setup_env.sh --dev' in content, 'batch job should invoke setup_env.sh with bash'
    assert 'source setup_env.sh' not in content
