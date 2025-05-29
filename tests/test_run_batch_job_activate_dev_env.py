import os


def test_activate_dev_env_after_setup():
    """Ensure conda activation occurs after running setup_env.sh."""
    with open("run_batch_job.sh") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if "bash ./setup_env.sh --dev" in line:
            remainder = "".join(lines[idx + 1:])
            assert (
                "conda activate ./dev_env" in remainder
                or "conda run -p ./dev_env" in remainder
            ), "run_batch_job.sh should activate dev_env after setup"
            break
    else:
        raise AssertionError("setup_env.sh --dev line not found")

