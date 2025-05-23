import os


def test_run_agent_simulation_save_struct():
    with open(os.path.join('Code', 'run_agent_simulation.m')) as f:
        content = f.read()
    assert "save(fullfile(output_dir, 'result.mat'), '-struct', 'result');" in content, \
        "run_agent_simulation.m should save results with -struct"

