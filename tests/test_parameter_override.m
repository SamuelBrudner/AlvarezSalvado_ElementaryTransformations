function tests = test_parameter_override
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testOverride(~)
    cfg = load_config(fullfile('tests','parameter_override.yaml'));
    out = run_navigation_cfg(cfg);
    assert(abs(out.params.turnbase - 0.5) < 1e-6, 'turnbase parameter not overridden');
end
