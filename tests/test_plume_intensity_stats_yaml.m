function tests = test_plume_intensity_stats_yaml
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testStatsMatchYaml(testCase)
    stats = plume_intensity_stats();
    yamlStats = load_yaml(fullfile('configs','plume_intensity_stats.yaml'));
    verifyEqual(testCase, stats, yamlStats);
end
