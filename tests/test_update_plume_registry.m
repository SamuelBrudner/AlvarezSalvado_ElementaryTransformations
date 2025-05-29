function tests = test_update_plume_registry
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    testCase.TestData.yml = [tempname '.yaml'];
end

function teardownOnce(testCase)
    if exist(testCase.TestData.yml, 'file')
        delete(testCase.TestData.yml);
    end
end

function testCreatesEntry(testCase)
    update_plume_registry('file.h5', 1, 2, testCase.TestData.yml);
    data = load_yaml(testCase.TestData.yml);
    verifyEqual(testCase, data.("file.h5").min, 1);
    verifyEqual(testCase, data.("file.h5").max, 2);
end

function testExpandsRange(testCase)
    update_plume_registry('file.h5', 1, 2, testCase.TestData.yml);
    update_plume_registry('file.h5', 0.5, 3, testCase.TestData.yml);
    data = load_yaml(testCase.TestData.yml);
    verifyEqual(testCase, data.("file.h5").min, 0.5);
    verifyEqual(testCase, data.("file.h5").max, 3);
end
