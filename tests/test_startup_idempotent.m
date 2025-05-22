function tests = test_startup_idempotent
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.rootDir = pwd;
end

function teardownOnce(testCase)
    codeDir = fullfile(testCase.TestData.rootDir, 'Code');
    while any(strcmp(strsplit(path, pathsep), codeDir))
        rmpath(codeDir);
    end
end

function testSinglePathAddition(testCase)
    rootDir = testCase.TestData.rootDir;
    codeDir = fullfile(rootDir, 'Code');
    startup; %#ok<NOPRT>
    startup; %#ok<NOPRT>
    entries = strsplit(path, pathsep);
    numMatches = sum(strcmp(entries, codeDir));
    verifyEqual(testCase, numMatches, 1);
end

function testNoStartupInCode(testCase)
    rootDir = testCase.TestData.rootDir;
    assert(~isfile(fullfile(rootDir, 'Code', 'startup.m')));
end
