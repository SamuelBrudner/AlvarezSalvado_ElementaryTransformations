function tests = test_startup_adds_subfolders
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.rootDir = pwd;
end

function teardownOnce(testCase)
    codeDir = fullfile(testCase.TestData.rootDir, 'Code');
    paths = strsplit(genpath(codeDir), pathsep);
    for i = 1:numel(paths)
        p = paths{i};
        if isempty(p)
            continue
        end
        while any(strcmp(strsplit(path, pathsep), p))
            rmpath(p);
        end
    end
end

function testAddsSubfolderToPath(testCase)
    rootDir = testCase.TestData.rootDir;
    startup; %#ok<NOPRT>
    subDir = fullfile(rootDir, 'Code', 'import functions feb2017');
    entries = strsplit(path, pathsep);
    verifyTrue(testCase, any(strcmp(entries, subDir)), ...
        'Startup should add Code subfolders to path');
end
