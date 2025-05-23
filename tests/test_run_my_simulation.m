function tests = test_run_my_simulation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.rootDir = pwd;
    cfgFile = fullfile('configs', 'my_complex_plume_config.yaml');
    fid = fopen(cfgFile, 'w');
    fprintf(fid, 'environment: gaussian\n');
    fprintf(fid, 'triallength: 50\n');
    fprintf(fid, 'plotting: 0\n');
    fprintf(fid, 'ntrials: 1\n');
    fclose(fid);
    testCase.TestData.cfgFile = cfgFile;
end

function teardownOnce(testCase)
    delete(testCase.TestData.cfgFile);
end

function testRunScript(~)
    run_my_simulation;
    assert(true); % ensure no error
end

function testRunScriptInDifferentDir(testCase)
    tmpDir = tempname;
    mkdir(tmpDir);
    oldDir = cd(tmpDir);
    run(fullfile(testCase.TestData.rootDir, 'Code', 'run_my_simulation.m'));
    cd(oldDir);
    rmdir(tmpDir, 's');
    assert(true);
end
