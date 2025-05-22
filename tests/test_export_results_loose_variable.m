function tests = test_export_results_loose_variable
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testLooseVariable(testCase)
    T = 3;
    x = (1:T)';
    y = (1:T)';
    theta = zeros(T,1);
    odor = zeros(T,1);
    ON = zeros(0,1); % empty due to short trial
    OFF = zeros(0,1); % empty due to short trial
    turn = zeros(T,1);
    params = struct();
    successrate = 1;
    latency = T;

    matfile = fullfile(tempdir, 'loose.mat');
    save(matfile, 'x', 'y', 'theta', 'odor', 'ON', 'OFF', 'turn', 'params', 'successrate', 'latency');
    outdir = fullfile(tempdir, 'export_loose');
    if exist(outdir, 'dir')
        rmdir(outdir, 's');
    end
    export_results(matfile, outdir);

    tbl = readtable(fullfile(outdir, 'trajectories.csv'));
    expected_t = (0:T-1)';
    verifyEqual(testCase, tbl.t, expected_t);

    rmdir(outdir, 's');
    delete(matfile);
end
