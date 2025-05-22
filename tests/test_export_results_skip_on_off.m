function tests = test_export_results_skip_on_off
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testSkipOnOff(testCase)
    T = 3;
    out.x = (1:T)';
    out.y = (1:T)';
    out.theta = zeros(T,1);
    out.odor = zeros(T,1);
    out.ON = [];
    out.OFF = [];
    out.turn = zeros(T,1);
    out.params = struct();
    out.successrate = 1;
    out.latency = T;

    matfile = fullfile(tempdir, 'skip.mat');
    save(matfile, 'out');
    outdir = fullfile(tempdir, 'export_skip');
    if exist(outdir, 'dir'); rmdir(outdir, 's'); end
    export_results(matfile, outdir);

    tbl = readtable(fullfile(outdir, 'trajectories.csv'));
    verifyFalse(testCase, any(strcmp('ON', tbl.Properties.VariableNames)));
    verifyFalse(testCase, any(strcmp('OFF', tbl.Properties.VariableNames)));

    rmdir(outdir, 's');
    delete(matfile);
end
