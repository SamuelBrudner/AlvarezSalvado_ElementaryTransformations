function tests = test_export_results_time_column
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testTimeColumn(testCase)
    T = 3;
    out.x = (1:T)';
    out.y = (1:T)';
    out.theta = zeros(T,1);
    out.odor = zeros(T,1);
    out.ON = zeros(T,1);
    out.OFF = zeros(T,1);
    out.turn = zeros(T,1);
    out.params = struct();
    out.successrate = 1;
    out.latency = T;

    matfile = fullfile(tempdir, 'out.mat');
    save(matfile, 'out');
    outdir = fullfile(tempdir, 'export_out');
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
