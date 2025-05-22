function tests = test_export_results_missing_field
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testMissingXErrors(testCase)
    y = 1;
    matfile = fullfile(tempdir, 'nofield.mat');
    save(matfile, 'y');
    outdir = fullfile(tempdir, 'export_missing');
    if exist(outdir, 'dir'); rmdir(outdir, 's'); end
    f = @() export_results(matfile, outdir);
    verifyError(testCase, f, 'export_results:NoTrajectories');
    delete(matfile);
end
