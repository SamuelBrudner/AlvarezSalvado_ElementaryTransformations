function tests = test_batch_job_custom_lists
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testPlumeAndSensingOverride(testCase)
    tmp = [tempname '.yaml'];
    fid = fopen(tmp, 'w');
    fprintf(fid, 'plumes:\n  - foo\n  - bar\n');
    fprintf(fid, 'sensing_modes:\n  - sniff\n');
    fclose(fid);
    cleanup = onCleanup(@() delete(tmp));

    cfg = load_experiment_config(tmp);

    verifyEqual(testCase, cfg.experiment.plume_types, {'foo', 'bar'});
    verifyEqual(testCase, cfg.experiment.sensing_modes, {'sniff'});
    verifyEqual(testCase, cfg.experiment.num_plumes, 2);
    verifyEqual(testCase, cfg.experiment.num_sensing, 1);
end
