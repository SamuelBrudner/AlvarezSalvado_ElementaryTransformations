function tests = test_run_navigation_metadata
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = fullfile(tempname);
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'dummy.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,3)));
    close(vw);
    meta_path = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta_path, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: dummy.avi\n');
    fprintf(fid, 'vid_mm_per_px: 0.2\n');
    fprintf(fid, 'fps: 40\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.metaFile = meta_path;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testRunNavigationCfg(testCase)
    cfg.plume_metadata = testCase.TestData.metaFile;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    run_navigation_cfg(cfg);
    assert(true); % ensure no error
end
