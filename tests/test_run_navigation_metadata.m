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
    meta.output_directory = tmpDir;
    meta.output_filename = 'dummy.avi';
    meta.vid_mm_per_px = 0.2;
    meta.fps = 40;
    fid = fopen(fullfile(tmpDir, 'meta.json'), 'w');
    fwrite(fid, jsonencode(meta));
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.metaFile = fullfile(tmpDir, 'meta.json');
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
