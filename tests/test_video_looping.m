function tests = test_video_looping
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = fullfile(tempname);
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'loop.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);
    meta = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: loop.avi\n');
    fprintf(fid, 'vid_mm_per_px: 1\n');
    fprintf(fid, 'fps: 1\n');
    fclose(fid);
    testCase.TestData.meta = meta;
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testLongTrial(testCase)
    cfg.plume_metadata = testCase.TestData.meta;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    cfg.triallength = 5; % longer than 1 frame
    run_navigation_cfg(cfg);
    assert(true); % ensure no error
end
