function tests = test_run_custom_plume_test
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd,'Code'));
    tmpDir = fullfile(tempname);
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'plume.avi'));
    open(vw);
    writeVideo(vw,uint8(zeros(2,2,1)));
    close(vw);
    meta = fullfile(tmpDir,'meta.yaml');
    fid = fopen(meta,'w');
    fprintf(fid,'output_directory: %s\n',tmpDir);
    fprintf(fid,'output_filename: plume.avi\n');
    fprintf(fid,'vid_mm_per_px: 1\n');
    fprintf(fid,'fps: 1\n');
    fclose(fid);
    cfg = fullfile(tmpDir,'cfg.yaml');
    fid = fopen(cfg,'w');
    fprintf(fid,'environment: video\n');
    fprintf(fid,'plume_metadata: %s\n',meta);
    fprintf(fid,'plotting: 0\n');
    fprintf(fid,'ntrials: 1\n');
    fclose(fid);
    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.cfg = cfg;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir,'s');
end

function testRun(testCase)
    run_custom_plume_test(testCase.TestData.cfg);
    assert(true); % ensure script runs without error
end
