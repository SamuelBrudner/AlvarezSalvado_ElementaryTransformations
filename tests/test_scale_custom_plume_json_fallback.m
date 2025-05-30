function tests = test_scale_custom_plume_json_fallback
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);

    vw = VideoWriter(fullfile(tmpDir, 'orig.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);

    meta = fullfile(tmpDir, 'meta.yaml');
    fid = fopen(meta, 'w');
    fprintf(fid, 'output_directory: %s\n', tmpDir);
    fprintf(fid, 'output_filename: orig.avi\n');
    fprintf(fid, 'vid_mm_per_px: 1\n');
    fprintf(fid, 'fps: 1\n');
    fclose(fid);

    if exist('yamlwrite', 'file') == 2
        yp = fileparts(which('yamlwrite'));
        rmpath(yp);
        testCase.TestData.removedPath = yp;
    end

    testCase.TestData.tmpDir = tmpDir;
    testCase.TestData.meta = meta;
    testCase.TestData.outVideo = fullfile(tmpDir, 'scaled.avi');
    testCase.TestData.outMeta = fullfile(tmpDir, 'scaled.yaml');
end

function teardownOnce(testCase)
    if isfield(testCase.TestData, 'removedPath')
        addpath(testCase.TestData.removedPath);
    end
    rmdir(testCase.TestData.tmpDir, 's');
end

function testJsonWritten(testCase)
    outMeta = scale_custom_plume(testCase.TestData.meta, ...
                                 testCase.TestData.outVideo, ...
                                 testCase.TestData.outMeta);
    verifyEqual(testCase, outMeta, testCase.TestData.outMeta);
    txt = fileread(outMeta);
    verifyEqual(testCase, txt(1), '{');
end

