function tests = test_load_plume_video_logging
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    addpath(fullfile(pwd, 'Code'));
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir, 'tiny.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,3,1)));
    writeVideo(vw, uint8(255*ones(2,3,1)));
    close(vw);
    testCase.TestData.video = fullfile(tmpDir, 'tiny.avi');
    testCase.TestData.tmpDir = tmpDir;
end

function teardownOnce(testCase)
    rmdir(testCase.TestData.tmpDir, 's');
end

function testLogsDimensions(testCase)
    out = evalc('load_plume_video(testCase.TestData.video, 1, 1)');
    hasDims = ~isempty(regexp(out, '2\s*x\s*3', 'once'));
    verifyTrue(testCase, hasDims, 'Output should include frame dimensions');
end

function testWarnLargeVideo(testCase)
    stubDir = fullfile(testCase.TestData.tmpDir, 'stub');
    mkdir(stubDir);
    fid = fopen(fullfile(stubDir, 'VideoReader.m'), 'w');
    fprintf(fid, ['classdef VideoReader\n', ...
        'properties\n', ...
        '    Height = 5000;\n', ...
        '    Width = 5000;\n', ...
        '    FrameRate = 30;\n', ...
        '    Duration = 1000;\n', ...
        'end\n', ...
        'methods\n', ...
        '    function obj = VideoReader(~)\n', ...
        '    end\n', ...
        '    function tf = hasFrame(~)\n', ...
        '        tf = false;\n', ...
        '    end\n', ...
        '    function frame = readFrame(~)\n', ...
        '        frame = uint8([]);\n', ...
        '    end\n', ...
        'end\n', ...
        'end']);
    fclose(fid);
    addpath(stubDir);
    c = onCleanup(@() rmpath(stubDir));

    lastwarn('');
    plume = load_plume_video('dummy.avi', 1, 1);
    [~, id] = lastwarn;
    verifyEqual(testCase, id, 'load_plume_video:MemoryExceeded');
    verifyEmpty(testCase, plume.data);
    delete(c);
end
