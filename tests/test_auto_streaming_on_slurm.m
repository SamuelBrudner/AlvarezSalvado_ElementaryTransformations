function tests = test_auto_streaming_on_slurm
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function teardownEach(~)
    unsetenv('SLURM_JOB_ID');
end

function testStreamingEnabledByDefault(testCase)
    % Create a temporary video file
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'dummy.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);

    % Stub navigation_model_vec_stream to detect streaming usage
    stubDir = fullfile(tmpDir,'stub');
    mkdir(stubDir);
    fid = fopen(fullfile(stubDir,'navigation_model_vec_stream.m'),'w');
    fprintf(fid, ['function out = navigation_model_vec_stream(varargin)\n', ...
        'global STREAM_CALLED; STREAM_CALLED = true; out = struct();\n']);
    fclose(fid);
    addpath(stubDir);
    c = onCleanup(@() rmpath(stubDir));

    setenv('SLURM_JOB_ID','123');
    cfg.environment = 'video';
    cfg.plume_video = fullfile(tmpDir,'dummy.avi');
    cfg.px_per_mm = 1;
    cfg.frame_rate = 1;
    cfg.plotting = 0;
    cfg.ntrials = 1;

    global STREAM_CALLED; STREAM_CALLED = false;
    run_navigation_cfg(cfg);
    verifyTrue(testCase, STREAM_CALLED, 'Streaming was not auto-enabled');

    rmdir(tmpDir,'s');
end
