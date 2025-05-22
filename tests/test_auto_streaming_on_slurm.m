function tests = test_auto_streaming_on_slurm
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function teardownEach(~)
    unsetenv('SLURM_JOB_ID');
end

function testStreamingEnabledByDefault(~)
    % Create a temporary video file
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'dummy.avi'));
    open(vw);
    writeVideo(vw, uint8(zeros(2,2,1)));
    close(vw);

    setenv('SLURM_JOB_ID','123');
    cfg.environment = 'video';
    cfg.plume_video = fullfile(tmpDir,'dummy.avi');
    cfg.px_per_mm = 1;
    cfg.frame_rate = 1;
    cfg.plotting = 0;
    cfg.ntrials = 1;

    try
        run_navigation_cfg(cfg);
        assert(false, 'Expected streaming error not thrown');
    catch ME
        assert(contains(ME.message,'navigation_model_vec_stream not yet implemented'), ...
               'Unexpected error message');
    end

    rmdir(tmpDir,'s');
end
