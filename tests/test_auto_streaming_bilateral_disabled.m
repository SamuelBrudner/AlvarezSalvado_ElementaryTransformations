function tests = test_auto_streaming_bilateral_disabled
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function teardownEach(~)
    unsetenv('SLURM_JOB_ID');
end

function testNoStreamingError(~)
    tmpDir = tempname;
    mkdir(tmpDir);
    vw = VideoWriter(fullfile(tmpDir,'dummy.avi'));
    open(vw);
    writeVideo(vw,uint8(zeros(2,2,1)));
    close(vw);

    setenv('SLURM_JOB_ID','123');
    cfg.plume_video = fullfile(tmpDir,'dummy.avi');
    cfg.px_per_mm = 1;
    cfg.frame_rate = 1;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    cfg.bilateral = true;

    try
        run_navigation_cfg(cfg);
        assert(true);
    catch ME
        assert(~strcmp(ME.identifier,'run_navigation_cfg:BilateralStreamingUnsupported'), ...
            'BilateralStreamingUnsupported should not be thrown');
    end

    rmdir(tmpDir,'s');
end
