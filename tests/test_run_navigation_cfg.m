function tests = test_run_navigation_cfg
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testVideoConfig(~)
    cfg.environment = 'video';
    cfg.plume_video = 'video.avi';
    cfg.px_per_mm = 10;
    cfg.frame_rate = 50;
    cfg.plotting = 0;
    cfg.ntrials = 1;
    try
        run_navigation_cfg(cfg);
        assert(true); % if no error
    catch
        assert(false, 'run_navigation_cfg threw an error');
    end
end

function testGaussianConfig(~)
    cfg = load_config(fullfile('tests','sample_config.yaml'));
    try
        run_navigation_cfg(cfg);
        assert(true);
    catch
        assert(false, 'run_navigation_cfg failed on gaussian config');
    end

end

function testBilateralConfig(~)
    cfg = load_config(fullfile('tests','sample_config_bilateral.yaml'));
    try
        run_navigation_cfg(cfg);
        assert(true);
    catch
        assert(false, 'run_navigation_cfg failed on bilateral config');
    end

end
