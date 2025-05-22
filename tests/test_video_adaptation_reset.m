function tests = test_video_adaptation_reset
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testReset(~)
    plume.data = zeros(1,1,2);
    plume.data(1,1,1) = 1;
    plume.px_per_mm = 1;
    plume.frame_rate = 1;
    out = navigation_model_vec(4,'video',0,1,plume);
    on1 = out.ON(2); % response to first odor frame
    on2 = out.ON(4); % response after loop restart
    assert(abs(on1 - on2) < 1e-3, 'ON response should reset when video loops');
end
