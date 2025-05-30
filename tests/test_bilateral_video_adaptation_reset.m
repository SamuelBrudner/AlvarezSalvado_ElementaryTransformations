function tests = test_bilateral_video_adaptation_reset
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
    params.kbil = 0;
    out = Elifenavmodel_bilateral(4,'video',0,1,plume,params);
    on1 = out.ON(2);
    on2 = out.ON(4);
    assert(abs(on1 - on2) < 1e-3, 'ON response should reset when video loops');
end
