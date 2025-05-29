function tests = test_video_dt_scaling
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testDtScaling(~)
    plume.data = ones(1,1,1);
    plume.px_per_mm = 1;
    params.turnbase = 0;
    params.tsigma = 0;
    params.kup = 0;
    params.kdown = 0;
    params.tmodON = 0;
    params.tmodOFF = 0;
    params.vmodON = 0;
    params.vmodOFF = 0;
    params.vbase = 10;

    plume.frame_rate = 5;
    rng(0);
    out1 = navigation_model_vec(plume.frame_rate,'video',0,1,plume,params);
    dist1 = norm([out1.x(end) out1.y(end)]);

    plume.frame_rate = 10;
    rng(0);
    out2 = navigation_model_vec(plume.frame_rate,'video',0,1,plume,params);
    dist2 = norm([out2.x(end) out2.y(end)]);

    assert(abs(dist1 - dist2) < 1e-12, 'displacement per second should be identical');
end
