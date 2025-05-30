function tests = test_bilateral_video_ws_param
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testWsParam(~)
    plume.data = zeros(1,1,1);
    plume.px_per_mm = 1;
    plume.frame_rate = 1;
    params.turnbase = 0;
    params.tsigma = 0;
    params.kup = 0;
    params.kdown = 1;
    params.tmodON = 0;
    params.tmodOFF = 0;
    params.vmodON = 0;
    params.vmodOFF = 0;
    params.vbase = 0;
    params.kbil = 0;
    params.ws = 1;

    rng(0);
    out1 = Elifenavmodel_bilateral(2,'video',0,1,plume,params);
    params = rmfield(params,'ws');
    rng(0);
    out2 = Elifenavmodel_bilateral(2,'video',0,1,plume,params);
    assert(abs(out1.theta(end) - out2.theta(end)) > 1e-9, 'ws parameter should influence heading');
end
