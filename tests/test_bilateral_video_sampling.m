function tests = test_bilateral_video_sampling
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd,'Code'));
end

function testSampling(~)
    plume.data = zeros(1,3,1);
    plume.data(1,1,1) = 2;
    plume.data(1,2,1) = 1;
    plume.data(1,3,1) = 3;
    plume.px_per_mm = 1;
    plume.frame_rate = 1;
    params.turnbase = 0;
    params.tsigma = 0;
    params.kup = 0;
    params.kdown = 0;
    params.tmodON = 0;
    params.tmodOFF = 0;
    params.vmodON = 0;
    params.vmodOFF = 0;
    params.vbase = 0;
    params.kbil = 0;
    params.L = 0.1;

    rng(0);
    out = Elifenavmodel_bilateral(1,'video',0,1,plume,params);
    heading0 = out.theta(1) + 90;
    lx = params.L * cosd(heading0);
    ly = params.L * sind(heading0);
    rx = params.L * cosd(heading0 - 180);
    ry = params.L * sind(heading0 - 180);
    xind = round(10*0*plume.px_per_mm)+1;
    yind = round(-10*0*plume.px_per_mm)+1;
    xLind = round(10*lx*plume.px_per_mm)+1;
    yLind = round(-10*ly*plume.px_per_mm)+1;
    xRind = round(10*rx*plume.px_per_mm)+1;
    yRind = round(-10*ry*plume.px_per_mm)+1;
    expOdor = 0;
    if xind >=1 && xind <= size(plume.data,2) && yind >=1 && yind <= size(plume.data,1)
        expOdor = plume.data(yind,xind,1);
    end
    expL = 0;
    if xLind >=1 && xLind <= size(plume.data,2) && yLind >=1 && yLind <= size(plume.data,1)
        expL = plume.data(yLind,xLind,1);
    end
    expR = 0;
    if xRind >=1 && xRind <= size(plume.data,2) && yRind >=1 && yRind <= size(plume.data,1)
        expR = plume.data(yRind,xRind,1);
    end
    assert(abs(out.odor(1) - expOdor) < 1e-12);
    assert(abs(out.odorL(1) - expL) < 1e-12);
    assert(abs(out.odorR(1) - expR) < 1e-12);
end
