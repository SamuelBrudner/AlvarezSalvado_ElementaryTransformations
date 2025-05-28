function tests = test_process_smoke_video
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    addpath(fullfile(pwd, 'Code'));
end

function testHelpContainsExample(~)
    txt = help('process_smoke_video');
    assert(~isempty(txt), 'Help text is empty');
    assert(contains(lower(txt), 'get_intensities_from_video_via_matlab'), ...
        'Help text missing Python example call');
    assert(contains(txt, 'process_smoke_video:LoadPathsFailed'), ...
        'Help text missing error identifier');
end
