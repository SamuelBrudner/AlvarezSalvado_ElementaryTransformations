addpath(genpath('Code'));
try
    [pf, pc] = get_plume_file();
    fprintf('Plume: %s\n', pf);
    fprintf('Duration: %.0f seconds\n', pc.simulation.duration_seconds);
    fprintf('File exists: %s\n', iif(exist(pf,'file'), 'YES ✓', 'NO ✗'));
    
    % Quick Crimaldi test
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('Crimaldi test: SUCCESS ✓ (%d samples)\n', size(out.x,1));
    
    % Config duration test
    out2 = navigation_model_vec('config', 'Crimaldi', 0, 1);
    fprintf('300s test: SUCCESS ✓ (%d samples = %.0f seconds)\n', ...
            size(out2.x,1), size(out2.x,1)/15);
catch ME
    fprintf('ERROR: %s\n', ME.message);
end
function r=iif(c,t,f), if c, r=t; else, r=f; end, end
