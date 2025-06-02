addpath('Code');

fprintf('=== Testing Complete Unified System ===\n\n');

% Test 1: Crimaldi simulation
fprintf('1. Testing Crimaldi simulation:\n');
cfg_crim = load_config('configs/batch_crimaldi.yaml');
cfg_crim.ntrials = 1;
cfg_crim.plotting = 0;
cfg_crim.triallength = 100; % Short test

try
    out_crim = run_navigation_cfg(cfg_crim);
    fprintf('   ✓ Crimaldi simulation completed\n');
    fprintf('   Final position: (%.2f, %.2f)\n', out_crim.x(end), out_crim.y(end));
catch ME
    fprintf('   ✗ ERROR: %s\n', ME.message);
    fprintf('   Location: %s line %d\n', ME.stack(1).name, ME.stack(1).line);
end

% Test 2: Smoke simulation  
fprintf('\n2. Testing Smoke simulation:\n');
cfg_smoke = load_config('configs/batch_smoke_hdf5.yaml');
cfg_smoke.ntrials = 1;
cfg_smoke.plotting = 0;
cfg_smoke.triallength = 100; % Short test

try
    out_smoke = run_navigation_cfg(cfg_smoke);
    fprintf('   ✓ Smoke simulation completed\n');
    fprintf('   Final position: (%.2f, %.2f)\n', out_smoke.x(end), out_smoke.y(end));
catch ME
    fprintf('   ✗ ERROR: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('   Location: %s line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
end

exit
