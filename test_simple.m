% test_simple.m - Simple test that adds paths manually

fprintf('\n=== Simple Path Test ===\n\n');

% Add Code to path manually
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir)
    this_dir = pwd;
end
addpath(genpath(fullfile(this_dir, 'Code')));

% Now test
fprintf('1. Current directory: %s\n', pwd);
fprintf('2. load_paths exists: %s\n', iif(exist('load_paths', 'file')==2, 'YES', 'NO'));

% Try to load paths
fprintf('\n3. Loading paths:\n');
try
    paths = load_paths();
    fprintf('   ✓ Success!\n');
    fprintf('   Project root: %s\n', paths.project_root);
    fprintf('   Correct path: %s\n', iif(contains(paths.project_root, '/home/snb6'), 'YES', 'NO'));
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
end

% Test get_plume_file
fprintf('\n4. Testing get_plume_file:\n');
try
    [pf, ~] = get_plume_file();
    fprintf('   ✓ Success!\n');
    fprintf('   Plume file: %s\n', pf);
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
end

% Quick simulation test
fprintf('\n5. Testing simulation:\n');
try
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('   ✓ SUCCESS! Generated %d samples\n', size(out.x, 1));
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
end

fprintf('\n=== Test Complete ===\n');

function r = iif(c,t,f)
    if c, r=t; else, r=f; end
end
