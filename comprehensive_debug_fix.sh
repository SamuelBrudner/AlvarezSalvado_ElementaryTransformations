#!/bin/bash
# comprehensive_debug_fix.sh - Debug and fix all issues

echo "=== Comprehensive Debug and Fix ==="
echo ""

# Step 1: Check current status
echo "STEP 1: Current Status"
echo "---------------------"
echo "PWD: $(pwd)"
echo "Real path: $(realpath .)"
echo ""

# Step 2: Check files exist
echo "STEP 2: File Check"
echo "-----------------"
echo "Config exists: $(ls -la configs/plumes/crimaldi_10cms_bounded.json 2>&1 | head -1)"
echo "Plume file exists: $(ls -la data/plumes/10302017_10cms_bounded.hdf5 2>&1 | head -1)"
echo "get_plume_file.m exists: $(ls -la Code/get_plume_file.m 2>&1 | head -1)"
echo ""

# Step 3: Test basic functionality
echo "STEP 3: Basic Function Test"
echo "--------------------------"
cat > test_basic.m << 'EOF'
try
    addpath(genpath('Code'));
    fprintf('✓ Code path added\n');
    
    % Test Gaussian (no plume needed)
    out = navigation_model_vec(100, 'gaussian', 0, 1);
    fprintf('✓ Gaussian works\n');
catch ME
    fprintf('✗ Basic test failed: %s\n', ME.message);
end
EOF

matlab -nodisplay -nosplash -r "test_basic; exit" 2>&1 | grep -E "(✓|✗|Error)"
rm -f test_basic.m
echo ""

# Step 4: Apply comprehensive fix
echo "STEP 4: Applying Comprehensive Fix"
echo "---------------------------------"

# Create a completely new get_plume_file.m that works
cat > Code/get_plume_file_new.m << 'EOF'
function [plume_file, plume_config] = get_plume_file()
%GET_PLUME_FILE Return plume HDF5 filename from configuration

% Simple approach - use environment or defaults
config_path = getenv('PLUME_CONFIG');
if isempty(config_path)
    % Use explicit path for HPC
    if exist('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', 'dir')
        config_path = '/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/configs/plumes/crimaldi_10cms_bounded.json';
    else
        % Fallback to relative path
        config_path = 'configs/plumes/crimaldi_10cms_bounded.json';
    end
end

fprintf('Loading plume config from %s\n', config_path);

% Initialize defaults
plume_config = struct();
plume_config.mm_per_pixel = 0.74;
plume_config.pixel_scale = 0.74;
plume_config.frame_rate = 15;
plume_config.time_scale_50hz = 15/50;
plume_config.time_scale_15hz = 1.0;
plume_config.plume_xlims = [1, 216];
plume_config.plume_ylims = [1, 406];
plume_config.dataset_name = '/dataset2';

% Load config if it exists
if exist(config_path, 'file')
    try
        cfg = jsondecode(fileread(config_path));
        
        % Extract plume file path
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'path')
            plume_file = cfg.data_path.path;
        elseif isfield(cfg, 'plume_file')
            plume_file = cfg.plume_file;
        else
            plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
        end
        
        % Update config parameters
        if isfield(cfg, 'spatial')
            if isfield(cfg.spatial, 'mm_per_pixel')
                plume_config.mm_per_pixel = cfg.spatial.mm_per_pixel;
                plume_config.pixel_scale = cfg.spatial.mm_per_pixel;
            end
            if isfield(cfg.spatial, 'resolution')
                plume_config.plume_xlims = [1, cfg.spatial.resolution.width];
                plume_config.plume_ylims = [1, cfg.spatial.resolution.height];
            end
        end
        
        if isfield(cfg, 'temporal')
            if isfield(cfg.temporal, 'frame_rate')
                plume_config.frame_rate = cfg.temporal.frame_rate;
                plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;
                plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;
            end
        end
        
        if isfield(cfg, 'data_path') && isfield(cfg.data_path, 'dataset_name')
            plume_config.dataset_name = cfg.data_path.dataset_name;
        end
        
        if isfield(cfg, 'simulation')
            if isfield(cfg.simulation, 'duration_seconds')
                plume_config.simulation.duration_seconds = cfg.simulation.duration_seconds;
            end
        end
        
    catch err
        warning('Could not parse config file: %s', err.message);
        plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
    end
else
    warning('Config file not found, using defaults');
    plume_file = 'data/plumes/10302017_10cms_bounded.hdf5';
end

% Make path absolute if needed
if exist(plume_file, 'file')
    % File exists as is
elseif plume_file(1) ~= '/'
    % Try with explicit base path
    if exist('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', 'dir')
        plume_file = fullfile('/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations', plume_file);
    end
end

% Environment override
env_plume = getenv('MATLAB_PLUME_FILE');
if ~isempty(env_plume)
    plume_file = env_plume;
end

if nargout < 2
    clear plume_config;
end

end
EOF

# Backup and install
if [ -f "Code/get_plume_file.m" ]; then
    cp Code/get_plume_file.m Code/get_plume_file.m.backup_comprehensive
fi
cp Code/get_plume_file_new.m Code/get_plume_file.m
echo "✓ Installed new get_plume_file.m"

# Update config
sed -i 's|"path": "data/plumes/10302017_10cms_bounded.hdf5"|"path": "/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"|' configs/plumes/crimaldi_10cms_bounded.json 2>/dev/null || echo "Config update skipped"

echo ""
echo "STEP 5: Final Test"
echo "-----------------"

cat > final_test.m << 'EOF'
addpath(genpath('Code'));

fprintf('\n=== Final Comprehensive Test ===\n\n');

% Test 1: Config loading
fprintf('1. Config loading test:\n');
try
    [pf, pc] = get_plume_file();
    fprintf('   ✓ Config loaded\n');
    fprintf('   Plume: %s\n', pf);
    if isfield(pc, 'simulation') && isfield(pc.simulation, 'duration_seconds')
        fprintf('   Duration: %.0f seconds\n', pc.simulation.duration_seconds);
    else
        fprintf('   Duration: not set\n');
    end
    fprintf('   File exists: %s\n', iif(exist(pf,'file'), 'YES', 'NO'));
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
    pf = '';
end

% Test 2: Gaussian (always works)
fprintf('\n2. Gaussian test:\n');
try
    out = navigation_model_vec(1000, 'gaussian', 0, 2);
    fprintf('   ✓ Success: %d agents, %d samples\n', size(out.x,2), size(out.x,1));
catch ME
    fprintf('   ✗ Failed: %s\n', ME.message);
end

% Test 3: Crimaldi (if file exists)
if exist(pf, 'file')
    fprintf('\n3. Crimaldi test:\n');
    try
        out = navigation_model_vec(100, 'Crimaldi', 0, 1);
        fprintf('   ✓ Success: %d samples\n', size(out.x,1));
    catch ME
        fprintf('   ✗ Failed: %s\n', ME.message);
    end
    
    % Test 4: Config duration
    fprintf('\n4. 300s duration test:\n');
    try
        out = navigation_model_vec('config', 'Crimaldi', 0, 1);
        fprintf('   ✓ Success: %d samples = %.1f seconds\n', size(out.x,1), size(out.x,1)/15);
    catch ME
        fprintf('   ✗ Failed: %s\n', ME.message);
    end
else
    fprintf('\n3-4. Skipping Crimaldi tests (plume file not found)\n');
end

fprintf('\n=== Test Complete ===\n');

function r=iif(c,t,f)
    if c, r=t; else, r=f; end
end
EOF

matlab -nodisplay -nosplash -r "final_test; exit" 2>&1 | tail -35
rm -f final_test.m

echo ""
echo "=== SUMMARY ==="
echo ""
echo "If tests passed:"
echo "  - You can run SLURM jobs with: navigation_model_vec('config', 'Crimaldi', 0, 10)"
echo ""
echo "If Crimaldi failed but Gaussian worked:"
echo "  - Use 'gaussian' instead of 'Crimaldi' in your simulations"
echo ""
echo "To restore original:"
echo "  cp Code/get_plume_file.m.backup_comprehensive Code/get_plume_file.m"