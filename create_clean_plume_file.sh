#!/bin/bash
# create_clean_plume_file.sh - Create a clean get_plume_file.m from scratch

echo "=== Creating clean get_plume_file.m ==="
echo ""

# Backup current file if it exists
if [ -f "Code/get_plume_file.m" ]; then
    mv Code/get_plume_file.m Code/get_plume_file.m.corrupted_$(date +%H%M%S)
    echo "✓ Backed up corrupted file"
fi

# Create new file line by line to avoid encoding issues
{
echo 'function [plume_file, plume_config] = get_plume_file()'
echo '%GET_PLUME_FILE Return plume HDF5 filename from configuration'
echo '%   STRICT VERSION - Fails if config not found'
echo ''
echo 'config_path = getenv('"'"'PLUME_CONFIG'"'"');'
echo 'if isempty(config_path)'
echo '    % Default to plumes directory'
echo '    config_path = fullfile(fileparts(mfilename('"'"'fullpath'"'"')), '"'"'..'"'"', '"'"'configs'"'"', '"'"'plumes'"'"', '"'"'crimaldi_10cms_bounded.json'"'"');'
echo 'end'
echo ''
echo 'fprintf('"'"'Loading plume config from %s\n'"'"', config_path);'
echo ''
echo '% FAIL if config doesn'"'"'t exist'
echo 'if ~exist(config_path, '"'"'file'"'"')'
echo '    error('"'"'PLUME_CONFIG_ERROR: Config file not found: %s'"'"', config_path);'
echo 'end'
echo ''
echo '% Initialize defaults'
echo 'plume_config = struct();'
echo 'plume_config.mm_per_pixel = 0.74;'
echo 'plume_config.pixel_scale = 0.74;'
echo 'plume_config.frame_rate = 15;'
echo 'plume_config.time_scale_50hz = 15/50;'
echo 'plume_config.time_scale_15hz = 1.0;'
echo 'plume_config.plume_xlims = [1, 216];'
echo 'plume_config.plume_ylims = [1, 406];'
echo 'plume_config.dataset_name = '"'"'/dataset2'"'"';'
echo ''
echo 'try'
echo '    cfg = jsondecode(fileread(config_path));'
echo 'catch err'
echo '    error('"'"'PLUME_CONFIG_ERROR: Failed to parse JSON: %s'"'"', err.message);'
echo 'end'
echo ''
echo '% Extract plume file path'
echo 'if isfield(cfg,'"'"'plume_file'"'"')'
echo '    % Old format'
echo '    plume_file = cfg.plume_file;'
echo '    if isfield(cfg,'"'"'plume_path'"'"') && ~isempty(cfg.plume_path)'
echo '        plume_file = fullfile(cfg.plume_path, plume_file);'
echo '    end'
echo 'elseif isfield(cfg, '"'"'data_path'"'"') && isfield(cfg.data_path, '"'"'path'"'"')'
echo '    % New format'
echo '    plume_file = cfg.data_path.path;'
echo '    '
echo '    % Update config struct'
echo '    if isfield(cfg, '"'"'spatial'"'"')'
echo '        plume_config.mm_per_pixel = cfg.spatial.mm_per_pixel;'
echo '        plume_config.pixel_scale = cfg.spatial.mm_per_pixel;'
echo '        if isfield(cfg.spatial, '"'"'resolution'"'"')'
echo '            plume_config.plume_xlims = [1, cfg.spatial.resolution.width];'
echo '            plume_config.plume_ylims = [1, cfg.spatial.resolution.height];'
echo '        end'
echo '    end'
echo '    '
echo '    if isfield(cfg, '"'"'temporal'"'"')'
echo '        plume_config.frame_rate = cfg.temporal.frame_rate;'
echo '        plume_config.time_scale_50hz = cfg.temporal.frame_rate / 50.0;'
echo '        plume_config.time_scale_15hz = cfg.temporal.frame_rate / 15.0;'
echo '    end'
echo '    '
echo '    if isfield(cfg.data_path, '"'"'dataset_name'"'"')'
echo '        plume_config.dataset_name = cfg.data_path.dataset_name;'
echo '    end'
echo 'else'
echo '    error('"'"'PLUME_CONFIG_ERROR: Invalid config format - missing plume_file or data_path.path'"'"');'
echo 'end'
echo ''
echo '% Load simulation parameters'
echo 'if isfield(cfg, '"'"'simulation'"'"')'
echo '    if isfield(cfg.simulation, '"'"'duration_seconds'"'"')'
echo '        plume_config.simulation.duration_seconds = cfg.simulation.duration_seconds;'
echo '    end'
echo '    if isfield(cfg.simulation, '"'"'comment'"'"')'
echo '        plume_config.simulation.comment = cfg.simulation.comment;'
echo '    end'
echo 'end'
echo ''
echo '% Handle relative paths'
echo 'if ~isempty(plume_file) && plume_file(1) ~= '"'"'/'"'"''
echo '    project_root = fileparts(fileparts(mfilename('"'"'fullpath'"'"')));'
echo '    plume_file = fullfile(project_root, plume_file);'
echo 'end'
echo ''
echo '% Return just filename if only one output requested'
echo 'if nargout < 2'
echo '    clear plume_config;'
echo 'end'
echo ''
echo 'end'
} > Code/get_plume_file.m

echo "✓ Created Code/get_plume_file.m"
echo ""

# Verify it's valid ASCII
echo "File check:"
file Code/get_plume_file.m
echo ""

# Quick syntax check
echo "Testing MATLAB syntax..."
matlab -nodisplay -nosplash -nojvm -r "try; pcode('Code/get_plume_file.m'); fprintf('✓ Valid MATLAB syntax\n'); catch ME; fprintf('✗ Syntax error: %s\n', ME.message); end; exit" 2>&1 | tail -5

echo ""
echo "Now test with: ./run_test.sh"