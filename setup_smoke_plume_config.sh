#!/bin/bash
# setup_smoke_plume_config.sh - Complete setup for smoke plume simulations
#
# This script:
# 1. Creates smoke plume config from template
# 2. Updates paths.json
# 3. Optionally analyzes the HDF5 file for dimensions and statistics
# 4. Updates config with all values
# 5. Creates visualization and test scripts

set -euo pipefail

echo "=== Complete Smoke Plume Configuration Setup ==="
echo ""

# Store the project root directory
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration files
CONFIG_DIR="configs/plumes"
TEMPLATE_CONFIG="$CONFIG_DIR/crimaldi_10cms_bounded.json"
SMOKE_CONFIG="$CONFIG_DIR/smoke_1a_backgroundsubtracted.json"
PATHS_CONFIG="configs/paths.json"

# CRITICAL: These are the CORRECT parameters for the smoke plume
MM_PER_PIXEL=0.15299877600979192  # ~0.153 mm/pixel
FPS=60.0                           # 60 Hz sampling rate

# Check if template exists
if [ ! -f "$TEMPLATE_CONFIG" ]; then
    echo "ERROR: Template config not found at: $TEMPLATE_CONFIG"
    exit 1
fi

# Check if smoke config exists and has wrong parameters
if [ -f "$SMOKE_CONFIG" ]; then
    echo "⚠️  Existing smoke config found. Checking parameters..."
    python3 << 'EOF'
import json
with open('configs/plumes/smoke_1a_backgroundsubtracted.json', 'r') as f:
    config = json.load(f)
    
mm_per_pixel = config.get('spatial', {}).get('mm_per_pixel', 0)
fps = config.get('temporal', {}).get('frame_rate', 0)

if abs(mm_per_pixel - 0.74) < 0.01 or abs(fps - 15) < 1:
    print("  ⚠️  WARNING: Existing config has Crimaldi plume parameters!")
    print(f"     Current: {mm_per_pixel} mm/px, {fps} Hz")
    print(f"     These will be corrected to: 0.153 mm/px, 60 Hz")
EOF
fi

# Step 1: Get HDF5 path
echo ""
echo "Step 1: Setting smoke plume parameters..."

# Try to read HDF5 path from existing config, or use default
if [ -f "$SMOKE_CONFIG" ]; then
    echo "  Reading HDF5 path from existing smoke config..."
    read SMOKE_HDF5 DATASET_NAME < <(python3 << 'EOF'
import json
with open('configs/plumes/smoke_1a_backgroundsubtracted.json', 'r') as f:
    config = json.load(f)
hdf5_path = config.get('data_path', {}).get('path', 'data/plumes/smoke_1a_orig_backgroundsubtracted_rotated.h5')
dataset_name = config.get('data_path', {}).get('dataset_name', '/dataset2')
print(hdf5_path, dataset_name)
EOF
)
else
    echo "  Using default paths..."
    SMOKE_HDF5="/vast/palmer/scratch/emonet/snb6/plume/smoke_1a_orig_backgroundsubtracted_rotated.h5"
    DATASET_NAME="/dataset2"
fi

# Convert relative path to absolute if needed
if [[ ! "$SMOKE_HDF5" = /* ]]; then
    SMOKE_HDF5="$PROJECT_ROOT/$SMOKE_HDF5"
fi

echo "  ⚠️  Forcing correct parameters for smoke plume:"
echo "     mm_per_pixel: $MM_PER_PIXEL (NOT 0.74)"
echo "     fps: $FPS Hz (NOT 15 Hz)"
echo ""
echo "  Parameters:"
echo "     mm_per_pixel: $MM_PER_PIXEL"
echo "     fps: $FPS Hz"
echo "     hdf5_path: $SMOKE_HDF5"
echo "     dataset_name: $DATASET_NAME"

# Check if HDF5 file exists
if [ ! -f "$SMOKE_HDF5" ]; then
    echo ""
    echo "ERROR: Smoke plume HDF5 not found at: $SMOKE_HDF5"
    echo "Please ensure the file exists or update the path"
    exit 1
fi

# Get file size
SMOKE_SIZE=$(stat -f%z "$SMOKE_HDF5" 2>/dev/null || stat -c%s "$SMOKE_HDF5" 2>/dev/null || echo "0")
if command -v bc &> /dev/null; then
    SMOKE_SIZE_GB=$(echo "scale=2; $SMOKE_SIZE / 1024 / 1024 / 1024" | bc)
else
    SMOKE_SIZE_GB=$(( SMOKE_SIZE / 1024 / 1024 / 1024 ))
fi

echo ""
echo "HDF5 file info:"
echo "  Path: $SMOKE_HDF5"
echo "  Size: ${SMOKE_SIZE_GB} GB"

if [ "$SMOKE_SIZE" -gt 10737418240 ]; then  # > 10 GB
    echo "  ⚠️  Large file detected! Analysis may take several minutes."
fi

# Step 2: Ask user about analysis
echo ""
echo "The analysis step samples the HDF5 file to determine arena dimensions and source position."
echo "For large files on network/scratch storage, this can be slow."
echo ""
read -p "Run analysis? (y/n/quick) [y]: " RUN_ANALYSIS
RUN_ANALYSIS=${RUN_ANALYSIS:-y}

# Step 3: Create config from template
echo ""
echo "Step 3: Creating smoke plume config from template..."
if [ -f "$SMOKE_CONFIG" ]; then
    echo "  Backing up existing config..."
    cp "$SMOKE_CONFIG" "${SMOKE_CONFIG}.backup_$(date +%Y%m%d_%H%M%S)"
fi

python3 << EOF
import json

# Load template config
with open('$TEMPLATE_CONFIG', 'r') as f:
    config = json.load(f)

# Update basic fields
config['plume_id'] = 'smoke_1a_backgroundsubtracted'
config['description'] = 'Smoke plume 1a with background subtraction and rotation (60 Hz, 0.153 mm/px)'

# Update data path
config['data_path']['path'] = '$SMOKE_HDF5'
config['data_path']['dataset_name'] = '$DATASET_NAME'

# CRITICAL: Force correct parameters for smoke plume
config['spatial']['mm_per_pixel'] = $MM_PER_PIXEL  # 0.153, NOT 0.74
config['temporal']['frame_rate'] = $FPS             # 60 Hz, NOT 15 Hz

# These will be updated after analysis, but set reasonable defaults
config['spatial']['resolution']['width'] = 1024   # Will be updated
config['spatial']['resolution']['height'] = 1024  # Will be updated

# Save initial config
with open('$SMOKE_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)

print("  ✓ Created smoke config with CORRECT parameters:")
print(f"    - mm_per_pixel: {config['spatial']['mm_per_pixel']} mm/px")
print(f"    - frame_rate: {config['temporal']['frame_rate']} Hz")
EOF

# Step 4: Update paths.json
echo ""
echo "Step 4: Updating paths.json..."
cp "$PATHS_CONFIG" "${PATHS_CONFIG}.backup_$(date +%Y%m%d_%H%M%S)"

FULL_SMOKE_CONFIG=$(realpath "$SMOKE_CONFIG")
python3 << EOF
import json

with open('$PATHS_CONFIG', 'r') as f:
    paths = json.load(f)

paths['plume_config'] = '$FULL_SMOKE_CONFIG'

with open('$PATHS_CONFIG', 'w') as f:
    json.dump(paths, f, indent=2)

print("  ✓ Updated plume_config path")
EOF

# Step 5: Run analysis or use defaults
if [[ "$RUN_ANALYSIS" == "n" ]]; then
    echo ""
    echo "Step 5: Skipping analysis (using defaults)..."
    
    # Set default values
    width=1024
    height=1024
    frames=36000  # 10 minutes at 60 Hz
    dataset="/dataset2"
    data_min=0.0
    data_max=1.0
    data_mean=0.1
    data_std=0.1
    source_x_cm=0.0
    source_y_cm=0.0
    temporal_scale=4.0
    spatial_scale=0.207
    beta_suggestion=0.01
    normalized=1
    
    # Calculate arena dimensions
    if command -v bc &> /dev/null; then
        arena_width_cm=$(echo "scale=1; $width * $MM_PER_PIXEL / 10" | bc)
        arena_height_cm=$(echo "scale=1; $height * $MM_PER_PIXEL / 10" | bc)
    else
        arena_width_cm=$(awk "BEGIN {printf \"%.1f\", $width * $MM_PER_PIXEL / 10}")
        arena_height_cm=$(awk "BEGIN {printf \"%.1f\", $height * $MM_PER_PIXEL / 10}")
    fi
    
else
    echo ""
    echo "Step 5: Running plume analysis..."
    
    # Set analysis parameters
    if [[ "$RUN_ANALYSIS" == "quick" ]]; then
        echo "  Mode: QUICK (sampling 10 frames)..."
        N_SAMPLE_FRAMES=10
        TIMEOUT_SECONDS=300  # 5 minutes for quick
    else
        echo "  Mode: FULL (sampling 100 frames)..."
        N_SAMPLE_FRAMES=100
        TIMEOUT_SECONDS=600  # 10 minutes for full
    fi
    
    # Check MATLAB
    if ! command -v matlab &> /dev/null; then
        echo "ERROR: MATLAB not found in PATH"
        echo "Please load MATLAB module or ensure it's available"
        exit 1
    fi
    
    # Create temp directory
    TEMP_DIR="$PROJECT_ROOT/temp_matlab_$$"
    mkdir -p "$TEMP_DIR"
    
    # Create wrapper script
    cat > "$TEMP_DIR/run_analysis.sh" << WRAPPER_EOF
#!/bin/bash
MATLAB_SCRIPT="\$1"

echo "Running MATLAB analysis with ${TIMEOUT_SECONDS}s timeout..."
echo "  Mode: ${N_SAMPLE_FRAMES} frames"

# Run MATLAB with timeout
if command -v timeout &> /dev/null; then
    timeout ${TIMEOUT_SECONDS}s matlab -nodisplay -nosplash -r "run('\${MATLAB_SCRIPT}')" 2>&1
    MATLAB_EXIT=\$?
else
    perl -e "alarm ${TIMEOUT_SECONDS}; exec @ARGV" matlab -nodisplay -nosplash -r "run('\${MATLAB_SCRIPT}')" 2>&1
    MATLAB_EXIT=\$?
fi

if [ \$MATLAB_EXIT -eq 124 ] || [ \$MATLAB_EXIT -eq 142 ]; then
    echo ""
    echo "ERROR: MATLAB analysis timed out after ${TIMEOUT_SECONDS} seconds!"
    echo "Try using 'quick' mode or 'n' to skip analysis"
    exit 1
elif [ \$MATLAB_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: MATLAB exited with code \$MATLAB_EXIT"
    exit \$MATLAB_EXIT
fi
WRAPPER_EOF
    
    chmod +x "$TEMP_DIR/run_analysis.sh"
    
    # Create MATLAB analysis script
    cat > "$TEMP_DIR/analyze_plume.m" << MATLAB_EOF
% Smoke plume analysis script
fprintf('\\n=== MATLAB STARTED SUCCESSFULLY ===\\n');
fprintf('Time: %s\\n', datestr(now));
fprintf('MATLAB version: %s\\n', version);
fprintf('Working directory: %s\\n\\n', pwd);

try
    % Change to project root
    cd('$PROJECT_ROOT');
    fprintf('Changed to project root: %s\\n', pwd);
    
    % Check Code directory
    if ~exist('Code', 'dir')
        error('Code directory not found in %s', pwd);
    end
    
    % Add to path
    addpath(genpath('Code'));
    fprintf('Added Code directory to path\\n');
    
    % Parameters from shell script
    mm_per_pixel = $MM_PER_PIXEL;
    fps = $FPS;
    n_sample_frames = $N_SAMPLE_FRAMES;
    
    % File info
    plume_file = '$SMOKE_HDF5';
    dataset_name = '$DATASET_NAME';
    
    fprintf('\\nConfiguration:\\n');
    fprintf('  mm_per_pixel: %.6f\\n', mm_per_pixel);
    fprintf('  fps: %.1f Hz\\n', fps);
    fprintf('  HDF5 file: %s\\n', plume_file);
    fprintf('  Dataset: %s\\n', dataset_name);
    fprintf('  Sampling: %d frames\\n', n_sample_frames);
    
    % Check file exists
    if ~exist(plume_file, 'file')
        error('HDF5 file not found: %s', plume_file);
    end
    
    % Get file info
    fprintf('\\nReading HDF5 file info (may take time for large files)...\\n');
    tic;
    info = h5info(plume_file);
    fprintf('  File info retrieved in %.1f seconds\\n', toc);
    
    % Get dataset info
    ds_info = h5info(plume_file, dataset_name);
    width = ds_info.Dataspace.Size(1);
    height = ds_info.Dataspace.Size(2);
    n_frames = ds_info.Dataspace.Size(3);
    
    fprintf('  Dimensions: %d x %d x %d\\n', width, height, n_frames);
    
    % Sample frames
    fprintf('\\nSampling %d frames...\\n', n_sample_frames);
    rng(42);
    sample_indices = sort(randperm(n_frames, min(n_sample_frames, n_frames)));
    
    all_values = [];
    mean_map = zeros(width, height);
    
    fprintf('  Progress: ');
    for i = 1:length(sample_indices)
        if mod(i, max(1, round(n_sample_frames/10))) == 0
            fprintf('%d%% ', round(i/length(sample_indices)*100));
        end
        
        frame = h5read(plume_file, dataset_name, [1 1 sample_indices(i)], [inf inf 1]);
        all_values = [all_values; frame(:)];
        
        if i <= 5  % Use first 5 frames for mean map
            mean_map = mean_map + double(frame);
        end
    end
    fprintf('Done!\\n');
    
    mean_map = mean_map / min(5, length(sample_indices));
    
    % Calculate statistics
    data_min = min(all_values);
    data_max = max(all_values);
    data_mean = mean(all_values);
    data_std = std(all_values);
    
    % Find source position
    [~, max_idx] = max(mean_map(:));
    [max_x, max_y] = ind2sub(size(mean_map), max_idx);
    
    center_x_px = width / 2;
    center_y_px = height / 2;
    source_x_cm = (max_x - center_x_px) * mm_per_pixel / 10;
    source_y_cm = -(max_y - center_y_px) * mm_per_pixel / 10;
    
    % Calculate arena bounds
    arena_width_cm = width * mm_per_pixel / 10;
    arena_height_cm = height * mm_per_pixel / 10;
    
    % Calculate scaling
    temporal_scale = fps / 15.0;  % Relative to Crimaldi at 15 Hz
    spatial_scale = mm_per_pixel / 0.74;  % Relative to Crimaldi at 0.74 mm/px
    
    % Determine beta
    if data_max <= 1.0 && data_min >= 0
        beta_suggestion = 0.01;
    else
        beta_suggestion = data_mean * 0.1;
    end
    
    % Save results
    results_file = fullfile('$TEMP_DIR', 'analysis_results.txt');
    fid = fopen(results_file, 'w');
    fprintf(fid, 'width=%d\\n', width);
    fprintf(fid, 'height=%d\\n', height);
    fprintf(fid, 'frames=%d\\n', n_frames);
    fprintf(fid, 'dataset=%s\\n', dataset_name);
    fprintf(fid, 'data_min=%.6f\\n', data_min);
    fprintf(fid, 'data_max=%.6f\\n', data_max);
    fprintf(fid, 'data_mean=%.6f\\n', data_mean);
    fprintf(fid, 'data_std=%.6f\\n', data_std);
    fprintf(fid, 'source_x_cm=%.3f\\n', source_x_cm);
    fprintf(fid, 'source_y_cm=%.3f\\n', source_y_cm);
    fprintf(fid, 'arena_width_cm=%.3f\\n', arena_width_cm);
    fprintf(fid, 'arena_height_cm=%.3f\\n', arena_height_cm);
    fprintf(fid, 'temporal_scale=%.3f\\n', temporal_scale);
    fprintf(fid, 'spatial_scale=%.3f\\n', spatial_scale);
    fprintf(fid, 'beta_suggestion=%.6f\\n', beta_suggestion);
    fprintf(fid, 'normalized=%d\\n', data_max <= 1.0 && data_min >= 0);
    fclose(fid);
    
    fprintf('\\n✓ Analysis complete\\n');
    exit(0);
    
catch ME
    fprintf('\\nERROR: %s\\n', ME.message);
    exit(1);
end
MATLAB_EOF
    
    # Run analysis
    echo "Starting MATLAB analysis (timeout: ${TIMEOUT_SECONDS}s)..."
    
    if ! "$TEMP_DIR/run_analysis.sh" "$TEMP_DIR/analyze_plume.m" | tee "$TEMP_DIR/matlab_output.log"; then
        echo ""
        echo "ERROR: MATLAB analysis failed!"
        echo "Try running with 'n' to skip analysis"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    # Load results
    if [ ! -f "$TEMP_DIR/analysis_results.txt" ]; then
        echo "ERROR: No results file created"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    source "$TEMP_DIR/analysis_results.txt"
    rm -rf "$TEMP_DIR"
    
    echo "✓ Analysis completed successfully"
fi

# Step 6: Update config with results
echo ""
echo "Step 6: Updating configuration with results..."

python3 << EOF
import json

with open('$SMOKE_CONFIG', 'r') as f:
    config = json.load(f)

# Update dimensions
config['spatial']['resolution']['width'] = $width
config['spatial']['resolution']['height'] = $height
config['spatial']['arena_bounds']['x_min'] = -$arena_width_cm/2
config['spatial']['arena_bounds']['x_max'] = $arena_width_cm/2
config['spatial']['arena_bounds']['y_min'] = -$arena_height_cm/2
config['spatial']['arena_bounds']['y_max'] = $arena_height_cm/2

# Update temporal
config['temporal']['total_frames'] = $frames
config['temporal']['duration'] = $frames / $FPS

# Update source position
config['simulation']['source_position']['x_cm'] = $source_x_cm
config['simulation']['source_position']['y_cm'] = $source_y_cm

# Update agent initialization
init_width = min(20, $arena_width_cm * 0.4)
init_y_start = -$arena_height_cm * 0.4
init_y_end = init_y_start + 5

config['simulation']['agent_initialization']['x_range_cm'] = [-init_width/2, init_width/2]
config['simulation']['agent_initialization']['y_range_cm'] = [init_y_start, init_y_end]

# Add analysis metadata
config['analysis'] = {
    'intensity_range': [$data_min, $data_max],
    'intensity_mean': $data_mean,
    'intensity_std': $data_std,
    'beta_suggestion': $beta_suggestion,
    'data_normalized': bool($normalized),
    'temporal_scale_factor': $temporal_scale,
    'spatial_scale_factor': $spatial_scale,
    'analysis_mode': '$RUN_ANALYSIS',
    'parameter_notes': {
        'time_constants_need_scaling': 'Multiply by %.1f for %d Hz' % ($temporal_scale, $FPS),
        'tau_Aon_scaled': int(490 * $temporal_scale),
        'tau_Aoff_scaled': int(504 * $temporal_scale),
        'tau_ON_scaled': int(36 * $temporal_scale),
        'tau_OFF1_scaled': int(31 * $temporal_scale),
        'tau_OFF2_scaled': int(242 * $temporal_scale)
    }
}

with open('$SMOKE_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)

print("  ✓ Configuration updated")
print("")
print(f"  Arena: {$arena_width_cm:.1f} × {$arena_height_cm:.1f} cm")
print(f"  Source: ({$source_x_cm:.1f}, {$source_y_cm:.1f}) cm")
print(f"  Frames: {$frames} ({$frames/$FPS/60:.1f} minutes at {$FPS:.0f} Hz)")

if '$RUN_ANALYSIS' == 'n':
    print("  ⚠️  Using default values (analysis skipped)")
elif '$RUN_ANALYSIS' == 'quick':
    print("  ⚠️  Based on quick analysis (10 frames)")
EOF

# Step 7: Create visualization script
echo ""
echo "Step 7: Creating scripts..."

cat > visualize_smoke_plume.m << 'MATLAB_EOF'
% visualize_smoke_plume.m - Visualize the configured smoke plume

% Store current directory
original_dir = pwd;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

fprintf('Loading smoke plume configuration...\n');

% Add Code directory to path
if exist('Code', 'dir')
    addpath(genpath('Code'));
end

try
    % Load config
    [plume_file, plume_config] = get_plume_file();
    
    fprintf('\nLoaded: %s\n', plume_config.plume_id);
    fprintf('File: %s\n', plume_file);
    
    % Read sample frames
    total_frames = plume_config.temporal.total_frames;
    frames_to_show = [1, round(total_frames/2), total_frames];
    
    figure('Position', [100 100 1200 400]);
    for i = 1:3
        subplot(1, 3, i);
        frame = h5read(plume_file, plume_config.data_path.dataset_name, ...
                       [1 1 frames_to_show(i)], [inf inf 1]);
        imagesc(frame');
        colormap(hot);
        colorbar;
        title(sprintf('Frame %d (t=%.1fs)', frames_to_show(i), ...
                      (frames_to_show(i)-1)/plume_config.temporal.frame_rate));
        axis equal tight;
    end
    
    sgtitle(sprintf('%s: %.1f×%.1f cm, %d Hz', ...
            plume_config.plume_id, ...
            plume_config.spatial.arena_bounds.x_max * 2, ...
            plume_config.spatial.arena_bounds.y_max * 2, ...
            plume_config.temporal.frame_rate));
    
catch ME
    fprintf('Error: %s\n', ME.message);
end

cd(original_dir);
MATLAB_EOF

cat > test_smoke_simulation.m << 'MATLAB_EOF'
% test_smoke_simulation.m - Test smoke plume simulation

% Store current directory
original_dir = pwd;
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
cd(script_dir);

% Add Code directory
if exist('Code', 'dir')
    addpath(genpath('Code'));
end

fprintf('\n=== Testing Smoke Plume Simulation ===\n\n');

try
    [plume_file, plume_config] = get_plume_file();
    fprintf('Active plume: %s\n', plume_config.plume_id);
    fprintf('Frame rate: %.1f Hz\n', plume_config.temporal.frame_rate);
    
    % 10 second test
    test_frames = round(10 * plume_config.temporal.frame_rate);
    
    fprintf('\nRunning 10-second test (%d frames)...\n', test_frames);
    
    out = navigation_model_vec(test_frames, 'Crimaldi', 0, 2);
    fprintf('✓ Success! Generated %d time points\n', size(out.x, 1));
    
    % Visualize
    figure;
    plot(out.x(:,1), out.y(:,1), 'b-', 'LineWidth', 2);
    hold on;
    if size(out.x, 2) > 1
        plot(out.x(:,2), out.y(:,2), 'r-', 'LineWidth', 2);
    end
    
    % Source
    plot(plume_config.simulation.source_position.x_cm, ...
         plume_config.simulation.source_position.y_cm, ...
         'r*', 'MarkerSize', 15, 'LineWidth', 2);
    
    xlabel('X (cm)'); ylabel('Y (cm)');
    title(sprintf('Test - %s', plume_config.plume_id));
    axis equal; grid on;
    
catch ME
    fprintf('Error: %s\n', ME.message);
end

cd(original_dir);
MATLAB_EOF

echo "  ✓ Created visualize_smoke_plume.m"
echo "  ✓ Created test_smoke_simulation.m"

# Final summary
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Configuration created: $SMOKE_CONFIG"
echo "  - Frame rate: $FPS Hz ✓"
echo "  - Pixel size: $MM_PER_PIXEL mm/px ✓"
echo "  - Arena: ${arena_width_cm}×${arena_height_cm} cm"
echo "  - Frames: $frames"

if [[ "$RUN_ANALYSIS" == "n" ]]; then
    echo ""
    echo "⚠️  Analysis was skipped. Default dimensions used."
    echo "   To analyze later: matlab -r \"h5info('$SMOKE_HDF5')\""
fi

echo ""
echo "To visualize: matlab -r \"run('visualize_smoke_plume.m')\""
echo "To test: matlab -r \"run('test_smoke_simulation.m')\""
echo ""
echo "✓ Your smoke plume is now the active plume!"