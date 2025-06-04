#!/bin/bash
# fix_nav_job_slurm.sh - Fix the nav_job_paths.slurm script

echo "Fixing nav_job_paths.slurm..."

# Create corrected version
cat > nav_job_paths_fixed.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/nav-%A_%a.out
#SBATCH --error=logs/nav-%A_%a.err

# Set project directory
PROJECT_ROOT="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations"

# Change to project directory
cd "$PROJECT_ROOT"

# Create directories
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Run simulation
matlab -nodisplay -nosplash -r "
% Load paths configuration
addpath(genpath('Code'));
paths = load_paths();
fprintf('Using paths from: %s\n', paths.project_root);

% Get task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

fprintf('Task %d: Starting simulation\n', task_id);

try
    % Run simulation with config duration (should be 300s)
    out = navigation_model_vec('config', 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Task %d: Success! Saved to %s\n', task_id, filename);
    fprintf('Task %d: Simulated %d samples = %.1f seconds\n', task_id, size(out.x,1), size(out.x,1)/15);
catch ME
    fprintf('Task %d ERROR: %s\n', task_id, ME.message);
    exit(1);
end
exit(0);
"
EOF

# Backup old version
if [ -f "nav_job_paths.slurm" ]; then
    mv nav_job_paths.slurm nav_job_paths.slurm.broken
fi

# Install fixed version
mv nav_job_paths_fixed.slurm nav_job_paths.slurm

echo "✓ Fixed nav_job_paths.slurm"
echo ""
echo "The problem was: trying to 'source' a JSON file"
echo "The fix: removed that line and just set PROJECT_ROOT directly"

# Also check the config
echo ""
echo "Checking simulation duration in config:"
grep -A2 "simulation" configs/plumes/crimaldi_10cms_bounded.json | grep duration

echo ""
echo "To test a single job:"
echo "  sbatch --array=0-0 nav_job_paths.slurm"