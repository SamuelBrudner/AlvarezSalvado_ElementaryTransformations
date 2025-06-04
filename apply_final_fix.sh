#!/bin/bash
# apply_final_fix.sh - Apply final clean SLURM script and check results

echo "=== Applying Final SLURM Script Fix ==="
echo ""

# First, let's check what results we already have
echo "1. Checking existing results:"
if [ -f "results/nav_results_0000.mat" ]; then
    echo "   ✓ Found nav_results_0000.mat"
    ls -lh results/nav_results_0000.mat
else
    echo "   No results found yet"
fi

echo ""
echo "2. Recent successful jobs:"
grep -l "Task.*COMPLETE\|Task.*Complete" logs/nav-*.out 2>/dev/null | tail -5

echo ""
echo "3. Creating final clean SLURM script..."

# Backup current version
cp nav_job_paths.slurm nav_job_paths.slurm.backup_final

# Create the final clean version
cat > nav_job_paths.slurm << 'EOF'
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

# Get array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Run MATLAB in batch mode with inline script
# The -batch flag is cleaner than here-docs for newer MATLAB versions
matlab -batch "
% Navigation model task $TASK_ID
fprintf('\n=== Navigation Model Task $TASK_ID ===\n');

% Setup
cd('$PROJECT_ROOT');
addpath(genpath('Code'));

% Load configuration
try
    paths = load_paths();
    fprintf('Project root: %s\n', paths.project_root);
catch
    fprintf('Note: Could not load paths config\n');
end

% Task parameters
task_id = $TASK_ID;

try
    % Load plume config
    fprintf('\nLoading configuration...\n');
    [plume_file, plume_config] = get_plume_file();
    
    % Get duration
    if isfield(plume_config, 'simulation') && isfield(plume_config.simulation, 'duration_seconds')
        duration = plume_config.simulation.duration_seconds;
    else
        duration = 300;
    end
    fprintf('Simulation duration: %.0f seconds\n', duration);
    
    % Calculate samples (15 Hz for Crimaldi)
    n_samples = round(duration * 15);
    fprintf('Total samples: %d at 15 Hz\n', n_samples);
    
    % Run simulation
    fprintf('\nRunning navigation simulation...\n');
    tic;
    out = navigation_model_vec(n_samples, 'Crimaldi', 0, 10);
    elapsed = toc;
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    
    % Report
    fprintf('\n=== Task %d Complete ===\n', task_id);
    fprintf('Agents: %d\n', size(out.x,2));
    fprintf('Duration: %.1f seconds simulated\n', size(out.x,1)/15);
    fprintf('Runtime: %.1f seconds actual\n', elapsed);
    fprintf('Output: %s\n', filename);
    
    if exist(filename, 'file')
        d = dir(filename);
        fprintf('Size: %.1f MB\n', d.bytes/1e6);
    end
    
    if isfield(out, 'successrate')
        fprintf('Success rate: %.1f%%\n', out.successrate * 100);
    end
    
catch ME
    fprintf('\n=== Task %d FAILED ===\n', task_id);
    fprintf('Error: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end
"
EOF

chmod +x nav_job_paths.slurm

echo "✓ Final SLURM script created"
echo ""
echo "4. Key improvements in final version:"
echo "   - Uses 'matlab -batch' for clean execution"
echo "   - No interactive prompts"
echo "   - No temporary files"
echo "   - Clean error handling with rethrow"
echo "   - Concise output"
echo ""
echo "5. Check if results already exist:"
find results -name "nav_results_*.mat" -type f | head -10

echo ""
echo "To submit remaining jobs (1-99):"
echo "  sbatch --array=1-99%20 nav_job_paths.slurm"
echo ""
echo "To check all results:"
echo "  ls -la results/nav_results_*.mat | wc -l"
echo ""
echo "To analyze results:"
echo "  ./matlab_results_check.sh results/nav_results_0000.mat"