#!/bin/bash
# comprehensive_hpc_fix.sh - Multiple solutions for HPC symlink issue

echo "=== Comprehensive HPC Symlink Fix ==="
echo ""

# Solution 1: Environment variable override
echo "SOLUTION 1: Environment Variable Override"
echo "-----------------------------------------"
cat > use_env_override.sh << 'EOF'
#!/bin/bash
# Set plume file path explicitly
export MATLAB_PLUME_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"

echo "Testing with MATLAB_PLUME_FILE=$MATLAB_PLUME_FILE"

matlab -nodisplay -nosplash -r "
addpath(genpath('Code'));

% Check for environment variable override
plume_override = getenv('MATLAB_PLUME_FILE');
if ~isempty(plume_override)
    fprintf('Using plume file override: %s\n', plume_override);
    % Temporarily modify get_plume_file
    plume_file = plume_override;
else
    [plume_file, ~] = get_plume_file();
end

if exist(plume_file, 'file')
    fprintf('✓ File exists!\n');
    out = navigation_model_vec(100, 'Crimaldi', 0, 1);
    fprintf('✓ Crimaldi works!\n');
else
    fprintf('✗ File not found\n');
end
exit;
" 2>&1 | tail -10
EOF

# Solution 2: Wrapper function
echo ""
echo "SOLUTION 2: Wrapper Function"
echo "-----------------------------------------"
cat > Code/get_plume_file_hpc.m << 'EOF'
function [plume_file, plume_config] = get_plume_file_hpc()
% GET_PLUME_FILE_HPC - HPC-aware wrapper for get_plume_file
%   Handles /vast/palmer/... to /home/snb6/... conversion

% First get the normal result
[plume_file, plume_config] = get_plume_file();

% Fix for HPC paths
if contains(plume_file, '/vast/palmer/home.grace/snb6/')
    plume_file = strrep(plume_file, '/vast/palmer/home.grace/snb6/', '/home/snb6/');
    fprintf('HPC path conversion: Using %s\n', plume_file);
end

% Also check environment override
env_override = getenv('MATLAB_PLUME_FILE');
if ~isempty(env_override) && exist(env_override, 'file')
    plume_file = env_override;
    fprintf('Using environment override: %s\n', plume_file);
end

end
EOF

# Solution 3: Modified navigation model
echo ""
echo "SOLUTION 3: Add Path Fix to Navigation Models"
echo "-----------------------------------------"
cat > add_hpc_fix_to_models.sh << 'SCRIPT'
#!/bin/bash
# Add HPC fix directly to navigation models

for model in "Elifenavmodel_bilateral.m" "navigation_model_vec.m"; do
    if [ -f "Code/$model" ]; then
        # Check if already patched
        if grep -q "HPC path fix" "Code/$model"; then
            echo "✓ $model already patched"
        else
            # Find line after get_plume_file call
            LINE=$(grep -n "plume_filename = get_plume_file" "Code/$model" | cut -d: -f1 | head -1)
            if [ ! -z "$LINE" ]; then
                # Insert fix after that line
                NEXT_LINE=$((LINE + 1))
                sed -i "${NEXT_LINE}i\\            % HPC path fix\\
            if contains(plume_filename, '/vast/palmer/home.grace/snb6/')\\
                plume_filename = strrep(plume_filename, '/vast/palmer/home.grace/snb6/', '/home/snb6/');\\
            end\\
            % Also check environment override\\
            env_override = getenv('MATLAB_PLUME_FILE');\\
            if ~isempty(env_override) && exist(env_override, 'file')\\
                plume_filename = env_override;\\
            end" "Code/$model"
                echo "✓ Patched $model"
            fi
        fi
    fi
done
SCRIPT

chmod +x add_hpc_fix_to_models.sh use_env_override.sh

# Solution 4: SLURM job template with fix
echo ""
echo "SOLUTION 4: SLURM Job Template with HPC Fix"
echo "-----------------------------------------"
cat > crimaldi_job_hpc.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=crimaldi_hpc
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99%20
#SBATCH --output=logs/crimaldi-%A_%a.out
#SBATCH --error=logs/crimaldi-%A_%a.err

# Set up HPC path fix
export MATLAB_PLUME_FILE="/home/snb6/Documents/AlvarezSalvado_ElementaryTransformations/data/plumes/10302017_10cms_bounded.hdf5"

# Create logs directory
mkdir -p logs results

# Load MATLAB
module load MATLAB/2023b

# Run simulation
matlab -nodisplay -nosplash -r "
addpath(genpath('Code'));

% Get task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

fprintf('Task %d: Starting Crimaldi simulation\n', task_id);

try
    % Run with config duration (300s)
    out = navigation_model_vec('config', 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/crimaldi_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Task %d: Success! Saved to %s\n', task_id, filename);
catch ME
    fprintf('Task %d ERROR: %s\n', task_id, ME.message);
    exit(1);
end
exit(0);
"
EOF

echo ""
echo "=== INSTRUCTIONS ==="
echo ""
echo "Try these solutions in order:"
echo ""
echo "1. Quick test with environment variable:"
echo "   ./use_env_override.sh"
echo ""
echo "2. Apply fix to navigation models:"
echo "   ./add_hpc_fix_to_models.sh"
echo "   Then test normally"
echo ""
echo "3. Use the HPC-ready SLURM script:"
echo "   sbatch crimaldi_job_hpc.slurm"
echo ""
echo "4. Or just use Gaussian environment (no plume file needed):"
echo "   Change 'Crimaldi' to 'gaussian' in your scripts"