#!/bin/bash
# cleanup_codebase.sh - Safely clean up redundant files

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Codebase Cleanup Script ==="
echo ""
echo -e "${YELLOW}This will help organize your navigation model codebase.${NC}"
echo "It will:"
echo "  1. Archive old outputs"
echo "  2. Remove redundant test scripts"
echo "  3. Create organized directory structure"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create organized directories
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p logs results old_files archived_scripts

# Archive old outputs
echo -e "\n${GREEN}Archiving old outputs...${NC}"
mv test_output_* old_files/ 2>/dev/null && echo "  ✓ Moved test_output directories"
mv test_*.out test_*.err old_files/ 2>/dev/null && echo "  ✓ Moved test outputs"
mv slurm-*.out slurm-*.err old_files/ 2>/dev/null && echo "  ✓ Moved old SLURM logs"
mv nav_output_*.log nav_error_*.log old_files/ 2>/dev/null && echo "  ✓ Moved nav logs"
mv *_test*.mat old_files/ 2>/dev/null && echo "  ✓ Moved test results"

# Archive redundant scripts (safer than deleting)
echo -e "\n${GREEN}Archiving redundant scripts...${NC}"
SCRIPTS_TO_ARCHIVE=(
    "test_run.sh"
    "debug_matlab_slurm.sh"
    "enhanced_test_run.sh"
    "small_test.sh"
    "submit_test.sh"
    "quick_test.sh"
    "run_single_test.sh"
    "just_run_this.sh"
    "START_HERE.sh"
    "fix_generated_job.sh"
    "proper_submit.sh"
    "submit_now.sh"
    "monitor_jobs.sh"
    "generated_job.slurm"
    "generated_job_fixed.slurm"
    "robust_slurm_job.slurm"
    "simple_slurm_fix.slurm"
    "debug_matlab_slurm.slurm"
    "test_job.slurm"
    "single_test.slurm"
    "quick_test_job.slurm"
    "full_sim_job.slurm"
    "array_job.slurm"
    "navigation_job.slurm"
    "working_nav_job.slurm"
    "test_nav_model.slurm"
)

for script in "${SCRIPTS_TO_ARCHIVE[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" archived_scripts/ && echo "  ✓ Archived $script"
    fi
done

# Remove empty files
echo -e "\n${GREEN}Removing empty files...${NC}"
[ -f "run_simulation.m" ] && [ ! -s "run_simulation.m" ] && rm "run_simulation.m" && echo "  ✓ Removed empty run_simulation.m"

# Create the final working script
echo -e "\n${GREEN}Creating final working script...${NC}"
if [ ! -f "nav_job_final.slurm" ]; then
    cat > nav_job_final.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=nav_model
#SBATCH --partition=day
#SBATCH --time=6:00:00
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-399%100
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Load MATLAB module (required on Grace)
module load MATLAB/2023b

# Setup
cd /home/snb6/Documents/AlvarezSalvado_ElementaryTransformations
mkdir -p logs results

# Log info
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

# Run MATLAB
matlab -nodisplay -nosplash << 'EOF'
% Add code to path
addpath(genpath('Code'));

% Get array task ID
task_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(task_id), task_id = 0; end

% Run simulation
try
    fprintf('Starting simulation for task %d\n', task_id);
    out = navigation_model_vec(3600, 'Crimaldi', 0, 10);
    
    % Save results
    filename = sprintf('results/nav_results_%04d.mat', task_id);
    save(filename, 'out', '-v7.3');
    fprintf('Results saved to %s\n', filename);
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    exit(1);
end

exit(0);
EOF

echo "Job completed at: $(date)"
EOF
    echo "  ✓ Created nav_job_final.slurm"
else
    echo "  ✓ nav_job_final.slurm already exists"
fi

# Create a simple run script
echo -e "\n${GREEN}Creating simple run script...${NC}"
cat > run_nav_model.sh << 'EOF'
#!/bin/bash
# Simple script to run navigation model

if [ "$1" == "test" ]; then
    echo "Submitting single test job..."
    sbatch --array=0-0 nav_job_final.slurm
elif [ "$1" == "full" ]; then
    echo "Submitting full array job (400 tasks)..."
    sbatch nav_job_final.slurm
else
    echo "Usage: ./run_nav_model.sh [test|full]"
    echo "  test - Run single test job"
    echo "  full - Run full 400-task array"
fi
EOF
chmod +x run_nav_model.sh
echo "  ✓ Created run_nav_model.sh"

# Update .gitignore
echo -e "\n${GREEN}Updating .gitignore...${NC}"
if ! grep -q "^logs/$" .gitignore 2>/dev/null; then
    echo -e "\n# SLURM outputs\nlogs/\nslurm-*.out\nslurm-*.err" >> .gitignore
    echo "  ✓ Added logs/ to .gitignore"
fi
if ! grep -q "^results/$" .gitignore 2>/dev/null; then
    echo "results/" >> .gitignore
    echo "  ✓ Added results/ to .gitignore"
fi

# Summary
echo -e "\n${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo "Directory structure:"
echo "  📁 Code/           - Your MATLAB functions (untouched)"
echo "  📁 logs/           - SLURM output logs"
echo "  📁 results/        - Simulation results"
echo "  📁 old_files/      - Archived outputs"
echo "  📁 archived_scripts/ - Old scripts (can delete after review)"
echo ""
echo "Key files:"
echo "  ✓ nav_job_final.slurm - Main SLURM job script"
echo "  ✓ run_nav_model.sh    - Simple run wrapper"
echo "  ✓ validate_output.py  - Output validation"
echo ""
echo -e "${YELLOW}To run:${NC}"
echo "  ./run_nav_model.sh test   # Single test job"
echo "  ./run_nav_model.sh full   # Full array job"
echo ""
echo -e "${YELLOW}To permanently delete archives:${NC}"
echo "  rm -rf old_files/ archived_scripts/"