#!/bin/bash

# Fix the heredoc issue and clean up Step 4
# Enhanced with verbose logging support

# Initialize verbose flag
VERBOSE=0

# Function to print verbose messages
verbose_echo() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_heredoc.sh: $1"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Fix heredoc syntax issues in shell scripts"
    echo ""
    echo "OPTIONS:"
    echo "  -v, --verbose    Enable verbose logging output"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Run with default settings"
    echo "  $0 --verbose      # Run with detailed trace output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            verbose_echo "Verbose mode enabled"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

verbose_echo "Starting heredoc syntax fixes"

# Check if target file exists
if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
    echo "Error: setup_smoke_plume_config.sh not found in current directory"
    verbose_echo "Working directory: $(pwd)"
    verbose_echo "Files in current directory: $(ls -la)"
    exit 1
fi

verbose_echo "Found target file: setup_smoke_plume_config.sh"

# Create backup if verbose mode is on
if [[ $VERBOSE -eq 1 ]]; then
    cp setup_smoke_plume_config.sh setup_smoke_plume_config.sh.backup
    verbose_echo "Created backup: setup_smoke_plume_config.sh.backup"
fi

verbose_echo "Applying perl-based fixes to heredoc syntax"

# Fix the heredoc issue and clean up Step 4
perl -i -pe '
# Fix the wrapper heredoc back to quoted to prevent variable expansion
s/cat > "\$TEMP_DIR\/run_analysis.sh" << WRAPPER_EOF/cat > "\$TEMP_DIR\/run_analysis.sh" << '\''WRAPPER_EOF'\''/;

# Remove duplicate "Step 4: Running comprehensive plume analysis..." message
s/^echo "Step 4: Running comprehensive plume analysis\.\.\."$//;

# Remove the duplicate "This will sample 100 random frames" message
s/^echo "  This will sample 100 random frames and calculate statistics\.\.\."$//;
' setup_smoke_plume_config.sh

verbose_echo "Perl-based fixes completed successfully"

verbose_echo "Applying sed-based fixes to MATLAB variable usage"

# Also fix the MATLAB script to use the variable correctly
sed -i 's/n_samples = min($N_SAMPLE_FRAMES, n_frames);/n_samples = min('$N_SAMPLE_FRAMES', n_frames);/' setup_smoke_plume_config.sh

verbose_echo "Fixed MATLAB variable interpolation"

verbose_echo "Applying timeout line escaping fix"

# Fix the timeout line in wrapper to properly escape
sed -i '/timeout \${TIMEOUT_SECONDS}s matlab/ s/\${TIMEOUT_SECONDS}s/\$TIMEOUT_SECONDS"s"/' setup_smoke_plume_config.sh

verbose_echo "Fixed timeout command escaping"

# Verify changes were applied
if [[ $VERBOSE -eq 1 ]]; then
    verbose_echo "Verifying applied changes:"
    
    # Check for heredoc fix
    if grep -q "cat > \"\$TEMP_DIR/run_analysis.sh\" << 'WRAPPER_EOF'" setup_smoke_plume_config.sh; then
        verbose_echo "✓ Heredoc syntax fix applied successfully"
    else
        verbose_echo "⚠ Warning: Heredoc syntax fix may not have been applied"
    fi
    
    # Check for duplicate message removal
    duplicate_count=$(grep -c "Step 4: Running comprehensive plume analysis" setup_smoke_plume_config.sh)
    if [[ $duplicate_count -le 1 ]]; then
        verbose_echo "✓ Duplicate message cleanup completed"
    else
        verbose_echo "⚠ Warning: Still found $duplicate_count instances of duplicate message"
    fi
    
    # Check MATLAB variable fix
    if grep -q "n_samples = min('.*', n_frames);" setup_smoke_plume_config.sh; then
        verbose_echo "✓ MATLAB variable interpolation fix applied"
    else
        verbose_echo "⚠ Warning: MATLAB variable fix may not have been applied"
    fi
    
    verbose_echo "Change verification completed"
fi

echo "Fixes applied!"
verbose_echo "fix_heredoc.sh execution completed successfully"

# Create log entry if logs directory exists
if [[ -d "logs" ]] && [[ $VERBOSE -eq 1 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] fix_heredoc.sh: Successfully applied heredoc and syntax fixes to setup_smoke_plume_config.sh" >> logs/fix_heredoc_$(date +%Y%m%d).log
    verbose_echo "Log entry created in logs/fix_heredoc_$(date +%Y%m%d).log"
fi