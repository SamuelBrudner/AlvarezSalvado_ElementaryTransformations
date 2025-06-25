#!/bin/bash

# Output unbuffering utility script for real-time output display
# Relocated to scripts/ directory with enhanced verbose logging support
#
# This script modifies setup_smoke_plume_config.sh to add output unbuffering
# capabilities using stdbuf command for real-time output display.
#
# Usage: unbuffer_output.sh [-v|--verbose]
#   -v, --verbose    Enable detailed verbose logging output

# Initialize verbose mode (default: off)
VERBOSE=0

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=1
                [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] unbuffer_output.sh: Verbose logging enabled"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [-v|--verbose]"
                echo "  -v, --verbose    Enable detailed verbose logging output"
                echo "  -h, --help       Show this help message"
                exit 0
                ;;
            *)
                echo "Error: Unknown option $1" >&2
                echo "Use -h or --help for usage information" >&2
                exit 1
                ;;
        esac
    done
}

# Verbose logging function
log_verbose() {
    local message="$1"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] unbuffer_output.sh: $message"
        # Also log to logs directory if it exists
        if [[ -d "logs" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] unbuffer_output.sh: $message" >> "logs/unbuffer_output_$(date '+%Y%m%d').log"
        fi
    fi
}

# Main execution function
main() {
    log_verbose "Starting output unbuffering configuration"
    
    # Check if target file exists
    if [[ ! -f "setup_smoke_plume_config.sh" ]]; then
        log_verbose "Target file setup_smoke_plume_config.sh not found in current directory"
        echo "Error: setup_smoke_plume_config.sh not found in current directory" >&2
        exit 1
    fi
    
    log_verbose "Found target file: setup_smoke_plume_config.sh"
    
    # Create backup if verbose mode is enabled
    if [[ $VERBOSE -eq 1 ]]; then
        cp setup_smoke_plume_config.sh setup_smoke_plume_config.sh.backup
        log_verbose "Created backup: setup_smoke_plume_config.sh.backup"
    fi
    
    log_verbose "Applying sed modifications to add stdbuf for unbuffered output"
    
    # Modify the wrapper to use stdbuf for unbuffered output
    if sed -i 's/matlab -nodisplay -nosplash/stdbuf -o0 -e0 matlab -nodisplay -nosplash/g' setup_smoke_plume_config.sh; then
        log_verbose "Successfully applied sed modifications"
    else
        log_verbose "Error: Failed to apply sed modifications"
        echo "Error: Failed to apply sed modifications" >&2
        exit 1
    fi
    
    log_verbose "Applying perl modifications to add stdbuf availability check"
    
    # If stdbuf is not available, use script command as alternative
    if perl -i -pe '
# Add check for stdbuf
if (/if command -v timeout &> \/dev\/null; then/) {
    print "# Check if we can unbuffer output\n";
    print "if command -v stdbuf &> /dev/null; then\n";
    print "    MATLAB_CMD=\"stdbuf -o0 -e0 matlab\"\n";
    print "else\n";
    print "    MATLAB_CMD=\"matlab\"\n";
    print "fi\n\n";
}

# Use MATLAB_CMD variable
s/matlab -nodisplay/\$MATLAB_CMD -nodisplay/g;
' setup_smoke_plume_config.sh; then
        log_verbose "Successfully applied perl modifications"
    else
        log_verbose "Error: Failed to apply perl modifications"
        echo "Error: Failed to apply perl modifications" >&2
        exit 1
    fi
    
    log_verbose "All modifications completed successfully"
    echo "Output unbuffering added!"
    
    if [[ $VERBOSE -eq 1 ]]; then
        log_verbose "Verifying modifications in setup_smoke_plume_config.sh"
        if grep -q "stdbuf" setup_smoke_plume_config.sh; then
            log_verbose "Verification successful: stdbuf modifications are present"
        else
            log_verbose "Warning: stdbuf modifications may not have been applied correctly"
        fi
    fi
    
    log_verbose "unbuffer_output.sh execution completed"
}

# Parse arguments and execute main function
parse_args "$@"
main