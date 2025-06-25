#!/bin/bash

# =============================================================================
# Quick Validation Script for AlvarezSalvado Elementary Transformations
# =============================================================================
# Purpose: Quick validation of simulation setup and configuration
# Author: Blitzy Platform - Repository Organization Agent
# Created: 2024
# License: MIT
#
# This script performs rapid validation of the simulation environment,
# configuration files, and essential dependencies to ensure the system
# is ready for navigation model execution.
#
# Usage: ./scripts/quick_validate.sh [OPTIONS]
# Options:
#   -v, --verbose    Enable verbose logging with detailed trace output
#   -h, --help       Display this help message
#
# Environment Requirements:
#   - MATLAB R2017b+ (R2023b recommended for HPC)
#   - Python 3.10+ with required packages
#   - Bash 4.0+
#   - Standard Unix utilities (grep, awk, find, etc.)
# =============================================================================

# Script metadata
SCRIPT_NAME="quick_validate.sh"
SCRIPT_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
VERBOSE=0
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/quick_validate_$(date +%Y%m%d_%H%M%S).log"

# Colors for output (only if terminal supports it)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Display help message
show_help() {
    cat << EOF
${CYAN}Quick Validation Script for AlvarezSalvado Elementary Transformations${NC}

${BLUE}USAGE:${NC}
    ${SCRIPT_NAME} [OPTIONS]

${BLUE}DESCRIPTION:${NC}
    Performs rapid validation of the simulation environment, configuration files,
    and essential dependencies to ensure the system is ready for navigation
    model execution.

${BLUE}OPTIONS:${NC}
    -v, --verbose    Enable verbose logging with detailed trace output
                     Logs detailed execution steps to both stdout and logs/
    -h, --help       Display this help message and exit

${BLUE}VALIDATION CHECKS:${NC}
    • Environment validation (MATLAB, Python, Bash versions)
    • Directory structure verification
    • Configuration file validation
    • Dependency availability
    • Basic functionality tests
    • HPC environment detection

${BLUE}OUTPUT:${NC}
    Results are displayed in real-time with color-coded status indicators:
    ${GREEN}✓ PASS${NC}    - Check completed successfully
    ${YELLOW}⚠ WARN${NC}    - Check completed with warnings
    ${RED}✗ FAIL${NC}    - Check failed (requires attention)

${BLUE}EXAMPLES:${NC}
    ${SCRIPT_NAME}                    # Standard validation
    ${SCRIPT_NAME} --verbose          # Detailed verbose output
    ${SCRIPT_NAME} -v > validation.log 2>&1  # Capture verbose output to file

${BLUE}EXIT CODES:${NC}
    0  - All critical checks passed
    1  - One or more critical checks failed
    2  - Invalid command line arguments
EOF
}

# Logging function with timestamp and verbose support
log_message() {
    local level="$1"
    local message="$2"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    # Always log to stdout/stderr based on level
    case "$level" in
        "INFO")
            echo -e "[${timestamp}] ${BLUE}INFO${NC}: $message"
            ;;
        "WARN")
            echo -e "[${timestamp}] ${YELLOW}WARN${NC}: $message"
            ;;
        "ERROR")
            echo -e "[${timestamp}] ${RED}ERROR${NC}: $message" >&2
            ;;
        "SUCCESS")
            echo -e "[${timestamp}] ${GREEN}SUCCESS${NC}: $message"
            ;;
        "VERBOSE")
            if [[ $VERBOSE -eq 1 ]]; then
                echo -e "[${timestamp}] ${CYAN}VERBOSE${NC}: $message"
            fi
            ;;
    esac
    
    # Log to file if logging is enabled and directory exists
    if [[ -d "$LOG_DIR" ]]; then
        echo "[${timestamp}] [$level] $message" >> "$LOG_FILE" 2>/dev/null
    fi
}

# Verbose logging wrapper
verbose_log() {
    log_message "VERBOSE" "$1"
}

# Check result processing
process_check_result() {
    local check_name="$1"
    local result="$2"
    local details="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    case "$result" in
        "PASS")
            echo -e "${GREEN}✓ PASS${NC}  $check_name"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            verbose_log "Check passed: $check_name - $details"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARN${NC}  $check_name"
            if [[ -n "$details" ]]; then
                echo -e "         ${YELLOW}$details${NC}"
            fi
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            log_message "WARN" "Check warning: $check_name - $details"
            ;;
        "FAIL")
            echo -e "${RED}✗ FAIL${NC}  $check_name"
            if [[ -n "$details" ]]; then
                echo -e "         ${RED}$details${NC}"
            fi
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            log_message "ERROR" "Check failed: $check_name - $details"
            ;;
    esac
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

# Check if command exists and get version if possible
check_command_version() {
    local cmd="$1"
    local version_flag="$2"
    local min_version="$3"
    
    verbose_log "Checking availability of command: $cmd"
    
    if ! command -v "$cmd" &> /dev/null; then
        return 1
    fi
    
    if [[ -n "$version_flag" ]]; then
        local version_output
        version_output=$($cmd $version_flag 2>&1 | head -n 1)
        verbose_log "Version output for $cmd: $version_output"
        echo "$version_output"
    fi
    
    return 0
}

# Validate environment setup
validate_environment() {
    echo -e "\n${BLUE}=== Environment Validation ===${NC}"
    verbose_log "Starting environment validation checks"
    
    # Check Bash version
    verbose_log "Checking Bash version requirements (minimum 4.0)"
    local bash_version
    bash_version=$(bash --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
    if [[ -n "$bash_version" ]]; then
        local major_version=${bash_version%%.*}
        if [[ $major_version -ge 4 ]]; then
            process_check_result "Bash Version ($bash_version)" "PASS" "Meets minimum requirement of 4.0+"
        else
            process_check_result "Bash Version ($bash_version)" "FAIL" "Requires Bash 4.0+ for HPC scripting features"
        fi
    else
        process_check_result "Bash Version" "FAIL" "Unable to determine Bash version"
    fi
    
    # Check MATLAB availability
    verbose_log "Checking MATLAB installation and accessibility"
    local matlab_version
    matlab_version=$(check_command_version "matlab" "-batch \"version; quit\"" "R2017b")
    if [[ $? -eq 0 ]]; then
        if echo "$matlab_version" | grep -qE "(R20[1-9][7-9]|R202[0-9]|R203[0-9])"; then
            process_check_result "MATLAB Installation" "PASS" "Version detected: $(echo "$matlab_version" | grep -oE 'R20[0-9][0-9][ab]?' | head -n1)"
        else
            process_check_result "MATLAB Installation" "WARN" "MATLAB found but version may be older than R2017b"
        fi
    else
        process_check_result "MATLAB Installation" "FAIL" "MATLAB not found in PATH - required for navigation model execution"
    fi
    
    # Check Python version
    verbose_log "Checking Python version requirements (minimum 3.10)"
    local python_version
    python_version=$(check_command_version "python3" "--version")
    if [[ $? -eq 0 ]]; then
        local version_num=$(echo "$python_version" | grep -oE '[0-9]+\.[0-9]+' | head -n1)
        local major_version=${version_num%%.*}
        local minor_version=${version_num##*.}
        if [[ $major_version -eq 3 && $minor_version -ge 10 ]] || [[ $major_version -gt 3 ]]; then
            process_check_result "Python Version ($version_num)" "PASS" "Meets minimum requirement of 3.10+"
        else
            process_check_result "Python Version ($version_num)" "WARN" "Python 3.10+ recommended for optimal performance"
        fi
    else
        # Try python as fallback
        python_version=$(check_command_version "python" "--version")
        if [[ $? -eq 0 ]]; then
            process_check_result "Python Installation" "WARN" "Found 'python' command, verify it's Python 3.10+"
        else
            process_check_result "Python Installation" "FAIL" "Python not found - required for data pipeline"
        fi
    fi
    
    # Check essential Unix utilities
    verbose_log "Checking essential Unix utilities availability"
    local missing_utils=()
    local required_utils=("grep" "awk" "sed" "find" "sort" "head" "tail" "cut" "tr" "wc")
    
    for util in "${required_utils[@]}"; do
        if ! command -v "$util" &> /dev/null; then
            missing_utils+=("$util")
        fi
    done
    
    if [[ ${#missing_utils[@]} -eq 0 ]]; then
        process_check_result "Unix Utilities" "PASS" "All required utilities available"
    else
        process_check_result "Unix Utilities" "FAIL" "Missing utilities: ${missing_utils[*]}"
    fi
}

# Validate directory structure
validate_directory_structure() {
    echo -e "\n${BLUE}=== Directory Structure Validation ===${NC}"
    verbose_log "Starting directory structure validation"
    
    # Define expected directories
    local expected_dirs=(
        "Code"
        "configs"
        "scripts"
        "slurm"
        "matlab"
        "logs"
        "slurm_logs"
        "results"
        "tests"
    )
    
    local missing_dirs=()
    local present_dirs=()
    
    for dir in "${expected_dirs[@]}"; do
        local dir_path="${PROJECT_ROOT}/$dir"
        verbose_log "Checking directory: $dir_path"
        
        if [[ -d "$dir_path" ]]; then
            present_dirs+=("$dir")
            verbose_log "Directory exists: $dir"
        else
            missing_dirs+=("$dir")
            verbose_log "Directory missing: $dir"
        fi
    done
    
    # Report results
    if [[ ${#missing_dirs[@]} -eq 0 ]]; then
        process_check_result "Directory Structure" "PASS" "All ${#expected_dirs[@]} expected directories present"
    elif [[ ${#missing_dirs[@]} -le 2 ]]; then
        process_check_result "Directory Structure" "WARN" "Missing directories: ${missing_dirs[*]}"
    else
        process_check_result "Directory Structure" "FAIL" "Missing critical directories: ${missing_dirs[*]}"
    fi
    
    # Check for core directories specifically
    local core_dirs=("Code" "configs" "scripts")
    local missing_core=()
    
    for dir in "${core_dirs[@]}"; do
        if [[ ! -d "${PROJECT_ROOT}/$dir" ]]; then
            missing_core+=("$dir")
        fi
    done
    
    if [[ ${#missing_core[@]} -gt 0 ]]; then
        process_check_result "Core Directories" "FAIL" "Critical directories missing: ${missing_core[*]}"
    else
        process_check_result "Core Directories" "PASS" "All core directories present"
    fi
}

# Validate configuration files
validate_configuration() {
    echo -e "\n${BLUE}=== Configuration Validation ===${NC}"
    verbose_log "Starting configuration validation"
    
    local config_dir="${PROJECT_ROOT}/configs"
    
    # Check main configs directory
    if [[ ! -d "$config_dir" ]]; then
        process_check_result "Configuration Directory" "FAIL" "configs/ directory not found"
        return
    fi
    
    # Check for configuration files
    verbose_log "Scanning for configuration files in $config_dir"
    local config_files
    config_files=$(find "$config_dir" -name "*.json" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null)
    local config_count=$(echo "$config_files" | grep -c . 2>/dev/null || echo "0")
    
    if [[ $config_count -gt 0 ]]; then
        process_check_result "Configuration Files" "PASS" "Found $config_count configuration files"
        verbose_log "Configuration files found: $(echo "$config_files" | tr '\n' ' ')"
        
        # Validate JSON syntax for JSON files
        local json_files
        json_files=$(find "$config_dir" -name "*.json" 2>/dev/null)
        local json_errors=0
        
        while IFS= read -r json_file; do
            if [[ -n "$json_file" ]]; then
                verbose_log "Validating JSON syntax: $json_file"
                if ! python3 -m json.tool "$json_file" > /dev/null 2>&1; then
                    log_message "ERROR" "Invalid JSON syntax in: $json_file"
                    json_errors=$((json_errors + 1))
                fi
            fi
        done <<< "$json_files"
        
        if [[ $json_errors -eq 0 ]]; then
            process_check_result "JSON Syntax" "PASS" "All JSON files have valid syntax"
        else
            process_check_result "JSON Syntax" "FAIL" "$json_errors JSON files have syntax errors"
        fi
    else
        process_check_result "Configuration Files" "WARN" "No configuration files found - may need to create experiment configs"
    fi
    
    # Check for startup.m in project root
    verbose_log "Checking for MATLAB startup configuration"
    if [[ -f "${PROJECT_ROOT}/startup.m" ]]; then
        process_check_result "MATLAB Startup" "PASS" "startup.m found in project root"
    else
        process_check_result "MATLAB Startup" "WARN" "startup.m not found - MATLAB path configuration may be needed"
    fi
}

# Validate Python dependencies
validate_python_dependencies() {
    echo -e "\n${BLUE}=== Python Dependencies Validation ===${NC}"
    verbose_log "Starting Python dependencies validation"
    
    # Check if Python is available first
    if ! command -v python3 &> /dev/null; then
        process_check_result "Python Dependencies" "FAIL" "Python3 not available"
        return
    fi
    
    # Define required packages
    local required_packages=(
        "numpy"
        "scipy"
        "h5py"
        "matplotlib"
        "pandas"
        "loguru"
    )
    
    local missing_packages=()
    local available_packages=()
    
    for package in "${required_packages[@]}"; do
        verbose_log "Checking Python package: $package"
        if python3 -c "import $package" 2>/dev/null; then
            available_packages+=("$package")
            verbose_log "Package available: $package"
        else
            missing_packages+=("$package")
            verbose_log "Package missing: $package"
        fi
    done
    
    # Report results
    if [[ ${#missing_packages[@]} -eq 0 ]]; then
        process_check_result "Python Packages" "PASS" "All ${#required_packages[@]} required packages available"
    elif [[ ${#missing_packages[@]} -le 2 ]]; then
        process_check_result "Python Packages" "WARN" "Missing packages: ${missing_packages[*]}"
    else
        process_check_result "Python Packages" "FAIL" "Missing critical packages: ${missing_packages[*]}"
    fi
    
    # Check for loguru specifically (for logging)
    if python3 -c "import loguru" 2>/dev/null; then
        process_check_result "Loguru Logging" "PASS" "loguru package available for structured logging"
    else
        process_check_result "Loguru Logging" "WARN" "loguru package not found - install for enhanced logging"
    fi
}

# Validate HPC environment
validate_hpc_environment() {
    echo -e "\n${BLUE}=== HPC Environment Validation ===${NC}"
    verbose_log "Starting HPC environment validation"
    
    # Check for SLURM commands
    local slurm_commands=("sbatch" "squeue" "scancel" "sinfo" "sacct")
    local available_slurm=()
    local missing_slurm=()
    
    for cmd in "${slurm_commands[@]}"; do
        verbose_log "Checking SLURM command: $cmd"
        if command -v "$cmd" &> /dev/null; then
            available_slurm+=("$cmd")
        else
            missing_slurm+=("$cmd")
        fi
    done
    
    if [[ ${#available_slurm[@]} -eq ${#slurm_commands[@]} ]]; then
        process_check_result "SLURM Commands" "PASS" "All SLURM commands available"
        
        # Check SLURM template files
        local slurm_dir="${PROJECT_ROOT}/slurm"
        if [[ -d "$slurm_dir" ]]; then
            local slurm_templates
            slurm_templates=$(find "$slurm_dir" -name "*.slurm" 2>/dev/null | wc -l)
            if [[ $slurm_templates -gt 0 ]]; then
                process_check_result "SLURM Templates" "PASS" "Found $slurm_templates SLURM template files"
            else
                process_check_result "SLURM Templates" "WARN" "No SLURM template files found in slurm/ directory"
            fi
        else
            process_check_result "SLURM Templates" "WARN" "slurm/ directory not found"
        fi
    elif [[ ${#available_slurm[@]} -gt 0 ]]; then
        process_check_result "SLURM Commands" "WARN" "Partial SLURM installation: available=${available_slurm[*]}, missing=${missing_slurm[*]}"
    else
        process_check_result "SLURM Commands" "WARN" "No SLURM commands found - HPC job submission not available"
    fi
    
    # Check for common HPC environment variables
    local hpc_vars=("SLURM_JOB_ID" "SLURM_PROCID" "SLURM_LOCALID" "PBS_JOBID" "LSB_JOBID")
    local detected_scheduler=""
    
    for var in "${hpc_vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            case "$var" in
                SLURM_*) detected_scheduler="SLURM" ;;
                PBS_*) detected_scheduler="PBS" ;;
                LSB_*) detected_scheduler="LSF" ;;
            esac
            break
        fi
    done
    
    if [[ -n "$detected_scheduler" ]]; then
        process_check_result "HPC Scheduler" "PASS" "Running under $detected_scheduler scheduler"
    else
        process_check_result "HPC Scheduler" "WARN" "Not currently running under HPC scheduler (normal for interactive use)"
    fi
}

# Validate logging setup
validate_logging_setup() {
    echo -e "\n${BLUE}=== Logging Setup Validation ===${NC}"
    verbose_log "Starting logging setup validation"
    
    # Check logs directory
    if [[ -d "$LOG_DIR" ]]; then
        if [[ -w "$LOG_DIR" ]]; then
            process_check_result "Logs Directory" "PASS" "logs/ directory exists and is writable"
        else
            process_check_result "Logs Directory" "WARN" "logs/ directory exists but is not writable"
        fi
    else
        verbose_log "Creating logs directory: $LOG_DIR"
        if mkdir -p "$LOG_DIR" 2>/dev/null; then
            process_check_result "Logs Directory" "PASS" "logs/ directory created successfully"
        else
            process_check_result "Logs Directory" "FAIL" "Unable to create logs/ directory"
        fi
    fi
    
    # Check slurm_logs directory
    local slurm_logs_dir="${PROJECT_ROOT}/slurm_logs"
    if [[ -d "$slurm_logs_dir" ]]; then
        process_check_result "SLURM Logs Directory" "PASS" "slurm_logs/ directory exists"
    else
        verbose_log "Creating SLURM logs directory: $slurm_logs_dir"
        if mkdir -p "$slurm_logs_dir" 2>/dev/null; then
            process_check_result "SLURM Logs Directory" "PASS" "slurm_logs/ directory created successfully"
        else
            process_check_result "SLURM Logs Directory" "WARN" "Unable to create slurm_logs/ directory"
        fi
    fi
    
    # Test log file creation
    local test_log="${LOG_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
    if echo "Test log entry" > "$test_log" 2>/dev/null; then
        process_check_result "Log File Creation" "PASS" "Successfully created test log file"
        rm -f "$test_log" 2>/dev/null
    else
        process_check_result "Log File Creation" "WARN" "Unable to create test log file"
    fi
}

# Run basic functionality tests
validate_basic_functionality() {
    echo -e "\n${BLUE}=== Basic Functionality Validation ===${NC}"
    verbose_log "Starting basic functionality validation"
    
    # Test script execution permissions
    local script_dir="${PROJECT_ROOT}/scripts"
    if [[ -d "$script_dir" ]]; then
        verbose_log "Checking script permissions in $script_dir"
        local executable_scripts
        executable_scripts=$(find "$script_dir" -name "*.sh" -executable 2>/dev/null | wc -l)
        local total_scripts
        total_scripts=$(find "$script_dir" -name "*.sh" 2>/dev/null | wc -l)
        
        if [[ $executable_scripts -eq $total_scripts ]] && [[ $total_scripts -gt 0 ]]; then
            process_check_result "Script Permissions" "PASS" "All $total_scripts shell scripts are executable"
        elif [[ $total_scripts -gt 0 ]]; then
            process_check_result "Script Permissions" "WARN" "$executable_scripts/$total_scripts scripts are executable"
        else
            process_check_result "Script Permissions" "WARN" "No shell scripts found in scripts/ directory"
        fi
    else
        process_check_result "Script Permissions" "WARN" "scripts/ directory not found"
    fi
    
    # Test MATLAB path detection
    if command -v matlab &> /dev/null; then
        verbose_log "Testing MATLAB basic functionality"
        local matlab_test_result
        matlab_test_result=$(timeout 30s matlab -batch "disp('MATLAB_OK'); quit" 2>/dev/null | grep -o "MATLAB_OK" | head -n1)
        if [[ "$matlab_test_result" == "MATLAB_OK" ]]; then
            process_check_result "MATLAB Functionality" "PASS" "MATLAB executes successfully"
        else
            process_check_result "MATLAB Functionality" "WARN" "MATLAB execution test failed or timed out"
        fi
    else
        process_check_result "MATLAB Functionality" "WARN" "MATLAB not available for testing"
    fi
    
    # Test Python import capabilities
    if command -v python3 &> /dev/null; then
        verbose_log "Testing Python import capabilities"
        local python_test
        python_test=$(python3 -c "import sys; print('PYTHON_OK')" 2>/dev/null)
        if [[ "$python_test" == "PYTHON_OK" ]]; then
            process_check_result "Python Functionality" "PASS" "Python executes successfully"
        else
            process_check_result "Python Functionality" "WARN" "Python execution test failed"
        fi
    else
        process_check_result "Python Functionality" "WARN" "Python not available for testing"
    fi
}

# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=1
                verbose_log "Verbose logging enabled"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                log_message "ERROR" "Unknown option: $1"
                echo "Use -h or --help for usage information."
                exit 2
                ;;
            *)
                log_message "ERROR" "Unexpected argument: $1"
                echo "Use -h or --help for usage information."
                exit 2
                ;;
        esac
    done
}

# Display summary results
display_summary() {
    echo -e "\n${BLUE}=== Validation Summary ===${NC}"
    echo -e "${GREEN}Passed:${NC}   $PASSED_CHECKS/$TOTAL_CHECKS"
    echo -e "${YELLOW}Warnings:${NC} $WARNING_CHECKS/$TOTAL_CHECKS"
    echo -e "${RED}Failed:${NC}   $FAILED_CHECKS/$TOTAL_CHECKS"
    
    local success_rate=0
    if [[ $TOTAL_CHECKS -gt 0 ]]; then
        success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
    fi
    
    echo -e "${CYAN}Success Rate: ${success_rate}%${NC}"
    
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${BLUE}Detailed log:${NC} $LOG_FILE"
    fi
    
    # Provide recommendations based on results
    echo -e "\n${BLUE}=== Recommendations ===${NC}"
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        if [[ $WARNING_CHECKS -eq 0 ]]; then
            echo -e "${GREEN}✓ System is ready for simulation execution${NC}"
            log_message "SUCCESS" "All validation checks passed successfully"
        else
            echo -e "${YELLOW}⚠ System is mostly ready with some optional improvements needed${NC}"
            echo -e "  Consider addressing the warnings above for optimal performance"
        fi
    else
        echo -e "${RED}✗ System has critical issues that should be addressed before simulation${NC}"
        echo -e "  Please resolve the failed checks above before proceeding"
        
        if [[ $FAILED_CHECKS -gt $PASSED_CHECKS ]]; then
            echo -e "  ${RED}Consider running setup scripts or checking the installation guide${NC}"
        fi
    fi
}

# Main execution function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Display header
    echo -e "${CYAN}Quick Validation Script for AlvarezSalvado Elementary Transformations${NC}"
    echo -e "${BLUE}Version: $SCRIPT_VERSION${NC}"
    echo -e "${BLUE}Timestamp: $(date)${NC}"
    
    if [[ $VERBOSE -eq 1 ]]; then
        echo -e "${CYAN}Verbose logging enabled${NC}"
        echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
        echo -e "${BLUE}Log file: $LOG_FILE${NC}"
    fi
    
    verbose_log "Starting quick validation process"
    verbose_log "Script directory: $SCRIPT_DIR"
    verbose_log "Project root: $PROJECT_ROOT"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$LOG_DIR" 2>/dev/null
    
    # Initialize log file
    log_message "INFO" "Quick validation started by user: $(whoami)"
    log_message "INFO" "Command line: $0 $*"
    log_message "INFO" "Working directory: $(pwd)"
    log_message "INFO" "System: $(uname -a)"
    
    # Run validation checks
    validate_environment
    validate_directory_structure
    validate_configuration
    validate_python_dependencies
    validate_hpc_environment
    validate_logging_setup
    validate_basic_functionality
    
    # Display summary
    display_summary
    
    # Log completion
    log_message "INFO" "Quick validation completed"
    verbose_log "Total runtime: $SECONDS seconds"
    
    # Set exit code based on results
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Ensure script is being run from the correct context
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Trap to ensure cleanup on exit
    trap 'log_message "INFO" "Quick validation script terminated"' EXIT
    
    # Execute main function with all arguments
    main "$@"
fi