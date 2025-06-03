#!/bin/bash

# =============================================================================
# BATCH SIMULATION EXECUTION SCRIPT
# =============================================================================
# 
# Comprehensive bash shell script for orchestrating batch simulation execution
# of 4000+ plume navigation algorithm simulations with scientific computing
# standards, automated environment validation, and performance optimization.
#
# Author: Plume Navigation System
# Version: 1.0.0
# License: Enterprise Research License
# 
# Performance Targets:
# - Complete 4000 simulations within 8 hours
# - <7.2 seconds average per simulation
# - >95% correlation accuracy
# - >99% reproducibility coefficient
#
# =============================================================================

# Global Script Constants
readonly SCRIPT_VERSION='1.0.0'
readonly SCRIPT_NAME='run_batch_simulation.sh'
readonly SCRIPT_DESCRIPTION='Batch simulation execution script for plume navigation algorithm analysis'
readonly DEFAULT_PYTHON_EXECUTABLE='python3'
readonly DEFAULT_CONFIG_DIR='../config'
readonly DEFAULT_OUTPUT_DIR='../../output'
readonly DEFAULT_LOG_LEVEL='INFO'
readonly BACKEND_MODULE_PATH='src.backend'
readonly VALIDATION_SCRIPT='validate_environment.py'
readonly CLI_MODULE='cli.py'
readonly MAIN_MODULE='__main__.py'

# Exit Code Constants
readonly EXIT_SUCCESS=0
readonly EXIT_FAILURE=1
readonly EXIT_CONFIG_ERROR=2
readonly EXIT_VALIDATION_ERROR=3
readonly EXIT_RESOURCE_ERROR=4
readonly EXIT_SIMULATION_ERROR=5

# Performance and Resource Constants
readonly BATCH_SIZE_DEFAULT=4000
readonly TARGET_COMPLETION_HOURS=8.0
readonly TARGET_SIMULATION_SECONDS=7.2
readonly REQUIRED_MEMORY_GB=8
readonly REQUIRED_DISK_GB=20

# Color Constants for Terminal Output
readonly COLOR_RED='\033[91m'
readonly COLOR_GREEN='\033[92m'
readonly COLOR_YELLOW='\033[93m'
readonly COLOR_BLUE='\033[94m'
readonly COLOR_CYAN='\033[96m'
readonly COLOR_RESET='\033[0m'
readonly COLOR_BOLD='\033[1m'

# Global Variables for Script State
declare -g PYTHON_EXEC="${DEFAULT_PYTHON_EXECUTABLE}"
declare -g CONFIG_DIR="${DEFAULT_CONFIG_DIR}"
declare -g OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
declare -g LOG_LEVEL="${DEFAULT_LOG_LEVEL}"
declare -g BATCH_SIZE="${BATCH_SIZE_DEFAULT}"
declare -g ENABLE_COLORS=true
declare -g STRICT_VALIDATION=false
declare -g SKIP_PERFORMANCE_CHECK=false
declare -g ENABLE_PARALLEL=true
declare -g PRESERVE_LOGS=true
declare -g PRESERVE_INTERMEDIATE=false
declare -g MONITORING_INTERVAL=5
declare -g PYTHON_PID=0
declare -g EXECUTION_START_TIME=0
declare -g INTERRUPTED=false

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

show_usage() {
    cat << EOF
${COLOR_BOLD}${SCRIPT_NAME} v${SCRIPT_VERSION}${COLOR_RESET}
${SCRIPT_DESCRIPTION}

${COLOR_BOLD}SYNOPSIS${COLOR_RESET}
    $0 [OPTIONS] <input_data_path> [output_directory]

${COLOR_BOLD}DESCRIPTION${COLOR_RESET}
    Orchestrates batch execution of 4000+ plume navigation algorithm simulations
    with comprehensive environment validation, progress monitoring, and performance
    optimization. Designed for scientific computing standards with automated
    resource management and reproducible results.

${COLOR_BOLD}REQUIRED ARGUMENTS${COLOR_RESET}
    input_data_path     Path to input plume data files (Crimaldi or custom format)

${COLOR_BOLD}OPTIONAL ARGUMENTS${COLOR_RESET}
    output_directory    Output directory for simulation results
                       (default: ${DEFAULT_OUTPUT_DIR})

${COLOR_BOLD}OPTIONS${COLOR_RESET}
    -h, --help                 Show this usage information and exit
    -v, --version             Show version information and exit
    -c, --config DIR          Configuration directory path (default: ${DEFAULT_CONFIG_DIR})
    -p, --python EXEC         Python executable path (default: ${DEFAULT_PYTHON_EXECUTABLE})
    -b, --batch-size NUM      Number of simulations to execute (default: ${BATCH_SIZE_DEFAULT})
    -l, --log-level LEVEL     Logging level: DEBUG,INFO,WARNING,ERROR (default: ${DEFAULT_LOG_LEVEL})
    -a, --algorithm CONFIG    Algorithm configuration file or preset name
    -j, --parallel            Enable parallel processing (default: enabled)
    -s, --serial              Force serial processing (disable parallel)
    -m, --monitor-interval    Progress monitoring interval in seconds (default: 5)
    --strict-validation       Enable strict prerequisite validation
    --skip-performance-check  Skip performance requirement validation
    --no-colors               Disable color output for non-interactive environments
    --preserve-intermediate   Preserve intermediate files after completion
    --no-preserve-logs        Do not preserve log files after cleanup

${COLOR_BOLD}PERFORMANCE OPTIONS${COLOR_RESET}
    --memory-limit GB         Maximum memory usage limit (default: auto-detect)
    --cpu-cores NUM           Number of CPU cores to utilize (default: auto-detect)
    --disk-space-check        Validate available disk space before execution

${COLOR_BOLD}ALGORITHM SELECTION${COLOR_RESET}
    --algorithm-preset NAME   Use predefined algorithm configuration:
                             - crimaldi_standard
                             - custom_optimized
                             - performance_test
                             - accuracy_validation

${COLOR_BOLD}EXAMPLES${COLOR_RESET}
    ${COLOR_CYAN}# Basic batch execution with default settings${COLOR_RESET}
    $0 /path/to/plume/data

    ${COLOR_CYAN}# Custom configuration with parallel processing${COLOR_RESET}
    $0 -c ./config -b 5000 --parallel /data/plumes /results

    ${COLOR_CYAN}# Performance validation with strict checks${COLOR_RESET}
    $0 --strict-validation --algorithm-preset performance_test /test/data

    ${COLOR_CYAN}# Serial processing for debugging${COLOR_RESET}
    $0 --serial --log-level DEBUG /debug/data /debug/output

${COLOR_BOLD}PERFORMANCE TARGETS${COLOR_RESET}
    - Complete 4000 simulations within 8 hours
    - Average processing time: <7.2 seconds per simulation
    - Correlation accuracy: >95% with reference implementations
    - Reproducibility coefficient: >99% across environments

${COLOR_BOLD}SYSTEM REQUIREMENTS${COLOR_RESET}
    - Python 3.9+ with scientific computing packages
    - Minimum ${REQUIRED_MEMORY_GB}GB RAM available
    - Minimum ${REQUIRED_DISK_GB}GB free disk space
    - Backend module accessibility: ${BACKEND_MODULE_PATH}

${COLOR_BOLD}TROUBLESHOOTING${COLOR_RESET}
    - Validation errors: Check Python environment and dependencies
    - Resource errors: Verify available memory and disk space
    - Performance issues: Consider enabling parallel processing
    - Format errors: Validate input data compatibility

${COLOR_BOLD}DOCUMENTATION${COLOR_RESET}
    Technical Specification: See project documentation
    Algorithm Details: Refer to backend module documentation
    Performance Analysis: Check output summary reports

EOF
}

show_version() {
    cat << EOF
${COLOR_BOLD}${SCRIPT_NAME} v${SCRIPT_VERSION}${COLOR_RESET}

${COLOR_BOLD}Build Information${COLOR_RESET}
    Script Version:          ${SCRIPT_VERSION}
    Backend Module:          ${BACKEND_MODULE_PATH}
    Required Python:         3.9+
    Scientific Computing:    NumPy 2.1.3+, SciPy 1.15.3+, OpenCV 4.11.0+

${COLOR_BOLD}Performance Specifications${COLOR_RESET}
    Target Batch Size:       ${BATCH_SIZE_DEFAULT} simulations
    Completion Target:       ${TARGET_COMPLETION_HOURS} hours
    Average Simulation Time: <${TARGET_SIMULATION_SECONDS} seconds
    Memory Requirement:      ${REQUIRED_MEMORY_GB}GB minimum
    Disk Space Requirement:  ${REQUIRED_DISK_GB}GB minimum

${COLOR_BOLD}Platform Compatibility${COLOR_RESET}
    Supported Formats:       Crimaldi dataset, custom AVI recordings
    Operating Systems:       Linux, macOS, Windows (with bash)
    Processing Modes:        Serial and parallel execution
    Resource Management:     Automatic detection and optimization

${COLOR_BOLD}Scientific Computing Standards${COLOR_RESET}
    Correlation Accuracy:    >95% target
    Reproducibility:         >99% coefficient target
    Cross-Platform Support:  Standardized configuration management
    Error Handling:          Fail-fast validation with recovery mechanisms

${COLOR_BOLD}Integration Compatibility${COLOR_RESET}
    Video Processing:        OpenCV 4.11.0+ for format handling
    Numerical Computation:   NumPy, SciPy for algorithm execution
    Statistical Analysis:    Pandas 2.2.0+ for result validation
    Parallel Processing:     Joblib 1.6.0+ for resource optimization

EOF
}

log_message() {
    local log_level="$1"
    local message="$2"
    local use_colors="${3:-$ENABLE_COLORS}"
    
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    
    local color_code=""
    local output_stream="/dev/stdout"
    
    if [[ "$use_colors" == "true" ]]; then
        case "$log_level" in
            "ERROR")   color_code="$COLOR_RED"; output_stream="/dev/stderr" ;;
            "WARNING") color_code="$COLOR_YELLOW" ;;
            "SUCCESS") color_code="$COLOR_GREEN" ;;
            "INFO")    color_code="$COLOR_BLUE" ;;
            "DEBUG")   color_code="$COLOR_CYAN" ;;
            *)         color_code="$COLOR_RESET" ;;
        esac
    fi
    
    local formatted_message
    if [[ "$use_colors" == "true" ]]; then
        formatted_message="${color_code}[${timestamp}] ${log_level}: ${message}${COLOR_RESET}"
    else
        formatted_message="[${timestamp}] ${log_level}: ${message}"
    fi
    
    echo -e "$formatted_message" > "$output_stream"
    
    # Flush output for immediate display
    if [[ "$output_stream" == "/dev/stderr" ]]; then
        exec 2>&2
    else
        exec 1>&1
    fi
}

# =============================================================================
# VALIDATION AND SETUP FUNCTIONS
# =============================================================================

check_prerequisites() {
    local strict_validation="$1"
    local skip_performance_check="$2"
    
    log_message "INFO" "Starting comprehensive prerequisite validation..."
    
    # Check Python executable availability and version
    log_message "INFO" "Validating Python environment..."
    if ! command -v "$PYTHON_EXEC" &> /dev/null; then
        log_message "ERROR" "Python executable not found: $PYTHON_EXEC"
        return $EXIT_VALIDATION_ERROR
    fi
    
    local python_version
    python_version=$("$PYTHON_EXEC" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
        log_message "ERROR" "Python 3.9+ required, found: $python_version"
        return $EXIT_VALIDATION_ERROR
    fi
    log_message "SUCCESS" "Python version validated: $python_version"
    
    # Validate system resource availability
    log_message "INFO" "Checking system resource availability..."
    
    # Memory check
    local available_memory_gb
    if command -v free &> /dev/null; then
        available_memory_gb=$(free -g | awk '/^Mem:/{print $7}')
    elif command -v vm_stat &> /dev/null; then
        # macOS memory check
        local free_pages
        free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        available_memory_gb=$((free_pages * 4096 / 1024 / 1024 / 1024))
    else
        log_message "WARNING" "Unable to determine available memory"
        available_memory_gb=0
    fi
    
    if [[ "$skip_performance_check" != "true" && "$available_memory_gb" -lt "$REQUIRED_MEMORY_GB" ]]; then
        log_message "ERROR" "Insufficient memory: ${available_memory_gb}GB available, ${REQUIRED_MEMORY_GB}GB required"
        return $EXIT_RESOURCE_ERROR
    fi
    log_message "SUCCESS" "Memory check passed: ${available_memory_gb}GB available"
    
    # Disk space check
    local available_disk_gb
    available_disk_gb=$(df -BG "$OUTPUT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ -z "$available_disk_gb" ]]; then
        available_disk_gb=$(df -h "$OUTPUT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    if [[ "$skip_performance_check" != "true" && "$available_disk_gb" -lt "$REQUIRED_DISK_GB" ]]; then
        log_message "ERROR" "Insufficient disk space: ${available_disk_gb}GB available, ${REQUIRED_DISK_GB}GB required"
        return $EXIT_RESOURCE_ERROR
    fi
    log_message "SUCCESS" "Disk space check passed: ${available_disk_gb}GB available"
    
    # Check backend module accessibility
    log_message "INFO" "Validating backend module accessibility..."
    if ! "$PYTHON_EXEC" -c "import $BACKEND_MODULE_PATH" 2>/dev/null; then
        log_message "ERROR" "Backend module not accessible: $BACKEND_MODULE_PATH"
        return $EXIT_VALIDATION_ERROR
    fi
    log_message "SUCCESS" "Backend module validated: $BACKEND_MODULE_PATH"
    
    # Validate configuration directory
    if [[ ! -d "$CONFIG_DIR" ]]; then
        log_message "WARNING" "Configuration directory not found: $CONFIG_DIR"
        if [[ "$strict_validation" == "true" ]]; then
            return $EXIT_CONFIG_ERROR
        fi
    else
        log_message "SUCCESS" "Configuration directory validated: $CONFIG_DIR"
    fi
    
    # Run environment validation script if available
    local validation_script_path="$CONFIG_DIR/$VALIDATION_SCRIPT"
    if [[ -f "$validation_script_path" ]]; then
        log_message "INFO" "Running environment validation script..."
        if ! "$PYTHON_EXEC" "$validation_script_path" 2>/dev/null; then
            log_message "WARNING" "Environment validation script reported issues"
            if [[ "$strict_validation" == "true" ]]; then
                return $EXIT_VALIDATION_ERROR
            fi
        else
            log_message "SUCCESS" "Environment validation script passed"
        fi
    fi
    
    # Check dependency versions and compatibility
    log_message "INFO" "Validating scientific computing dependencies..."
    local dependencies=("numpy>=2.1.3" "scipy>=1.15.3" "opencv-python>=4.11.0" "pandas>=2.2.0")
    for dep in "${dependencies[@]}"; do
        local package_name
        package_name=$(echo "$dep" | cut -d'>=' -f1)
        if ! "$PYTHON_EXEC" -c "import $package_name" 2>/dev/null; then
            log_message "WARNING" "Dependency not found: $package_name"
            if [[ "$strict_validation" == "true" ]]; then
                return $EXIT_VALIDATION_ERROR
            fi
        else
            log_message "SUCCESS" "Dependency validated: $package_name"
        fi
    done
    
    # Check write permissions for output directories
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
            log_message "ERROR" "Cannot create output directory: $OUTPUT_DIR"
            return $EXIT_CONFIG_ERROR
        fi
    fi
    
    if [[ ! -w "$OUTPUT_DIR" ]]; then
        log_message "ERROR" "No write permission for output directory: $OUTPUT_DIR"
        return $EXIT_CONFIG_ERROR
    fi
    log_message "SUCCESS" "Output directory permissions validated: $OUTPUT_DIR"
    
    log_message "SUCCESS" "All prerequisite validations completed successfully"
    return $EXIT_SUCCESS
}

validate_arguments() {
    local -a script_arguments=("$@")
    
    log_message "INFO" "Validating command-line arguments and configuration..."
    
    # Check for minimum required arguments
    if [[ ${#script_arguments[@]} -lt 1 ]]; then
        log_message "ERROR" "Missing required argument: input_data_path"
        show_usage
        return $EXIT_CONFIG_ERROR
    fi
    
    local input_data_path="${script_arguments[0]}"
    local output_directory="${script_arguments[1]:-$OUTPUT_DIR}"
    
    # Validate input data path existence and accessibility
    if [[ ! -e "$input_data_path" ]]; then
        log_message "ERROR" "Input data path does not exist: $input_data_path"
        return $EXIT_CONFIG_ERROR
    fi
    
    if [[ ! -r "$input_data_path" ]]; then
        log_message "ERROR" "Input data path not readable: $input_data_path"
        return $EXIT_CONFIG_ERROR
    fi
    log_message "SUCCESS" "Input data path validated: $input_data_path"
    
    # Validate and create output directory if necessary
    if [[ ! -d "$output_directory" ]]; then
        log_message "INFO" "Creating output directory: $output_directory"
        if ! mkdir -p "$output_directory" 2>/dev/null; then
            log_message "ERROR" "Cannot create output directory: $output_directory"
            return $EXIT_CONFIG_ERROR
        fi
    fi
    
    if [[ ! -w "$output_directory" ]]; then
        log_message "ERROR" "Output directory not writable: $output_directory"
        return $EXIT_CONFIG_ERROR
    fi
    log_message "SUCCESS" "Output directory validated: $output_directory"
    
    # Validate batch size against system capabilities
    if [[ "$BATCH_SIZE" -lt 1 ]]; then
        log_message "ERROR" "Invalid batch size: $BATCH_SIZE (must be >= 1)"
        return $EXIT_CONFIG_ERROR
    fi
    
    if [[ "$BATCH_SIZE" -gt 10000 ]]; then
        log_message "WARNING" "Large batch size detected: $BATCH_SIZE (may impact performance)"
    fi
    log_message "SUCCESS" "Batch size validated: $BATCH_SIZE"
    
    # Validate configuration file paths and formats
    if [[ -d "$CONFIG_DIR" ]]; then
        local config_files
        config_files=$(find "$CONFIG_DIR" -name "*.json" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l)
        if [[ "$config_files" -eq 0 ]]; then
            log_message "WARNING" "No configuration files found in: $CONFIG_DIR"
        else
            log_message "SUCCESS" "Configuration files found: $config_files files in $CONFIG_DIR"
        fi
    fi
    
    # Validate monitoring interval
    if [[ "$MONITORING_INTERVAL" -lt 1 ]]; then
        log_message "WARNING" "Invalid monitoring interval: $MONITORING_INTERVAL, using default: 5"
        MONITORING_INTERVAL=5
    fi
    
    log_message "SUCCESS" "All arguments validated successfully"
    return $EXIT_SUCCESS
}

setup_environment() {
    local config_directory="$1"
    local output_directory="$2"
    local log_level="$3"
    
    log_message "INFO" "Setting up execution environment for batch simulation..."
    
    # Configure Python path for backend module access
    local backend_path
    backend_path=$(dirname "$(dirname "$(realpath "$0")")")
    export PYTHONPATH="${backend_path}:${PYTHONPATH:-}"
    log_message "SUCCESS" "Python path configured: $PYTHONPATH"
    
    # Set environment variables for configuration paths
    export PLUME_CONFIG_DIR="$config_directory"
    export PLUME_OUTPUT_DIR="$output_directory"
    export PLUME_LOG_LEVEL="$log_level"
    export PLUME_BATCH_SIZE="$BATCH_SIZE"
    log_message "SUCCESS" "Environment variables configured"
    
    # Setup logging configuration and output redirection
    local log_file="$output_directory/batch_simulation_$(date +%Y%m%d_%H%M%S).log"
    exec 1> >(tee -a "$log_file")
    exec 2> >(tee -a "$log_file" >&2)
    log_message "SUCCESS" "Logging configured: $log_file"
    
    # Configure resource limits and allocation
    if command -v ulimit &> /dev/null; then
        # Set reasonable memory limits
        ulimit -v $((REQUIRED_MEMORY_GB * 1024 * 1024)) 2>/dev/null || true
        log_message "INFO" "Resource limits configured"
    fi
    
    # Setup temporary directory for intermediate files
    local temp_dir="$output_directory/temp_$$"
    if ! mkdir -p "$temp_dir" 2>/dev/null; then
        log_message "ERROR" "Cannot create temporary directory: $temp_dir"
        return $EXIT_CONFIG_ERROR
    fi
    export PLUME_TEMP_DIR="$temp_dir"
    log_message "SUCCESS" "Temporary directory created: $temp_dir"
    
    # Configure parallel processing environment
    if [[ "$ENABLE_PARALLEL" == "true" ]]; then
        local cpu_count
        cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
        export PLUME_CPU_CORES="$cpu_count"
        export OMP_NUM_THREADS="$cpu_count"
        log_message "SUCCESS" "Parallel processing configured: $cpu_count cores"
    else
        export PLUME_CPU_CORES="1"
        export OMP_NUM_THREADS="1"
        log_message "INFO" "Serial processing mode configured"
    fi
    
    # Set scientific computing precision and optimization flags
    export PYTHONOPTIMIZE="1"
    export PYTHONHASHSEED="0"  # For reproducible results
    log_message "SUCCESS" "Scientific computing environment optimized"
    
    # Initialize performance monitoring environment
    EXECUTION_START_TIME=$(date +%s)
    export PLUME_START_TIME="$EXECUTION_START_TIME"
    log_message "SUCCESS" "Performance monitoring initialized"
    
    log_message "SUCCESS" "Environment setup completed successfully"
    return $EXIT_SUCCESS
}

# =============================================================================
# EXECUTION AND MONITORING FUNCTIONS
# =============================================================================

execute_batch_simulation() {
    local input_data_path="$1"
    local output_directory="$2"
    local algorithm_config="$3"
    local batch_size="$4"
    local enable_parallel="$5"
    local additional_options="$6"
    
    log_message "INFO" "Starting batch simulation execution..."
    log_message "INFO" "Input: $input_data_path"
    log_message "INFO" "Output: $output_directory"
    log_message "INFO" "Batch size: $batch_size"
    log_message "INFO" "Parallel processing: $enable_parallel"
    
    # Construct Python command with backend module and parameters
    local python_cmd=("$PYTHON_EXEC" "-m" "$BACKEND_MODULE_PATH")
    
    # Add input and output parameters
    python_cmd+=("--input" "$input_data_path")
    python_cmd+=("--output" "$output_directory")
    python_cmd+=("--batch-size" "$batch_size")
    
    # Add configuration options
    if [[ -n "$algorithm_config" ]]; then
        python_cmd+=("--algorithm-config" "$algorithm_config")
    fi
    
    if [[ -d "$CONFIG_DIR" ]]; then
        python_cmd+=("--config-dir" "$CONFIG_DIR")
    fi
    
    # Configure parallel processing and resource allocation
    if [[ "$enable_parallel" == "true" ]]; then
        python_cmd+=("--parallel")
        if [[ -n "$PLUME_CPU_CORES" ]]; then
            python_cmd+=("--cpu-cores" "$PLUME_CPU_CORES")
        fi
    else
        python_cmd+=("--serial")
    fi
    
    # Add logging and monitoring options
    python_cmd+=("--log-level" "$LOG_LEVEL")
    python_cmd+=("--progress-monitoring")
    
    # Add additional options if provided
    if [[ -n "$additional_options" ]]; then
        IFS=' ' read -ra ADDR <<< "$additional_options"
        python_cmd+=("${ADDR[@]}")
    fi
    
    log_message "INFO" "Executing command: ${python_cmd[*]}"
    
    # Execute Python backend with comprehensive error handling
    local execution_log="$output_directory/execution_$(date +%Y%m%d_%H%M%S).log"
    
    # Start execution in background for monitoring
    "${python_cmd[@]}" > "$execution_log" 2>&1 &
    PYTHON_PID=$!
    
    log_message "INFO" "Python backend started with PID: $PYTHON_PID"
    log_message "INFO" "Execution log: $execution_log"
    
    # Monitor execution progress in background
    monitor_execution "$PYTHON_PID" "$output_directory" "$MONITORING_INTERVAL" &
    local monitor_pid=$!
    
    # Wait for Python process completion
    local exit_code=0
    if wait "$PYTHON_PID"; then
        log_message "SUCCESS" "Batch simulation completed successfully"
        exit_code=$EXIT_SUCCESS
    else
        exit_code=$?
        log_message "ERROR" "Batch simulation failed with exit code: $exit_code"
        exit_code=$EXIT_SIMULATION_ERROR
    fi
    
    # Stop monitoring
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    
    # Process execution output and errors
    if [[ -f "$execution_log" ]]; then
        local error_count
        error_count=$(grep -c "ERROR" "$execution_log" 2>/dev/null || echo "0")
        local warning_count
        warning_count=$(grep -c "WARNING" "$execution_log" 2>/dev/null || echo "0")
        
        log_message "INFO" "Execution summary: $error_count errors, $warning_count warnings"
        
        if [[ "$error_count" -gt 0 ]]; then
            log_message "WARNING" "Errors detected in execution log: $execution_log"
        fi
    fi
    
    return $exit_code
}

monitor_execution() {
    local python_process_id="$1"
    local output_directory="$2"
    local monitoring_interval="$3"
    
    log_message "INFO" "Starting execution monitoring (PID: $python_process_id, interval: ${monitoring_interval}s)"
    
    local start_time
    start_time=$(date +%s)
    local last_progress=0
    
    while kill -0 "$python_process_id" 2>/dev/null; do
        sleep "$monitoring_interval"
        
        # Calculate elapsed time
        local current_time
        current_time=$(date +%s)
        local elapsed_seconds=$((current_time - start_time))
        local elapsed_hours=$((elapsed_seconds / 3600))
        local elapsed_minutes=$(((elapsed_seconds % 3600) / 60))
        local elapsed_secs=$((elapsed_seconds % 60))
        
        # Monitor simulation progress from output files
        local progress_file="$output_directory/progress.txt"
        local current_progress=0
        if [[ -f "$progress_file" ]]; then
            current_progress=$(tail -1 "$progress_file" 2>/dev/null | grep -o '[0-9]\+' | head -1 || echo "0")
        fi
        
        # Calculate progress metrics
        local progress_percentage=0
        if [[ "$BATCH_SIZE" -gt 0 ]]; then
            progress_percentage=$((current_progress * 100 / BATCH_SIZE))
        fi
        
        local simulations_since_last=$((current_progress - last_progress))
        local avg_time_per_simulation=0
        if [[ "$current_progress" -gt 0 && "$elapsed_seconds" -gt 0 ]]; then
            avg_time_per_simulation=$((elapsed_seconds * 100 / current_progress))
            avg_time_per_simulation=$((avg_time_per_simulation))
        fi
        
        # Calculate ETA
        local remaining_simulations=$((BATCH_SIZE - current_progress))
        local eta_seconds=0
        if [[ "$avg_time_per_simulation" -gt 0 && "$remaining_simulations" -gt 0 ]]; then
            eta_seconds=$((remaining_simulations * avg_time_per_simulation / 100))
        fi
        local eta_hours=$((eta_seconds / 3600))
        local eta_minutes=$(((eta_seconds % 3600) / 60))
        
        # Display progress with ASCII progress bar
        local bar_width=50
        local filled_width=$((progress_percentage * bar_width / 100))
        local bar=""
        for ((i=0; i<filled_width; i++)); do bar+="█"; done
        for ((i=filled_width; i<bar_width; i++)); do bar+="░"; done
        
        # Monitor system resource utilization
        local memory_usage="N/A"
        local cpu_usage="N/A"
        if command -v ps &> /dev/null; then
            memory_usage=$(ps -p "$python_process_id" -o rss= 2>/dev/null | awk '{print int($1/1024)" MB"}' || echo "N/A")
            cpu_usage=$(ps -p "$python_process_id" -o %cpu= 2>/dev/null | awk '{print $1"%"}' || echo "N/A")
        fi
        
        # Display comprehensive progress information
        printf "\r${COLOR_CYAN}Progress: ${COLOR_RESET}[${COLOR_GREEN}%s${COLOR_RESET}] ${COLOR_BOLD}%d%%${COLOR_RESET} (%d/%d)" \
               "$bar" "$progress_percentage" "$current_progress" "$BATCH_SIZE"
        
        printf " ${COLOR_BLUE}Time: ${COLOR_RESET}%02d:%02d:%02d" \
               "$elapsed_hours" "$elapsed_minutes" "$elapsed_secs"
        
        if [[ "$eta_seconds" -gt 0 ]]; then
            printf " ${COLOR_YELLOW}ETA: ${COLOR_RESET}%02d:%02d" "$eta_hours" "$eta_minutes"
        fi
        
        printf " ${COLOR_CYAN}Mem: ${COLOR_RESET}%s ${COLOR_CYAN}CPU: ${COLOR_RESET}%s" \
               "$memory_usage" "$cpu_usage"
        
        if [[ "$avg_time_per_simulation" -gt 0 ]]; then
            printf " ${COLOR_YELLOW}Avg: ${COLOR_RESET}%.1fs/sim" \
                   "$(echo "scale=1; $avg_time_per_simulation / 100" | bc -l 2>/dev/null || echo "$avg_time_per_simulation")"
        fi
        
        # Check for performance target compliance
        if [[ "$avg_time_per_simulation" -gt $((${TARGET_SIMULATION_SECONDS//.}0)) ]]; then
            printf " ${COLOR_RED}(SLOW)${COLOR_RESET}"
        fi
        
        last_progress="$current_progress"
        
        # Check for error conditions and warnings
        local error_file="$output_directory/errors.log"
        if [[ -f "$error_file" ]]; then
            local new_errors
            new_errors=$(find "$error_file" -newer "$progress_file" 2>/dev/null | wc -l)
            if [[ "$new_errors" -gt 0 ]]; then
                printf " ${COLOR_RED}ERRORS!${COLOR_RESET}"
            fi
        fi
    done
    
    printf "\n"
    log_message "SUCCESS" "Execution monitoring completed"
}

handle_interruption() {
    local signal_number="$1"
    local python_process_id="$2"
    
    INTERRUPTED=true
    
    log_message "WARNING" "Received interruption signal: $signal_number"
    log_message "INFO" "Initiating graceful shutdown..."
    
    # Send termination signal to Python backend process
    if [[ "$python_process_id" -gt 0 ]]; then
        log_message "INFO" "Terminating Python backend process (PID: $python_process_id)..."
        kill -TERM "$python_process_id" 2>/dev/null || true
        
        # Wait for backend process cleanup with timeout
        local timeout=30
        local count=0
        while kill -0 "$python_process_id" 2>/dev/null && [[ "$count" -lt "$timeout" ]]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$python_process_id" 2>/dev/null; then
            log_message "WARNING" "Force killing unresponsive process..."
            kill -KILL "$python_process_id" 2>/dev/null || true
        fi
    fi
    
    # Preserve partial simulation results if available
    if [[ -d "$OUTPUT_DIR" ]]; then
        local partial_results
        partial_results=$(find "$OUTPUT_DIR" -name "*.json" -o -name "*.csv" 2>/dev/null | wc -l)
        if [[ "$partial_results" -gt 0 ]]; then
            log_message "INFO" "Preserving $partial_results partial result files"
            
            local backup_dir="$OUTPUT_DIR/interrupted_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$backup_dir" 2>/dev/null || true
            cp "$OUTPUT_DIR"/*.{json,csv} "$backup_dir/" 2>/dev/null || true
        fi
    fi
    
    # Generate interruption summary and status report
    local current_time
    current_time=$(date +%s)
    local execution_time=$((current_time - EXECUTION_START_TIME))
    
    cat << EOF

${COLOR_YELLOW}=== INTERRUPTION SUMMARY ===${COLOR_RESET}
Signal Received: $signal_number
Execution Time: $((execution_time / 3600))h $((execution_time % 3600 / 60))m $((execution_time % 60))s
Partial Results: Preserved in $OUTPUT_DIR
Status: Graceful shutdown completed

EOF
    
    log_message "INFO" "Cleanup completion status: SUCCESS"
    
    # Exit with appropriate interruption exit code
    exit $EXIT_FAILURE
}

# =============================================================================
# VALIDATION AND REPORTING FUNCTIONS
# =============================================================================

validate_results() {
    local output_directory="$1"
    local expected_simulation_count="$2"
    local strict_validation="$3"
    
    log_message "INFO" "Validating batch simulation results..."
    
    # Check simulation result files existence and completeness
    local result_files
    result_files=$(find "$output_directory" -name "simulation_*.json" 2>/dev/null | wc -l)
    
    log_message "INFO" "Found $result_files simulation result files"
    
    # Validate simulation completion count against expected
    local completion_percentage=0
    if [[ "$expected_simulation_count" -gt 0 ]]; then
        completion_percentage=$((result_files * 100 / expected_simulation_count))
    fi
    
    log_message "INFO" "Completion rate: $completion_percentage% ($result_files/$expected_simulation_count)"
    
    # Check performance metrics against target thresholds
    local performance_file="$output_directory/performance_summary.json"
    local avg_simulation_time=0
    
    if [[ -f "$performance_file" ]]; then
        avg_simulation_time=$(grep -o '"avg_time": [0-9.]*' "$performance_file" 2>/dev/null | cut -d' ' -f2 || echo "0")
        log_message "INFO" "Average simulation time: ${avg_simulation_time}s (target: <${TARGET_SIMULATION_SECONDS}s)"
        
        # Validate performance against targets
        if [[ $(echo "$avg_simulation_time > $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
            log_message "WARNING" "Performance target not met: ${avg_simulation_time}s > ${TARGET_SIMULATION_SECONDS}s"
            if [[ "$strict_validation" == "true" ]]; then
                return $EXIT_SIMULATION_ERROR
            fi
        else
            log_message "SUCCESS" "Performance target achieved"
        fi
    else
        log_message "WARNING" "Performance summary file not found: $performance_file"
    fi
    
    # Validate result file formats and data integrity
    local valid_files=0
    local invalid_files=0
    
    while IFS= read -r -d '' file; do
        if "$PYTHON_EXEC" -c "import json; json.load(open('$file'))" 2>/dev/null; then
            valid_files=$((valid_files + 1))
        else
            invalid_files=$((invalid_files + 1))
            log_message "WARNING" "Invalid result file format: $file"
        fi
    done < <(find "$output_directory" -name "simulation_*.json" -print0 2>/dev/null)
    
    log_message "INFO" "File validation: $valid_files valid, $invalid_files invalid"
    
    # Check error rates and failure analysis
    local error_log="$output_directory/errors.log"
    local error_count=0
    if [[ -f "$error_log" ]]; then
        error_count=$(wc -l < "$error_log" 2>/dev/null || echo "0")
    fi
    
    local error_rate=0
    if [[ "$expected_simulation_count" -gt 0 ]]; then
        error_rate=$((error_count * 100 / expected_simulation_count))
    fi
    
    log_message "INFO" "Error analysis: $error_count errors, ${error_rate}% error rate"
    
    # Overall validation assessment
    local validation_score=0
    
    # Completion rate scoring (40% weight)
    if [[ "$completion_percentage" -ge 95 ]]; then
        validation_score=$((validation_score + 40))
    elif [[ "$completion_percentage" -ge 90 ]]; then
        validation_score=$((validation_score + 30))
    elif [[ "$completion_percentage" -ge 80 ]]; then
        validation_score=$((validation_score + 20))
    fi
    
    # Performance scoring (30% weight)
    if [[ $(echo "$avg_simulation_time <= $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        validation_score=$((validation_score + 30))
    elif [[ $(echo "$avg_simulation_time <= ${TARGET_SIMULATION_SECONDS} * 1.2" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        validation_score=$((validation_score + 20))
    fi
    
    # Data integrity scoring (20% weight)
    if [[ "$invalid_files" -eq 0 ]]; then
        validation_score=$((validation_score + 20))
    elif [[ "$invalid_files" -lt 5 ]]; then
        validation_score=$((validation_score + 10))
    fi
    
    # Error rate scoring (10% weight)
    if [[ "$error_rate" -lt 1 ]]; then
        validation_score=$((validation_score + 10))
    elif [[ "$error_rate" -lt 5 ]]; then
        validation_score=$((validation_score + 5))
    fi
    
    log_message "INFO" "Overall validation score: $validation_score/100"
    
    # Generate result validation report
    local validation_report="$output_directory/validation_report.txt"
    cat > "$validation_report" << EOF
BATCH SIMULATION VALIDATION REPORT
Generated: $(date)

COMPLETION ANALYSIS:
- Simulations completed: $result_files/$expected_simulation_count ($completion_percentage%)
- Target completion rate: 100%
- Status: $(if [[ "$completion_percentage" -ge 95 ]]; then echo "PASS"; else echo "NEEDS REVIEW"; fi)

PERFORMANCE ANALYSIS:
- Average simulation time: ${avg_simulation_time}s
- Target simulation time: <${TARGET_SIMULATION_SECONDS}s
- Status: $(if [[ $(echo "$avg_simulation_time <= $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then echo "PASS"; else echo "NEEDS OPTIMIZATION"; fi)

DATA INTEGRITY ANALYSIS:
- Valid result files: $valid_files
- Invalid result files: $invalid_files
- File format compliance: $(if [[ "$invalid_files" -eq 0 ]]; then echo "100%"; else echo "$((valid_files * 100 / (valid_files + invalid_files)))%"; fi)

ERROR ANALYSIS:
- Total errors: $error_count
- Error rate: ${error_rate}%
- Target error rate: <1%

OVERALL ASSESSMENT:
- Validation score: $validation_score/100
- Status: $(if [[ "$validation_score" -ge 80 ]]; then echo "PASS"; elif [[ "$validation_score" -ge 60 ]]; then echo "ACCEPTABLE"; else echo "NEEDS IMPROVEMENT"; fi)

RECOMMENDATIONS:
$(if [[ "$completion_percentage" -lt 95 ]]; then echo "- Investigate incomplete simulations"; fi)
$(if [[ $(echo "$avg_simulation_time > $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then echo "- Optimize processing performance"; fi)
$(if [[ "$invalid_files" -gt 0 ]]; then echo "- Review and repair invalid result files"; fi)
$(if [[ "$error_rate" -gt 1 ]]; then echo "- Analyze and address error conditions"; fi)
EOF
    
    log_message "SUCCESS" "Validation report generated: $validation_report"
    
    # Determine validation exit code
    if [[ "$validation_score" -ge 80 ]]; then
        log_message "SUCCESS" "Result validation passed with score: $validation_score/100"
        return $EXIT_SUCCESS
    elif [[ "$validation_score" -ge 60 ]]; then
        log_message "WARNING" "Result validation acceptable with score: $validation_score/100"
        if [[ "$strict_validation" == "true" ]]; then
            return $EXIT_SIMULATION_ERROR
        else
            return $EXIT_SUCCESS
        fi
    else
        log_message "ERROR" "Result validation failed with score: $validation_score/100"
        return $EXIT_SIMULATION_ERROR
    fi
}

generate_summary_report() {
    local output_directory="$1"
    local execution_time_hours="$2"
    local total_simulations="$3"
    local include_detailed_analysis="$4"
    
    log_message "INFO" "Generating comprehensive execution summary report..."
    
    local current_time
    current_time=$(date)
    local execution_time_seconds
    execution_time_seconds=$(echo "$execution_time_hours * 3600" | bc -l 2>/dev/null || echo "0")
    
    # Collect execution statistics and performance metrics
    local completed_simulations
    completed_simulations=$(find "$output_directory" -name "simulation_*.json" 2>/dev/null | wc -l)
    
    local success_rate=0
    if [[ "$total_simulations" -gt 0 ]]; then
        success_rate=$((completed_simulations * 100 / total_simulations))
    fi
    
    local avg_time_per_simulation=0
    if [[ "$completed_simulations" -gt 0 && $(echo "$execution_time_seconds > 0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        avg_time_per_simulation=$(echo "scale=2; $execution_time_seconds / $completed_simulations" | bc -l 2>/dev/null || echo "0")
    fi
    
    # Calculate performance metrics
    local target_met="NO"
    if [[ $(echo "$avg_time_per_simulation <= $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        target_met="YES"
    fi
    
    local completion_target_met="NO"
    if [[ $(echo "$execution_time_hours <= $TARGET_COMPLETION_HOURS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        completion_target_met="YES"
    fi
    
    # Analyze error statistics
    local error_count=0
    local warning_count=0
    local error_log="$output_directory/errors.log"
    
    if [[ -f "$error_log" ]]; then
        error_count=$(grep -c "ERROR" "$error_log" 2>/dev/null || echo "0")
        warning_count=$(grep -c "WARNING" "$error_log" 2>/dev/null || echo "0")
    fi
    
    # Generate comprehensive summary report
    local summary_report="$output_directory/execution_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$summary_report" << EOF
===============================================================================
BATCH SIMULATION EXECUTION SUMMARY REPORT
===============================================================================

EXECUTION OVERVIEW:
Generated: $current_time
Script Version: $SCRIPT_VERSION
Backend Module: $BACKEND_MODULE_PATH

CONFIGURATION:
- Total Simulations Requested: $total_simulations
- Batch Processing Mode: $(if [[ "$ENABLE_PARALLEL" == "true" ]]; then echo "Parallel"; else echo "Serial"; fi)
- CPU Cores Utilized: ${PLUME_CPU_CORES:-"Auto-detected"}
- Configuration Directory: $CONFIG_DIR
- Output Directory: $output_directory

EXECUTION RESULTS:
- Simulations Completed: $completed_simulations
- Success Rate: $success_rate%
- Total Execution Time: ${execution_time_hours}h ($(printf "%.0f" "$execution_time_seconds")s)
- Average Time per Simulation: ${avg_time_per_simulation}s

PERFORMANCE ANALYSIS:
- Target Simulation Time: <${TARGET_SIMULATION_SECONDS}s
- Target Achievement: $target_met
- Target Completion Time: <${TARGET_COMPLETION_HOURS}h
- Completion Target Met: $completion_target_met
- Processing Efficiency: $(if [[ "$completed_simulations" -gt 0 ]]; then echo "scale=1; $total_simulations * 100 / $completed_simulations" | bc -l 2>/dev/null || echo "N/A"; else echo "N/A"; fi)%

QUALITY METRICS:
- Simulation Errors: $error_count
- Warnings Generated: $warning_count
- Error Rate: $(if [[ "$total_simulations" -gt 0 ]]; then echo "scale=2; $error_count * 100 / $total_simulations" | bc -l 2>/dev/null || echo "0"; else echo "0"; fi)%
- Data Integrity: $(find "$output_directory" -name "simulation_*.json" | wc -l) valid result files

RESOURCE UTILIZATION:
- Peak Memory Usage: $(if [[ -f "$output_directory/resource_usage.log" ]]; then grep "Peak Memory" "$output_directory/resource_usage.log" | tail -1 | awk '{print $3 " " $4}' 2>/dev/null || echo "Not monitored"; else echo "Not monitored"; fi)
- Peak CPU Usage: $(if [[ -f "$output_directory/resource_usage.log" ]]; then grep "Peak CPU" "$output_directory/resource_usage.log" | tail -1 | awk '{print $3}' 2>/dev/null || echo "Not monitored"; else echo "Not monitored"; fi)
- Disk Space Used: $(du -sh "$output_directory" 2>/dev/null | cut -f1 || echo "Unknown")

SCIENTIFIC COMPUTING COMPLIANCE:
- Reproducibility: $(if [[ -f "$output_directory/reproducibility_report.txt" ]]; then grep "Coefficient:" "$output_directory/reproducibility_report.txt" | awk '{print $2}' 2>/dev/null || echo "Not validated"; else echo "Not validated"; fi)
- Cross-Platform Compatibility: $(if [[ -f "$output_directory/compatibility_report.txt" ]]; then echo "Validated"; else echo "Not tested"; fi)
- Algorithm Validation: $(if [[ -f "$output_directory/algorithm_validation.txt" ]]; then echo "Completed"; else echo "Not performed"; fi)

EOF

    # Add detailed analysis if requested
    if [[ "$include_detailed_analysis" == "true" ]]; then
        cat >> "$summary_report" << EOF

DETAILED PERFORMANCE BREAKDOWN:
- Initialization Time: $(if [[ -f "$output_directory/timing_breakdown.log" ]]; then grep "Initialization" "$output_directory/timing_breakdown.log" | awk '{print $2}' 2>/dev/null || echo "Not tracked"; else echo "Not tracked"; fi)
- Data Loading Time: $(if [[ -f "$output_directory/timing_breakdown.log" ]]; then grep "Data Loading" "$output_directory/timing_breakdown.log" | awk '{print $3}' 2>/dev/null || echo "Not tracked"; else echo "Not tracked"; fi)
- Simulation Execution Time: $(if [[ -f "$output_directory/timing_breakdown.log" ]]; then grep "Simulation" "$output_directory/timing_breakdown.log" | awk '{print $3}' 2>/dev/null || echo "Not tracked"; else echo "Not tracked"; fi)
- Result Processing Time: $(if [[ -f "$output_directory/timing_breakdown.log" ]]; then grep "Result Processing" "$output_directory/timing_breakdown.log" | awk '{print $3}' 2>/dev/null || echo "Not tracked"; else echo "Not tracked"; fi)

ALGORITHM PERFORMANCE ANALYSIS:
$(if [[ -f "$output_directory/algorithm_performance.txt" ]]; then cat "$output_directory/algorithm_performance.txt"; else echo "Detailed algorithm analysis not available"; fi)

ERROR CATEGORIZATION:
$(if [[ -f "$error_log" && "$error_count" -gt 0 ]]; then
    echo "Configuration Errors: $(grep -c "CONFIG" "$error_log" 2>/dev/null || echo "0")"
    echo "Data Format Errors: $(grep -c "FORMAT" "$error_log" 2>/dev/null || echo "0")"
    echo "Processing Errors: $(grep -c "PROCESSING" "$error_log" 2>/dev/null || echo "0")"
    echo "Resource Errors: $(grep -c "RESOURCE" "$error_log" 2>/dev/null || echo "0")"
else
    echo "No errors detected or error log not available"
fi)

EOF
    fi
    
    # Add optimization recommendations
    cat >> "$summary_report" << EOF

OPTIMIZATION RECOMMENDATIONS:
$(if [[ $(echo "$avg_time_per_simulation > $TARGET_SIMULATION_SECONDS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
    echo "- Consider enabling parallel processing to improve simulation speed"
    echo "- Review algorithm configuration for performance optimization opportunities"
    echo "- Validate system resources meet minimum requirements"
fi)
$(if [[ "$success_rate" -lt 95 ]]; then
    echo "- Investigate failed simulations for systematic issues"
    echo "- Review input data quality and format compliance"
    echo "- Consider implementing retry mechanisms for transient failures"
fi)
$(if [[ "$error_count" -gt 0 ]]; then
    echo "- Address error conditions identified in execution log"
    echo "- Implement additional validation checks for error prevention"
fi)
$(if [[ $(echo "$execution_time_hours > $TARGET_COMPLETION_HOURS" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
    echo "- Optimize batch processing configuration for faster completion"
    echo "- Consider resource scaling or distributed processing options"
fi)

NEXT STEPS:
1. Review execution summary and performance metrics
2. Address any optimization recommendations
3. Validate scientific reproducibility of results
4. Archive results and execution logs for future reference
5. Update processing parameters based on performance analysis

===============================================================================
Report generated by: $SCRIPT_NAME v$SCRIPT_VERSION
Contact: See project documentation for support information
===============================================================================
EOF
    
    log_message "SUCCESS" "Summary report generated: $summary_report"
    
    # Display summary highlights to console
    cat << EOF

${COLOR_BOLD}=== EXECUTION SUMMARY HIGHLIGHTS ===${COLOR_RESET}
${COLOR_CYAN}Completed:${COLOR_RESET} $completed_simulations/$total_simulations simulations ($success_rate%)
${COLOR_CYAN}Time:${COLOR_RESET} ${execution_time_hours}h (target: <${TARGET_COMPLETION_HOURS}h)
${COLOR_CYAN}Performance:${COLOR_RESET} ${avg_time_per_simulation}s avg (target: <${TARGET_SIMULATION_SECONDS}s)
${COLOR_CYAN}Quality:${COLOR_RESET} $error_count errors, $warning_count warnings
${COLOR_CYAN}Report:${COLOR_RESET} $summary_report

EOF
}

cleanup_resources() {
    local preserve_logs="$1"
    local preserve_intermediate_files="$2"
    
    log_message "INFO" "Starting comprehensive resource cleanup..."
    
    # Cleanup temporary files and directories
    if [[ -n "$PLUME_TEMP_DIR" && -d "$PLUME_TEMP_DIR" ]]; then
        if [[ "$preserve_intermediate_files" == "true" ]]; then
            log_message "INFO" "Preserving temporary directory: $PLUME_TEMP_DIR"
        else
            log_message "INFO" "Cleaning up temporary directory: $PLUME_TEMP_DIR"
            rm -rf "$PLUME_TEMP_DIR" 2>/dev/null || true
        fi
    fi
    
    # Cleanup process-specific temporary files
    local temp_files
    temp_files=$(find "${TMPDIR:-/tmp}" -name "plume_sim_$$_*" 2>/dev/null || true)
    if [[ -n "$temp_files" ]]; then
        log_message "INFO" "Cleaning up process-specific temporary files"
        echo "$temp_files" | xargs rm -f 2>/dev/null || true
    fi
    
    # Preserve or cleanup logs based on preference
    if [[ "$preserve_logs" == "true" ]]; then
        log_message "INFO" "Preserving execution logs in: $OUTPUT_DIR"
        
        # Compress logs for space efficiency
        if command -v gzip &> /dev/null; then
            find "$OUTPUT_DIR" -name "*.log" -not -name "*.gz" -exec gzip {} \; 2>/dev/null || true
            log_message "SUCCESS" "Log files compressed for preservation"
        fi
    else
        log_message "INFO" "Cleaning up execution logs"
        find "$OUTPUT_DIR" -name "*.log" -delete 2>/dev/null || true
    fi
    
    # Release system resources and memory
    if [[ "$PYTHON_PID" -gt 0 ]]; then
        # Ensure Python process is fully terminated
        kill -0 "$PYTHON_PID" 2>/dev/null && kill -KILL "$PYTHON_PID" 2>/dev/null || true
    fi
    
    # Cleanup environment variables and configuration
    unset PLUME_CONFIG_DIR PLUME_OUTPUT_DIR PLUME_LOG_LEVEL PLUME_BATCH_SIZE
    unset PLUME_TEMP_DIR PLUME_CPU_CORES PLUME_START_TIME
    unset PYTHONOPTIMIZE PYTHONHASHSEED
    
    # Reset resource limits
    if command -v ulimit &> /dev/null; then
        ulimit -v unlimited 2>/dev/null || true
    fi
    
    log_message "SUCCESS" "Resource cleanup completed successfully"
    
    # Generate final cleanup summary
    local cleanup_summary="$OUTPUT_DIR/cleanup_summary.txt"
    cat > "$cleanup_summary" << EOF
RESOURCE CLEANUP SUMMARY
========================
Timestamp: $(date)
Logs Preserved: $preserve_logs
Intermediate Files Preserved: $preserve_intermediate_files
Temporary Directory Cleaned: $(if [[ "$preserve_intermediate_files" != "true" ]]; then echo "Yes"; else echo "No (preserved)"; fi)
Environment Variables Reset: Yes
Resource Limits Reset: Yes
Process Cleanup: Completed

Cleanup Status: SUCCESS
EOF
    
    log_message "INFO" "Cleanup summary: $cleanup_summary"
}

# =============================================================================
# SIGNAL HANDLING SETUP
# =============================================================================

setup_signal_handlers() {
    # Set up signal handlers for graceful interruption handling
    trap 'handle_interruption SIGINT $PYTHON_PID' SIGINT
    trap 'handle_interruption SIGTERM $PYTHON_PID' SIGTERM
    trap 'handle_interruption SIGHUP $PYTHON_PID' SIGHUP
    
    log_message "INFO" "Signal handlers configured for graceful shutdown"
}

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

main() {
    local -a command_line_arguments=("$@")
    
    # Initialize script execution
    log_message "INFO" "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    log_message "INFO" "Command line: $0 $*"
    
    # Parse command-line arguments and options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit $EXIT_SUCCESS
                ;;
            -v|--version)
                show_version
                exit $EXIT_SUCCESS
                ;;
            -c|--config)
                CONFIG_DIR="$2"
                shift 2
                ;;
            -p|--python)
                PYTHON_EXEC="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -a|--algorithm)
                ALGORITHM_CONFIG="$2"
                shift 2
                ;;
            -j|--parallel)
                ENABLE_PARALLEL=true
                shift
                ;;
            -s|--serial)
                ENABLE_PARALLEL=false
                shift
                ;;
            -m|--monitor-interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --strict-validation)
                STRICT_VALIDATION=true
                shift
                ;;
            --skip-performance-check)
                SKIP_PERFORMANCE_CHECK=true
                shift
                ;;
            --no-colors)
                ENABLE_COLORS=false
                shift
                ;;
            --preserve-intermediate)
                PRESERVE_INTERMEDIATE=true
                shift
                ;;
            --no-preserve-logs)
                PRESERVE_LOGS=false
                shift
                ;;
            -*)
                log_message "ERROR" "Unknown option: $1"
                show_usage
                exit $EXIT_CONFIG_ERROR
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Validate remaining arguments
    if [[ $# -lt 1 ]]; then
        log_message "ERROR" "Missing required argument: input_data_path"
        show_usage
        exit $EXIT_CONFIG_ERROR
    fi
    
    local input_data_path="$1"
    local output_directory="${2:-$OUTPUT_DIR}"
    OUTPUT_DIR="$output_directory"
    
    # Setup signal handlers for graceful interruption handling
    setup_signal_handlers
    
    # Validate prerequisites and system readiness
    log_message "INFO" "Validating prerequisites and system readiness..."
    if ! check_prerequisites "$STRICT_VALIDATION" "$SKIP_PERFORMANCE_CHECK"; then
        local exit_code=$?
        log_message "ERROR" "Prerequisite validation failed (exit code: $exit_code)"
        exit $exit_code
    fi
    
    # Setup execution environment and configuration
    log_message "INFO" "Setting up execution environment..."
    if ! setup_environment "$CONFIG_DIR" "$OUTPUT_DIR" "$LOG_LEVEL"; then
        local exit_code=$?
        log_message "ERROR" "Environment setup failed (exit code: $exit_code)"
        exit $exit_code
    fi
    
    # Validate input arguments and configuration consistency
    log_message "INFO" "Validating arguments and configuration..."
    if ! validate_arguments "$input_data_path" "$output_directory"; then
        local exit_code=$?
        log_message "ERROR" "Argument validation failed (exit code: $exit_code)"
        cleanup_resources "$PRESERVE_LOGS" "$PRESERVE_INTERMEDIATE"
        exit $exit_code
    fi
    
    # Record execution start time
    EXECUTION_START_TIME=$(date +%s)
    
    # Execute batch simulation with monitoring and error handling
    log_message "INFO" "Executing batch simulation with monitoring..."
    local simulation_exit_code=0
    if ! execute_batch_simulation "$input_data_path" "$OUTPUT_DIR" "${ALGORITHM_CONFIG:-}" "$BATCH_SIZE" "$ENABLE_PARALLEL" ""; then
        simulation_exit_code=$?
        log_message "ERROR" "Batch simulation execution failed (exit code: $simulation_exit_code)"
    fi
    
    # Calculate execution time
    local execution_end_time
    execution_end_time=$(date +%s)
    local total_execution_time=$((execution_end_time - EXECUTION_START_TIME))
    local execution_hours
    execution_hours=$(echo "scale=2; $total_execution_time / 3600" | bc -l 2>/dev/null || echo "0")
    
    # Validate simulation results and performance metrics
    if [[ "$simulation_exit_code" -eq 0 ]]; then
        log_message "INFO" "Validating simulation results..."
        if ! validate_results "$OUTPUT_DIR" "$BATCH_SIZE" "$STRICT_VALIDATION"; then
            local validation_exit_code=$?
            log_message "WARNING" "Result validation reported issues (exit code: $validation_exit_code)"
            if [[ "$STRICT_VALIDATION" == "true" ]]; then
                simulation_exit_code=$validation_exit_code
            fi
        fi
    fi
    
    # Generate comprehensive summary report
    log_message "INFO" "Generating comprehensive summary report..."
    generate_summary_report "$OUTPUT_DIR" "$execution_hours" "$BATCH_SIZE" true
    
    # Cleanup resources and finalize execution
    log_message "INFO" "Cleaning up resources and finalizing execution..."
    cleanup_resources "$PRESERVE_LOGS" "$PRESERVE_INTERMEDIATE"
    
    # Final execution summary
    if [[ "$simulation_exit_code" -eq 0 ]]; then
        log_message "SUCCESS" "Batch simulation completed successfully"
        log_message "SUCCESS" "Execution time: ${execution_hours}h"
        log_message "SUCCESS" "Results available in: $OUTPUT_DIR"
    else
        log_message "ERROR" "Batch simulation completed with issues (exit code: $simulation_exit_code)"
        log_message "INFO" "Execution time: ${execution_hours}h"
        log_message "INFO" "Partial results and logs available in: $OUTPUT_DIR"
    fi
    
    # Return appropriate exit code based on execution outcome
    exit $simulation_exit_code
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Execute main function with all command-line arguments
main "$@"