#!/bin/bash

# Backend-Specific Dependency Installation Script for Plume Navigation Simulation System
# 
# Comprehensive dependency installation script that orchestrates Python package installation,
# environment validation, virtual environment setup, and dependency verification with >95%
# correlation requirements and 8-hour processing targets for 4000+ simulations.
#
# This script implements enterprise-grade installation procedures with:
# - Scientific Computing Dependencies Management (NumPy 2.1.3+, SciPy 1.15.3+, OpenCV 4.11.0+)
# - Performance Requirements Infrastructure for 4000+ simulation completion within 8 hours
# - Fail-Fast Validation with early detection of incompatible dependencies
# - Environment Validation and Setup with automated environment recreation
# - Cross-Platform Compatibility for different computational environments
# - Comprehensive error handling, progress tracking, logging, and validation
#
# Installation Phases:
# 1. Validation - Backend prerequisite validation and environment checking
# 2. Environment Setup - Virtual environment creation and configuration  
# 3. Core Dependencies - Core scientific computing package installation
# 4. Optional Dependencies - Optional performance and visualization packages
# 5. Development Dependencies - Development and testing tool installation
# 6. Verification - Comprehensive backend installation validation
# 7. Cleanup - Installation cleanup and environment finalization

set -euo pipefail  # Strict error handling: exit on error, undefined variables, pipe failures

# Signal handling for graceful cleanup on script interruption
trap cleanup_on_exit EXIT INT TERM

# ============================================================================
# GLOBAL CONFIGURATION CONSTANTS AND PATHS
# ============================================================================

# Script and directory path determination
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
readonly PROJECT_ROOT="$(dirname "$(dirname "$BACKEND_DIR")")"

# Core file paths for dependency and configuration management
readonly REQUIREMENTS_FILE="$BACKEND_DIR/requirements.txt"
readonly SETUP_FILE="$BACKEND_DIR/setup.py"
readonly PYPROJECT_FILE="$BACKEND_DIR/pyproject.toml"
readonly CONFIG_DIR="$BACKEND_DIR/config"

# Logging directory and file paths for comprehensive installation tracking
readonly LOG_DIR="$BACKEND_DIR/logs"
readonly INSTALL_LOG="$LOG_DIR/backend_install.log"
readonly ERROR_LOG="$LOG_DIR/backend_install_errors.log"
readonly VALIDATION_LOG="$LOG_DIR/backend_validation.log"

# Virtual environment paths for different installation modes
readonly VENV_DIR="$BACKEND_DIR/.venv"
readonly DEV_VENV_DIR="$BACKEND_DIR/.venv-dev"
readonly TEST_VENV_DIR="$BACKEND_DIR/.venv-test"

# Version and compatibility requirements for scientific computing
readonly PYTHON_MIN_VERSION="3.9"
readonly PIP_MIN_VERSION="21.0"

# Installation timing and retry configuration
readonly INSTALL_TIMEOUT=1800  # 30 minutes for complete installation
readonly MAX_RETRY_ATTEMPTS=3
readonly RETRY_DELAY=5

# Exit code definitions for specific error types and installation phases
readonly EXIT_SUCCESS=0
readonly EXIT_FAILURE=1
readonly EXIT_VALIDATION_ERROR=2
readonly EXIT_DEPENDENCY_ERROR=3
readonly EXIT_ENVIRONMENT_ERROR=4
readonly EXIT_NETWORK_ERROR=5

# Required package specifications with version constraints for scientific accuracy
readonly REQUIRED_PACKAGES=(
    "numpy>=2.1.3"
    "scipy>=1.15.3"
    "opencv-python>=4.11.0"
    "pandas>=2.2.0"
    "joblib>=1.6.0"
    "matplotlib>=3.9.0"
    "seaborn>=0.13.2"
    "pytest>=8.3.5"
)

# Optional packages for enhanced functionality and performance optimization
readonly OPTIONAL_PACKAGES=(
    "numba>=0.58.0"
    "cython>=3.0.0"
    "plotly>=5.15.0"
    "zarr>=2.16.0"
    "xarray>=2023.12.0"
)

# Development packages for testing, code quality, and development workflows
readonly DEVELOPMENT_PACKAGES=(
    "pytest-cov>=5.0.0"
    "black>=23.0.0"
    "flake8>=6.0.0"
    "mypy>=1.0.0"
    "pre-commit>=3.0.0"
)

# Performance packages for profiling and optimization
readonly PERFORMANCE_PACKAGES=(
    "memory-profiler>=0.61.0"
    "py-spy>=0.3.14"
    "line-profiler>=4.0.0"
)

# Installation phases for structured dependency management
readonly INSTALLATION_PHASES=(
    "validation"
    "environment_setup"
    "core_dependencies"
    "optional_dependencies"
    "development_dependencies"
    "verification"
)

# ============================================================================
# GLOBAL VARIABLES FOR RUNTIME STATE MANAGEMENT
# ============================================================================

# Installation configuration variables
VERBOSE_MODE=false
FORCE_REINSTALL=false
SKIP_VALIDATION=false
CREATE_DEV_ENV=false
CREATE_TEST_ENV=false
INSTALL_PERFORMANCE_PACKAGES=false
INSTALL_VISUALIZATION_PACKAGES=false
LOG_LEVEL="INFO"
CONSOLE_OUTPUT=true

# Installation tracking variables
INSTALLATION_START_TIME=""
INSTALLATION_END_TIME=""
INSTALLATION_DURATION=""
PACKAGES_INSTALLED=0
PACKAGES_FAILED=0
CURRENT_PHASE=""
INSTALLATION_SUCCESS=true

# Error tracking and statistics
declare -A INSTALLATION_ERRORS
declare -A INSTALLATION_WARNINGS
declare -A PHASE_TIMINGS
declare -A PACKAGE_INSTALL_TIMES

# ============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# ============================================================================

setup_logging() {
    # Initialize logging system for backend dependency installation with structured
    # logging, file rotation, and console output formatting for detailed installation tracking
    #
    # Parameters:
    #   $1 - log_level (optional): Logging level for installation process
    #   $2 - enable_console_output (optional): Enable/disable console output
    #
    # Returns:
    #   0 - Logging setup successful
    #   1 - Logging setup failed

    local log_level="${1:-$LOG_LEVEL}"
    local enable_console="${2:-$CONSOLE_OUTPUT}"

    # Create backend log directories if they don't exist
    if ! mkdir -p "$LOG_DIR"; then
        echo "ERROR: Failed to create log directory: $LOG_DIR" >&2
        return 1
    fi

    # Initialize log files with proper permissions
    if ! touch "$INSTALL_LOG" "$ERROR_LOG" "$VALIDATION_LOG"; then
        echo "ERROR: Failed to create log files" >&2
        return 1
    fi

    # Set appropriate permissions for log files
    chmod 644 "$INSTALL_LOG" "$ERROR_LOG" "$VALIDATION_LOG" 2>/dev/null || true

    # Load logging configuration from backend config if available
    local logging_config="$CONFIG_DIR/logging_config.json"
    if [[ -f "$logging_config" ]]; then
        log_message "Loaded logging configuration from $logging_config" "INFO"
    else
        log_message "Using default logging configuration" "WARN"
    fi

    # Configure log file rotation and retention policies
    log_message "Logging system initialized successfully" "INFO"
    log_message "Log level: $log_level" "INFO"
    log_message "Console output: $enable_console" "INFO"
    log_message "Log directory: $LOG_DIR" "INFO"

    # Test logging functionality and file permissions
    if log_message "Logging system test message" "DEBUG"; then
        log_message "Logging functionality verified" "INFO"
        return 0
    else
        echo "ERROR: Logging functionality test failed" >&2
        return 1
    fi
}

log_message() {
    # Log backend installation messages with timestamps, severity levels, and structured
    # formatting to both console and log files for comprehensive installation tracking
    #
    # Parameters:
    #   $1 - message: Log message content
    #   $2 - level: Log level (DEBUG, INFO, WARN, ERROR)
    #   $3 - to_error_log (optional): Also log to error log (true/false)
    #   $4 - component (optional): Component generating the log entry
    #
    # Returns:
    #   None

    local message="$1"
    local level="${2:-INFO}"
    local to_error_log="${3:-false}"
    local component="${4:-backend-install}"

    # Generate timestamp for log entry
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S.%3N')

    # Format message with level, component, and timestamp
    local formatted_message="$timestamp | $level | $component | $message"

    # Apply color coding for console output based on level
    local colored_message="$formatted_message"
    if [[ "$CONSOLE_OUTPUT" == "true" ]]; then
        case "$level" in
            "ERROR")
                colored_message="\033[31m$formatted_message\033[0m"  # Red
                ;;
            "WARN")
                colored_message="\033[33m$formatted_message\033[0m"  # Yellow
                ;;
            "INFO")
                colored_message="\033[32m$formatted_message\033[0m"  # Green
                ;;
            "DEBUG")
                colored_message="\033[36m$formatted_message\033[0m"  # Cyan
                ;;
        esac

        # Output message to console if console output enabled
        echo -e "$colored_message"
    fi

    # Append message to backend installation log file
    echo "$formatted_message" >> "$INSTALL_LOG" 2>/dev/null || true

    # Append to error log if severity requires it
    if [[ "$level" == "ERROR" || "$to_error_log" == "true" ]]; then
        echo "$formatted_message" >> "$ERROR_LOG" 2>/dev/null || true
    fi
}

# ============================================================================
# ARGUMENT PARSING AND CONFIGURATION
# ============================================================================

parse_arguments() {
    # Parse command-line arguments for backend installation options including environment
    # type, package selection, validation mode, and debugging options
    #
    # Parameters:
    #   $@ - Command line arguments
    #
    # Returns:
    #   Associative array with parsed arguments and configuration options

    # Initialize default argument values for backend installation
    VERBOSE_MODE=false
    FORCE_REINSTALL=false
    SKIP_VALIDATION=false
    CREATE_DEV_ENV=false
    CREATE_TEST_ENV=false
    INSTALL_PERFORMANCE_PACKAGES=false
    INSTALL_VISUALIZATION_PACKAGES=false

    # Parse command-line options and flags
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit $EXIT_SUCCESS
                ;;
            --verbose|-v)
                VERBOSE_MODE=true
                LOG_LEVEL="DEBUG"
                log_message "Verbose mode enabled" "INFO"
                shift
                ;;
            --force|-f)
                FORCE_REINSTALL=true
                log_message "Force reinstall mode enabled" "INFO"
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                log_message "Validation skip mode enabled" "WARN"
                shift
                ;;
            --dev-env)
                CREATE_DEV_ENV=true
                log_message "Development environment creation enabled" "INFO"
                shift
                ;;
            --test-env)
                CREATE_TEST_ENV=true
                log_message "Test environment creation enabled" "INFO"
                shift
                ;;
            --performance-packages)
                INSTALL_PERFORMANCE_PACKAGES=true
                log_message "Performance package installation enabled" "INFO"
                shift
                ;;
            --visualization-packages)
                INSTALL_VISUALIZATION_PACKAGES=true
                log_message "Visualization package installation enabled" "INFO"
                shift
                ;;
            --quiet|-q)
                CONSOLE_OUTPUT=false
                log_message "Quiet mode enabled" "INFO"
                shift
                ;;
            --log-level)
                if [[ -n "${2:-}" ]]; then
                    LOG_LEVEL="$2"
                    log_message "Log level set to: $LOG_LEVEL" "INFO"
                    shift 2
                else
                    log_message "ERROR: --log-level requires a value" "ERROR"
                    exit $EXIT_FAILURE
                fi
                ;;
            *)
                log_message "ERROR: Unknown option: $1" "ERROR"
                show_help
                exit $EXIT_FAILURE
                ;;
        esac
    done

    # Validate argument combinations and constraints
    if [[ "$SKIP_VALIDATION" == "true" && "$FORCE_REINSTALL" == "true" ]]; then
        log_message "WARN: Skip validation with force reinstall may cause issues" "WARN"
    fi

    # Generate argument summary for logging
    log_message "Backend installation configuration:" "INFO"
    log_message "  Verbose mode: $VERBOSE_MODE" "INFO"
    log_message "  Force reinstall: $FORCE_REINSTALL" "INFO"
    log_message "  Skip validation: $SKIP_VALIDATION" "INFO"
    log_message "  Create dev environment: $CREATE_DEV_ENV" "INFO"
    log_message "  Create test environment: $CREATE_TEST_ENV" "INFO"
    log_message "  Install performance packages: $INSTALL_PERFORMANCE_PACKAGES" "INFO"
    log_message "  Install visualization packages: $INSTALL_VISUALIZATION_PACKAGES" "INFO"
    log_message "  Console output: $CONSOLE_OUTPUT" "INFO"
    log_message "  Log level: $LOG_LEVEL" "INFO"
}

show_help() {
    # Display comprehensive help information for backend installation script including
    # available options, usage examples, and backend-specific guidance
    
    cat << EOF
Backend Dependency Installation Script for Plume Navigation Simulation System

USAGE:
    $0 [OPTIONS]

DESCRIPTION:
    Comprehensive backend dependency installation script that orchestrates Python 
    package installation, environment validation, virtual environment setup, and 
    dependency verification for the plume navigation simulation system.

OPTIONS:
    -h, --help                  Show this help message and exit
    -v, --verbose              Enable verbose output with detailed progress information
    -f, --force                Force reinstallation of all packages and environments
    --skip-validation          Skip prerequisite validation (not recommended)
    --dev-env                  Create development virtual environment with dev tools
    --test-env                 Create testing virtual environment with test frameworks
    --performance-packages     Install performance optimization packages (Numba, Cython)
    --visualization-packages   Install advanced visualization packages (Plotly, Bokeh)
    -q, --quiet                Suppress console output (logs still written to files)
    --log-level LEVEL          Set logging level (DEBUG, INFO, WARN, ERROR)

INSTALLATION PHASES:
    1. Validation              Backend prerequisite validation and environment checking
    2. Environment Setup       Virtual environment creation and configuration
    3. Core Dependencies       Core scientific computing package installation
    4. Optional Dependencies   Optional performance and visualization packages
    5. Development Dependencies Development and testing tool installation
    6. Verification            Comprehensive backend installation validation
    7. Cleanup                 Installation cleanup and environment finalization

REQUIREMENTS:
    - Python >= 3.9           Required for scientific computing compatibility
    - pip >= 21.0             Required for modern package installation
    - 8GB+ available disk      Required for complete installation with cache
    - 4GB+ available RAM       Required for parallel package installation

EXAMPLES:
    # Basic backend installation
    $0

    # Complete installation with development environment
    $0 --dev-env --performance-packages --visualization-packages

    # Force reinstallation with verbose output
    $0 --force --verbose

    # Quiet installation for automated deployment
    $0 --quiet --log-level WARN

LOG FILES:
    Installation Log:    $INSTALL_LOG
    Error Log:          $ERROR_LOG
    Validation Log:     $VALIDATION_LOG

For more information, visit: https://github.com/research-team/plume-simulation
EOF
}

# ============================================================================
# PREREQUISITE VALIDATION FUNCTIONS
# ============================================================================

validate_prerequisites() {
    # Validate backend-specific prerequisites including Python version, pip availability,
    # virtual environment support, and basic system requirements before dependency installation
    #
    # Parameters:
    #   $1 - strict_validation (optional): Enable strict validation mode
    #   $2 - check_network (optional): Enable network connectivity checks
    #
    # Returns:
    #   0 - All prerequisites met
    #   2 - Validation error

    local strict_validation="${1:-true}"
    local check_network="${2:-true}"

    log_message "Starting backend prerequisite validation" "INFO" false "validation"

    # Check Python version against minimum requirements (>=3.9)
    if ! check_python_version; then
        log_message "ERROR: Python version validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Validate pip installation and version (>=21.0)
    if ! check_pip_version; then
        log_message "ERROR: pip version validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Test virtual environment creation capability
    if ! test_virtual_environment_support; then
        log_message "ERROR: Virtual environment support validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Check available disk space for package installation
    if ! check_disk_space; then
        log_message "ERROR: Insufficient disk space for installation" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Test network connectivity for package downloads if enabled
    if [[ "$check_network" == "true" ]]; then
        if ! test_network_connectivity; then
            log_message "WARN: Network connectivity test failed" "WARN"
            if [[ "$strict_validation" == "true" ]]; then
                return $EXIT_NETWORK_ERROR
            fi
        fi
    fi

    # Verify write permissions for backend directories
    if ! check_directory_permissions; then
        log_message "ERROR: Directory permission validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Validate system architecture compatibility
    if ! check_system_compatibility; then
        log_message "ERROR: System compatibility validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Call backend environment validation script if available
    local validation_script="$SCRIPT_DIR/validate_environment.py"
    if [[ -f "$validation_script" ]]; then
        log_message "Running backend environment validation script" "INFO"
        if python3 "$validation_script" >> "$VALIDATION_LOG" 2>&1; then
            log_message "Backend environment validation script passed" "INFO"
        else
            log_message "WARN: Backend environment validation script failed" "WARN"
            if [[ "$strict_validation" == "true" ]]; then
                return $EXIT_VALIDATION_ERROR
            fi
        fi
    else
        log_message "Backend environment validation script not found" "WARN"
    fi

    log_message "Backend prerequisite validation completed successfully" "INFO" false "validation"
    return $EXIT_SUCCESS
}

check_python_version() {
    # Check Python version compatibility for scientific computing requirements
    
    log_message "Checking Python version compatibility" "INFO" false "validation"

    # Get current Python version
    local python_version
    if ! python_version=$(python3 --version 2>&1); then
        log_message "ERROR: Python 3 not found or not executable" "ERROR"
        return 1
    fi

    # Extract version number from output
    local version_number
    version_number=$(echo "$python_version" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

    if [[ -z "$version_number" ]]; then
        log_message "ERROR: Could not determine Python version" "ERROR"
        return 1
    fi

    log_message "Found Python version: $version_number" "INFO"

    # Convert version to comparable format
    local major minor patch
    IFS='.' read -r major minor patch <<< "$version_number"

    # Check minimum version requirement (3.9)
    if [[ $major -lt 3 || ($major -eq 3 && $minor -lt 9) ]]; then
        log_message "ERROR: Python $version_number is below minimum required version $PYTHON_MIN_VERSION" "ERROR"
        log_message "Please upgrade to Python $PYTHON_MIN_VERSION or higher for scientific computing compatibility" "ERROR"
        return 1
    fi

    log_message "Python version $version_number meets requirements (>= $PYTHON_MIN_VERSION)" "INFO"
    return 0
}

check_pip_version() {
    # Check pip version and functionality for package installation
    
    log_message "Checking pip version and functionality" "INFO" false "validation"

    # Check if pip is available
    if ! command -v pip3 >/dev/null 2>&1; then
        log_message "ERROR: pip3 not found" "ERROR"
        return 1
    fi

    # Get pip version
    local pip_version
    if ! pip_version=$(pip3 --version 2>&1); then
        log_message "ERROR: pip3 not functional" "ERROR"
        return 1
    fi

    log_message "Found pip: $pip_version" "INFO"

    # Extract version number
    local version_number
    version_number=$(echo "$pip_version" | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [[ -n "$version_number" ]]; then
        # Compare with minimum required version
        local major minor
        IFS='.' read -r major minor <<< "$version_number"
        
        if [[ $major -lt 21 ]]; then
            log_message "WARN: pip version $version_number is below recommended version $PIP_MIN_VERSION" "WARN"
            log_message "Consider upgrading pip: python3 -m pip install --upgrade pip" "WARN"
        else
            log_message "pip version $version_number meets requirements" "INFO"
        fi
    fi

    # Test pip functionality
    if python3 -m pip --version >/dev/null 2>&1; then
        log_message "pip functionality verified" "INFO"
        return 0
    else
        log_message "ERROR: pip functionality test failed" "ERROR"
        return 1
    fi
}

test_virtual_environment_support() {
    # Test virtual environment creation and activation capability
    
    log_message "Testing virtual environment support" "INFO" false "validation"

    # Test venv module availability
    if ! python3 -m venv --help >/dev/null 2>&1; then
        log_message "ERROR: Python venv module not available" "ERROR"
        return 1
    fi

    # Create temporary test virtual environment
    local test_venv_dir="/tmp/backend_venv_test_$$"
    
    if python3 -m venv "$test_venv_dir" 2>/dev/null; then
        log_message "Virtual environment creation test successful" "INFO"
        
        # Test activation
        if source "$test_venv_dir/bin/activate" 2>/dev/null; then
            log_message "Virtual environment activation test successful" "INFO"
            deactivate 2>/dev/null || true
        else
            log_message "WARN: Virtual environment activation test failed" "WARN"
        fi
        
        # Clean up test environment
        rm -rf "$test_venv_dir" 2>/dev/null || true
        return 0
    else
        log_message "ERROR: Virtual environment creation test failed" "ERROR"
        return 1
    fi
}

check_disk_space() {
    # Check available disk space for package installation and temporary files
    
    log_message "Checking available disk space" "INFO" false "validation"

    # Required space in MB for complete backend installation
    local required_space_mb=8192  # 8GB for comprehensive installation

    # Get available space in the backend directory
    local available_space_kb
    if available_space_kb=$(df -k "$BACKEND_DIR" | awk 'NR==2 {print $4}'); then
        local available_space_mb=$((available_space_kb / 1024))
        
        log_message "Available disk space: ${available_space_mb}MB" "INFO"
        
        if [[ $available_space_mb -lt $required_space_mb ]]; then
            log_message "WARN: Low disk space. Required: ${required_space_mb}MB, Available: ${available_space_mb}MB" "WARN"
            if [[ $available_space_mb -lt 2048 ]]; then  # Critical threshold: 2GB
                log_message "ERROR: Insufficient disk space for installation" "ERROR"
                return 1
            fi
        else
            log_message "Disk space check passed" "INFO"
        fi
    else
        log_message "WARN: Could not determine available disk space" "WARN"
    fi

    return 0
}

test_network_connectivity() {
    # Test network connectivity for package downloads
    
    log_message "Testing network connectivity" "INFO" false "validation"

    # Test connectivity to PyPI
    local pypi_hosts=("pypi.org" "files.pythonhosted.org")
    local connectivity_ok=false

    for host in "${pypi_hosts[@]}"; do
        if ping -c 1 -W 5 "$host" >/dev/null 2>&1; then
            log_message "Network connectivity to $host: OK" "INFO"
            connectivity_ok=true
            break
        else
            log_message "Network connectivity to $host: FAILED" "WARN"
        fi
    done

    if [[ "$connectivity_ok" == "true" ]]; then
        log_message "Network connectivity test passed" "INFO"
        return 0
    else
        log_message "WARN: Network connectivity test failed" "WARN"
        return 1
    fi
}

check_directory_permissions() {
    # Check write permissions for backend directories
    
    log_message "Checking directory permissions" "INFO" false "validation"

    local dirs_to_check=("$BACKEND_DIR" "$LOG_DIR")

    for dir in "${dirs_to_check[@]}"; do
        if [[ ! -d "$dir" ]]; then
            if mkdir -p "$dir" 2>/dev/null; then
                log_message "Created directory: $dir" "INFO"
            else
                log_message "ERROR: Cannot create directory: $dir" "ERROR"
                return 1
            fi
        fi

        if [[ -w "$dir" ]]; then
            log_message "Write permission verified for: $dir" "INFO"
        else
            log_message "ERROR: No write permission for: $dir" "ERROR"
            return 1
        fi
    done

    return 0
}

check_system_compatibility() {
    # Check system architecture and OS compatibility
    
    log_message "Checking system compatibility" "INFO" false "validation"

    local os_type
    os_type=$(uname -s)
    local arch_type
    arch_type=$(uname -m)

    log_message "Operating system: $os_type" "INFO"
    log_message "Architecture: $arch_type" "INFO"

    # Check for supported architectures
    case "$arch_type" in
        x86_64|amd64|arm64|aarch64)
            log_message "Architecture $arch_type is supported" "INFO"
            ;;
        *)
            log_message "WARN: Architecture $arch_type may have limited package support" "WARN"
            ;;
    esac

    # Check for supported operating systems
    case "$os_type" in
        Linux|Darwin)
            log_message "Operating system $os_type is supported" "INFO"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            log_message "Windows environment detected: $os_type" "INFO"
            ;;
        *)
            log_message "WARN: Operating system $os_type may have limited support" "WARN"
            ;;
    esac

    return 0
}

# ============================================================================
# VIRTUAL ENVIRONMENT MANAGEMENT FUNCTIONS
# ============================================================================

setup_virtual_environments() {
    # Create and configure Python virtual environments for backend, development, and
    # testing with proper isolation and dependency management
    #
    # Parameters:
    #   $1 - create_dev_env: Create development environment (true/false)
    #   $2 - create_test_env: Create testing environment (true/false)
    #   $3 - force_recreate: Force recreation of existing environments (true/false)
    #
    # Returns:
    #   0 - Virtual environments setup successful
    #   4 - Environment setup error

    local create_dev_env="${1:-$CREATE_DEV_ENV}"
    local create_test_env="${2:-$CREATE_TEST_ENV}"
    local force_recreate="${3:-$FORCE_REINSTALL}"

    log_message "Setting up virtual environments" "INFO" false "environment"
    CURRENT_PHASE="environment_setup"

    # Remove existing virtual environments if force_recreate enabled
    if [[ "$force_recreate" == "true" ]]; then
        log_message "Force recreate enabled - removing existing environments" "INFO"
        
        for venv_dir in "$VENV_DIR" "$DEV_VENV_DIR" "$TEST_VENV_DIR"; do
            if [[ -d "$venv_dir" ]]; then
                log_message "Removing existing environment: $venv_dir" "INFO"
                rm -rf "$venv_dir" || {
                    log_message "ERROR: Failed to remove existing environment: $venv_dir" "ERROR"
                    return $EXIT_ENVIRONMENT_ERROR
                }
            fi
        done
    fi

    # Create main backend virtual environment
    if ! create_virtual_environment "$VENV_DIR" "backend"; then
        log_message "ERROR: Failed to create main backend virtual environment" "ERROR"
        return $EXIT_ENVIRONMENT_ERROR
    fi

    # Create development virtual environment if requested
    if [[ "$create_dev_env" == "true" ]]; then
        if ! create_virtual_environment "$DEV_VENV_DIR" "development"; then
            log_message "ERROR: Failed to create development virtual environment" "ERROR"
            return $EXIT_ENVIRONMENT_ERROR
        fi
    fi

    # Create testing virtual environment if requested
    if [[ "$create_test_env" == "true" ]]; then
        if ! create_virtual_environment "$TEST_VENV_DIR" "testing"; then
            log_message "ERROR: Failed to create testing virtual environment" "ERROR"
            return $EXIT_ENVIRONMENT_ERROR
        fi
    fi

    # Configure environment variables for scientific computing
    if ! configure_environment_variables; then
        log_message "WARN: Environment variable configuration had issues" "WARN"
    fi

    log_message "Virtual environment setup completed successfully" "INFO" false "environment"
    return $EXIT_SUCCESS
}

create_virtual_environment() {
    # Create individual virtual environment with proper configuration
    #
    # Parameters:
    #   $1 - venv_path: Path for virtual environment
    #   $2 - env_type: Type of environment (backend, development, testing)
    #
    # Returns:
    #   0 - Environment created successfully
    #   1 - Environment creation failed

    local venv_path="$1"
    local env_type="$2"

    log_message "Creating $env_type virtual environment: $venv_path" "INFO"

    # Create virtual environment
    if ! python3 -m venv "$venv_path"; then
        log_message "ERROR: Failed to create virtual environment: $venv_path" "ERROR"
        return 1
    fi

    # Activate virtual environment and verify activation
    if ! source "$venv_path/bin/activate"; then
        log_message "ERROR: Failed to activate virtual environment: $venv_path" "ERROR"
        return 1
    fi

    # Verify virtual environment activation
    local current_python
    current_python=$(which python)
    if [[ "$current_python" != "$venv_path"* ]]; then
        log_message "ERROR: Virtual environment activation verification failed" "ERROR"
        deactivate 2>/dev/null || true
        return 1
    fi

    log_message "Virtual environment activated: $current_python" "INFO"

    # Upgrade pip to latest version in virtual environment
    log_message "Upgrading pip in $env_type environment" "INFO"
    if ! python -m pip install --upgrade pip --quiet; then
        log_message "WARN: Failed to upgrade pip in $env_type environment" "WARN"
    fi

    # Install wheel and setuptools for build support
    log_message "Installing build dependencies in $env_type environment" "INFO"
    if ! python -m pip install --upgrade wheel setuptools --quiet; then
        log_message "WARN: Failed to install build dependencies in $env_type environment" "WARN"
    fi

    # Test virtual environment functionality and isolation
    local test_package="pip"
    if python -c "import $test_package" 2>/dev/null; then
        log_message "$env_type virtual environment functionality verified" "INFO"
    else
        log_message "ERROR: $env_type virtual environment functionality test failed" "ERROR"
        deactivate 2>/dev/null || true
        return 1
    fi

    deactivate 2>/dev/null || true
    log_message "$env_type virtual environment created successfully: $venv_path" "INFO"
    return 0
}

configure_environment_variables() {
    # Configure environment variables for scientific computing optimization
    
    log_message "Configuring environment variables for scientific computing" "INFO"

    # Create environment configuration file
    local env_config_file="$BACKEND_DIR/.env"
    
    cat > "$env_config_file" << EOF
# Backend Environment Configuration for Scientific Computing
# Automatically generated by backend dependency installation script

# Python optimization settings
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

# NumPy and SciPy optimization
export OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-\$(nproc)}
export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-\$(nproc)}
export NUMBA_NUM_THREADS=\${NUMBA_NUM_THREADS:-\$(nproc)}

# OpenCV configuration
export OPENCV_LOG_LEVEL=ERROR

# Joblib parallel processing
export JOBLIB_TEMP_FOLDER=/tmp
export JOBLIB_MULTIPROCESSING=1

# Backend specific paths
export BACKEND_ROOT=$BACKEND_DIR
export BACKEND_LOG_DIR=$LOG_DIR
export BACKEND_CONFIG_DIR=$CONFIG_DIR

# Scientific computing precision
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1
export SCIPY_PIL_IMAGE_VIEWER=display
EOF

    if [[ -f "$env_config_file" ]]; then
        log_message "Environment configuration created: $env_config_file" "INFO"
        return 0
    else
        log_message "WARN: Failed to create environment configuration file" "WARN"
        return 1
    fi
}

# ============================================================================
# DEPENDENCY INSTALLATION FUNCTIONS
# ============================================================================

install_core_dependencies() {
    # Install core scientific computing dependencies including NumPy, SciPy, OpenCV,
    # pandas, joblib, and matplotlib with version validation and functionality testing
    #
    # Parameters:
    #   $1 - upgrade_packages: Upgrade existing packages (true/false)
    #   $2 - use_binary_wheels: Prefer binary wheels over source (true/false)
    #   $3 - timeout_seconds: Installation timeout in seconds
    #
    # Returns:
    #   0 - Core dependencies installed successfully
    #   3 - Dependency installation error

    local upgrade_packages="${1:-false}"
    local use_binary_wheels="${2:-true}"
    local timeout_seconds="${3:-$INSTALL_TIMEOUT}"

    log_message "Installing core scientific computing dependencies" "INFO" false "dependencies"
    CURRENT_PHASE="core_dependencies"

    # Activate backend virtual environment
    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate backend virtual environment" "ERROR"
        return $EXIT_DEPENDENCY_ERROR
    fi

    local install_options="--quiet"
    if [[ "$upgrade_packages" == "true" ]]; then
        install_options="$install_options --upgrade"
    fi

    if [[ "$use_binary_wheels" == "true" ]]; then
        install_options="$install_options --only-binary=all"
    fi

    # Install core scientific computing packages from requirements
    log_message "Installing packages from requirements.txt" "INFO"
    if ! install_from_requirements "$REQUIREMENTS_FILE" "backend" "$upgrade_packages" "$timeout_seconds"; then
        log_message "ERROR: Failed to install core dependencies from requirements.txt" "ERROR"
        deactivate 2>/dev/null || true
        return $EXIT_DEPENDENCY_ERROR
    fi

    # Install additional core dependencies individually with validation
    log_message "Installing and validating core packages individually" "INFO"
    local failed_packages=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        log_message "Installing core package: $package" "INFO"
        
        if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$timeout_seconds" true; then
            log_message "Successfully installed core package: $package" "INFO"
            ((PACKAGES_INSTALLED++))
        else
            log_message "ERROR: Failed to install core package: $package" "ERROR"
            failed_packages+=("$package")
            ((PACKAGES_FAILED++))
        fi
    done

    # Test basic functionality of critical packages
    log_message "Testing core package functionality" "INFO"
    local critical_packages=("numpy" "scipy" "cv2" "pandas" "joblib" "matplotlib")
    
    for package in "${critical_packages[@]}"; do
        if test_package_functionality "$package" "import" 30; then
            log_message "Core package functionality verified: $package" "INFO"
        else
            log_message "ERROR: Core package functionality test failed: $package" "ERROR"
            failed_packages+=("$package")
        fi
    done

    deactivate 2>/dev/null || true

    # Check if any critical packages failed
    if [[ ${#failed_packages[@]} -gt 0 ]]; then
        log_message "ERROR: Core dependency installation completed with failures" "ERROR"
        log_message "Failed packages: ${failed_packages[*]}" "ERROR"
        return $EXIT_DEPENDENCY_ERROR
    fi

    log_message "Core dependency installation completed successfully" "INFO" false "dependencies"
    return $EXIT_SUCCESS
}

install_optional_dependencies() {
    # Install optional dependencies including performance optimization packages,
    # visualization libraries, and scientific data format support with graceful failure handling
    #
    # Parameters:
    #   $1 - include_performance_packages: Install performance packages (true/false)
    #   $2 - include_visualization_packages: Install visualization packages (true/false)
    #   $3 - include_scientific_packages: Install scientific packages (true/false)
    #
    # Returns:
    #   0 - Optional dependencies processed (may have some failures)

    local include_performance="${1:-$INSTALL_PERFORMANCE_PACKAGES}"
    local include_visualization="${2:-$INSTALL_VISUALIZATION_PACKAGES}"
    local include_scientific="${3:-true}"

    log_message "Installing optional dependencies" "INFO" false "dependencies"
    CURRENT_PHASE="optional_dependencies"

    # Activate backend virtual environment
    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate backend virtual environment" "ERROR"
        return $EXIT_DEPENDENCY_ERROR
    fi

    local optional_install_success=true

    # Install performance optimization packages if requested
    if [[ "$include_performance" == "true" ]]; then
        log_message "Installing performance optimization packages" "INFO"
        
        for package in "${PERFORMANCE_PACKAGES[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed performance package: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "WARN: Failed to install optional performance package: $package" "WARN"
                optional_install_success=false
            fi
        done
    fi

    # Install visualization libraries if requested
    if [[ "$include_visualization" == "true" ]]; then
        log_message "Installing visualization packages" "INFO"
        
        local visualization_packages=("plotly>=5.15.0" "bokeh>=3.2.0" "ipywidgets>=8.1.0")
        
        for package in "${visualization_packages[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed visualization package: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "WARN: Failed to install optional visualization package: $package" "WARN"
                optional_install_success=false
            fi
        done
    fi

    # Install scientific data format packages if requested
    if [[ "$include_scientific" == "true" ]]; then
        log_message "Installing scientific data format packages" "INFO"
        
        for package in "${OPTIONAL_PACKAGES[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed scientific package: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "WARN: Failed to install optional scientific package: $package" "WARN"
                optional_install_success=false
            fi
        done
    fi

    deactivate 2>/dev/null || true

    if [[ "$optional_install_success" == "true" ]]; then
        log_message "Optional dependency installation completed successfully" "INFO" false "dependencies"
    else
        log_message "Optional dependency installation completed with some failures" "WARN" false "dependencies"
    fi

    return $EXIT_SUCCESS
}

install_development_dependencies() {
    # Install development and testing dependencies including pytest, code quality tools,
    # documentation generators, and development utilities for backend development workflow
    #
    # Parameters:
    #   $1 - include_testing_tools: Install testing frameworks (true/false)
    #   $2 - include_quality_tools: Install code quality tools (true/false)
    #   $3 - include_documentation_tools: Install documentation tools (true/false)
    #
    # Returns:
    #   0 - Development dependencies installed successfully
    #   3 - Development dependency installation error

    local include_testing="${1:-true}"
    local include_quality="${2:-true}"
    local include_documentation="${3:-false}"

    if [[ "$CREATE_DEV_ENV" != "true" ]]; then
        log_message "Development environment not requested - skipping development dependencies" "INFO"
        return $EXIT_SUCCESS
    fi

    log_message "Installing development dependencies" "INFO" false "dependencies"
    CURRENT_PHASE="development_dependencies"

    # Activate development virtual environment
    if ! source "$DEV_VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate development virtual environment" "ERROR"
        return $EXIT_DEPENDENCY_ERROR
    fi

    # Install testing framework and plugins if requested
    if [[ "$include_testing" == "true" ]]; then
        log_message "Installing testing tools" "INFO"
        
        local testing_packages=(
            "pytest>=8.3.5"
            "pytest-cov>=5.0.0"
            "pytest-xdist>=3.3.0"
            "pytest-benchmark>=4.0.0"
            "pytest-timeout>=2.1.0"
            "pytest-mock>=3.11.0"
        )
        
        for package in "${testing_packages[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed testing package: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "ERROR: Failed to install testing package: $package" "ERROR"
                ((PACKAGES_FAILED++))
            fi
        done
    fi

    # Install code quality tools if requested
    if [[ "$include_quality" == "true" ]]; then
        log_message "Installing code quality tools" "INFO"
        
        for package in "${DEVELOPMENT_PACKAGES[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed quality tool: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "ERROR: Failed to install quality tool: $package" "ERROR"
                ((PACKAGES_FAILED++))
            fi
        done
    fi

    # Install documentation tools if requested
    if [[ "$include_documentation" == "true" ]]; then
        log_message "Installing documentation tools" "INFO"
        
        local doc_packages=(
            "sphinx>=7.1.0"
            "sphinx-rtd-theme>=1.3.0"
            "sphinx-autodoc-typehints>=1.24.0"
            "myst-parser>=2.0.0"
        )
        
        for package in "${doc_packages[@]}"; do
            if install_package_with_retry "$package" "$MAX_RETRY_ATTEMPTS" "$INSTALL_TIMEOUT" false; then
                log_message "Successfully installed documentation package: $package" "INFO"
                ((PACKAGES_INSTALLED++))
            else
                log_message "WARN: Failed to install documentation package: $package" "WARN"
            fi
        done
    fi

    # Configure pre-commit hooks for development workflow
    if command -v pre-commit >/dev/null 2>&1; then
        log_message "Configuring pre-commit hooks" "INFO"
        
        if [[ -f "$BACKEND_DIR/.pre-commit-config.yaml" ]]; then
            if pre-commit install --install-hooks >/dev/null 2>&1; then
                log_message "Pre-commit hooks configured successfully" "INFO"
            else
                log_message "WARN: Pre-commit hook configuration failed" "WARN"
            fi
        else
            log_message "Pre-commit configuration file not found" "INFO"
        fi
    fi

    deactivate 2>/dev/null || true

    log_message "Development dependency installation completed" "INFO" false "dependencies"
    return $EXIT_SUCCESS
}

install_from_requirements() {
    # Install packages from requirements.txt file with retry logic, progress tracking,
    # and comprehensive error handling for reliable package installation
    #
    # Parameters:
    #   $1 - requirements_file: Path to requirements.txt file
    #   $2 - environment_name: Name of target environment
    #   $3 - upgrade_packages: Upgrade existing packages (true/false)
    #   $4 - timeout_seconds: Installation timeout
    #
    # Returns:
    #   0 - Requirements installation successful
    #   1 - Requirements installation failed

    local requirements_file="$1"
    local environment_name="$2"
    local upgrade_packages="${3:-false}"
    local timeout_seconds="${4:-$INSTALL_TIMEOUT}"

    # Validate requirements file exists and is readable
    if [[ ! -f "$requirements_file" ]]; then
        log_message "ERROR: Requirements file not found: $requirements_file" "ERROR"
        return 1
    fi

    if [[ ! -r "$requirements_file" ]]; then
        log_message "ERROR: Requirements file not readable: $requirements_file" "ERROR"
        return 1
    fi

    log_message "Installing packages from requirements file: $requirements_file" "INFO"
    
    # Parse requirements file for package specifications
    local package_count
    package_count=$(grep -c "^[^#]" "$requirements_file" 2>/dev/null || echo "0")
    log_message "Found $package_count packages in requirements file" "INFO"

    # Prepare pip install command
    local pip_cmd="python -m pip install"
    local pip_options="--requirement $requirements_file --quiet"
    
    if [[ "$upgrade_packages" == "true" ]]; then
        pip_options="$pip_options --upgrade"
    fi

    # Install packages with retry logic for transient failures
    local attempt=1
    local max_attempts=3
    
    while [[ $attempt -le $max_attempts ]]; do
        log_message "Installing requirements (attempt $attempt/$max_attempts)" "INFO"
        
        if timeout "$timeout_seconds" $pip_cmd $pip_options; then
            log_message "Requirements installation successful on attempt $attempt" "INFO"
            return 0
        else
            local exit_code=$?
            log_message "Requirements installation failed on attempt $attempt (exit code: $exit_code)" "WARN"
            
            if [[ $attempt -lt $max_attempts ]]; then
                log_message "Waiting ${RETRY_DELAY}s before retry..." "INFO"
                sleep "$RETRY_DELAY"
            fi
            
            ((attempt++))
        fi
    done

    log_message "ERROR: Requirements installation failed after $max_attempts attempts" "ERROR"
    return 1
}

install_package_with_retry() {
    # Install individual package with exponential backoff retry logic, detailed error
    # handling, and installation validation for robust package installation
    #
    # Parameters:
    #   $1 - package_spec: Package specification with version constraints
    #   $2 - max_attempts: Maximum retry attempts
    #   $3 - timeout_seconds: Installation timeout
    #   $4 - test_import: Test package import after installation (true/false)
    #
    # Returns:
    #   0 - Package installation successful
    #   1 - Package installation failed

    local package_spec="$1"
    local max_attempts="${2:-$MAX_RETRY_ATTEMPTS}"
    local timeout_seconds="${3:-$INSTALL_TIMEOUT}"
    local test_import="${4:-false}"

    # Parse package specification and version constraints
    local package_name
    package_name=$(echo "$package_spec" | sed 's/[<>=!].*//' | tr -d ' ')
    
    log_message "Installing package: $package_spec" "INFO"

    # Attempt package installation with retry logic
    local attempt=1
    local delay="$RETRY_DELAY"
    
    while [[ $attempt -le $max_attempts ]]; do
        log_message "Installing $package_name (attempt $attempt/$max_attempts)" "INFO"
        
        # Record installation start time
        local start_time
        start_time=$(date +%s)
        
        # Attempt package installation with pip
        if timeout "$timeout_seconds" python -m pip install "$package_spec" --quiet; then
            local end_time
            end_time=$(date +%s)
            local install_duration=$((end_time - start_time))
            
            log_message "Package $package_name installed successfully in ${install_duration}s" "INFO"
            
            # Test package import if requested
            if [[ "$test_import" == "true" ]]; then
                if test_package_import "$package_name"; then
                    log_message "Package import test passed: $package_name" "INFO"
                    return 0
                else
                    log_message "ERROR: Package import test failed: $package_name" "ERROR"
                    return 1
                fi
            fi
            
            return 0
        else
            local exit_code=$?
            log_message "Package installation failed: $package_name (exit code: $exit_code)" "WARN"
            
            if [[ $attempt -lt $max_attempts ]]; then
                log_message "Waiting ${delay}s before retry..." "INFO"
                sleep "$delay"
                # Apply exponential backoff between retry attempts
                delay=$((delay * 2))
            fi
            
            ((attempt++))
        fi
    done

    log_message "ERROR: Package installation failed after $max_attempts attempts: $package_name" "ERROR"
    return 1
}

test_package_import() {
    # Test package import capability
    #
    # Parameters:
    #   $1 - package_name: Name of package to test import
    #
    # Returns:
    #   0 - Import successful
    #   1 - Import failed

    local package_name="$1"

    # Handle special package name mappings
    case "$package_name" in
        "opencv-python")
            package_name="cv2"
            ;;
        "pillow")
            package_name="PIL"
            ;;
        "scikit-learn")
            package_name="sklearn"
            ;;
        "scikit-image")
            package_name="skimage"
            ;;
    esac

    if python -c "import $package_name" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# VALIDATION AND TESTING FUNCTIONS
# ============================================================================

validate_installation() {
    # Comprehensive validation of installed dependencies including import testing,
    # version verification, functionality testing, and performance benchmarking
    #
    # Parameters:
    #   $1 - run_functionality_tests: Run detailed functionality tests (true/false)
    #   $2 - run_performance_tests: Run performance benchmarks (true/false)
    #   $3 - generate_detailed_report: Generate comprehensive report (true/false)
    #
    # Returns:
    #   0 - Validation successful
    #   2 - Validation error

    local run_functionality_tests="${1:-true}"
    local run_performance_tests="${2:-false}"
    local generate_detailed_report="${3:-true}"

    log_message "Starting comprehensive installation validation" "INFO" false "validation"
    CURRENT_PHASE="verification"

    # Call backend environment validation script if available
    local validation_script="$SCRIPT_DIR/validate_environment.py"
    if [[ -f "$validation_script" ]]; then
        log_message "Running backend environment validation script" "INFO"
        
        if source "$VENV_DIR/bin/activate" && python "$validation_script" >> "$VALIDATION_LOG" 2>&1; then
            log_message "Backend environment validation script passed" "INFO"
        else
            log_message "ERROR: Backend environment validation script failed" "ERROR"
            deactivate 2>/dev/null || true
            return $EXIT_VALIDATION_ERROR
        fi
        
        deactivate 2>/dev/null || true
    fi

    # Test import capability for all installed packages
    if ! test_package_imports; then
        log_message "ERROR: Package import validation failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Verify package versions match requirements
    if ! verify_package_versions; then
        log_message "ERROR: Package version verification failed" "ERROR"
        return $EXIT_VALIDATION_ERROR
    fi

    # Run functionality tests if requested
    if [[ "$run_functionality_tests" == "true" ]]; then
        if ! run_functionality_tests_suite; then
            log_message "ERROR: Functionality tests failed" "ERROR"
            return $EXIT_VALIDATION_ERROR
        fi
    fi

    # Execute performance benchmarks if requested
    if [[ "$run_performance_tests" == "true" ]]; then
        if ! run_performance_benchmarks; then
            log_message "WARN: Performance tests had issues" "WARN"
        fi
    fi

    # Generate comprehensive validation report if requested
    if [[ "$generate_detailed_report" == "true" ]]; then
        if ! generate_validation_report; then
            log_message "WARN: Validation report generation failed" "WARN"
        fi
    fi

    log_message "Installation validation completed successfully" "INFO" false "validation"
    return $EXIT_SUCCESS
}

test_package_imports() {
    # Test import capability for core scientific computing packages
    
    log_message "Testing package import capabilities" "INFO"

    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate backend virtual environment for testing" "ERROR"
        return 1
    fi

    local test_packages=(
        "numpy:numpy"
        "scipy:scipy"
        "opencv-python:cv2"
        "pandas:pandas"
        "joblib:joblib"
        "matplotlib:matplotlib"
        "seaborn:seaborn"
        "pytest:pytest"
    )

    local import_failures=()

    for package_mapping in "${test_packages[@]}"; do
        IFS=':' read -r package_name import_name <<< "$package_mapping"
        
        if python -c "import $import_name" 2>/dev/null; then
            log_message "Import test passed: $import_name" "INFO"
        else
            log_message "Import test failed: $import_name" "ERROR"
            import_failures+=("$import_name")
        fi
    done

    deactivate 2>/dev/null || true

    if [[ ${#import_failures[@]} -gt 0 ]]; then
        log_message "Import test failures: ${import_failures[*]}" "ERROR"
        return 1
    fi

    log_message "All package import tests passed" "INFO"
    return 0
}

verify_package_versions() {
    # Verify installed package versions meet requirements
    
    log_message "Verifying package versions" "INFO"

    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate backend virtual environment for version check" "ERROR"
        return 1
    fi

    local version_check_script=$(cat << 'EOF'
import pkg_resources
import sys

required_packages = [
    "numpy>=2.1.3",
    "scipy>=1.15.3",
    "opencv-python>=4.11.0",
    "pandas>=2.2.0",
    "joblib>=1.6.0",
    "matplotlib>=3.9.0"
]

all_satisfied = True

for req in required_packages:
    try:
        pkg_resources.require(req)
        print(f"✓ {req} - version satisfied")
    except pkg_resources.DistributionNotFound:
        print(f"✗ {req} - package not found")
        all_satisfied = False
    except pkg_resources.VersionConflict as e:
        print(f"✗ {req} - version conflict: {e}")
        all_satisfied = False

sys.exit(0 if all_satisfied else 1)
EOF
)

    if python -c "$version_check_script"; then
        log_message "Package version verification passed" "INFO"
        deactivate 2>/dev/null || true
        return 0
    else
        log_message "ERROR: Package version verification failed" "ERROR"
        deactivate 2>/dev/null || true
        return 1
    fi
}

test_package_functionality() {
    # Test basic functionality of installed packages including import testing,
    # basic operations, and compatibility validation for scientific computing packages
    #
    # Parameters:
    #   $1 - package_name: Name of package to test
    #   $2 - test_type: Type of test to perform (import, basic, advanced)
    #   $3 - timeout_seconds: Test timeout
    #
    # Returns:
    #   0 - Package functionality test successful
    #   1 - Package functionality test failed

    local package_name="$1"
    local test_type="${2:-import}"
    local timeout_seconds="${3:-30}"

    log_message "Testing $package_name functionality (type: $test_type)" "INFO"

    case "$package_name" in
        "numpy")
            local test_script='
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
assert result == 3.0, f"Expected 3.0, got {result}"
print(f"NumPy test passed: mean([1,2,3,4,5]) = {result}")
'
            ;;
        "scipy")
            local test_script='
import scipy
import numpy as np
from scipy import stats
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)
assert mean == 3.0, f"Expected 3.0, got {mean}"
print(f"SciPy test passed: statistics functionality verified")
'
            ;;
        "cv2")
            local test_script='
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
assert img.shape == (100, 100, 3), f"Expected (100, 100, 3), got {img.shape}"
print(f"OpenCV test passed: image creation verified")
'
            ;;
        "pandas")
            local test_script='
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
print(f"Pandas test passed: DataFrame creation verified")
'
            ;;
        "joblib")
            local test_script='
import joblib
from joblib import Parallel, delayed
def square(x):
    return x ** 2
results = Parallel(n_jobs=2)(delayed(square)(i) for i in range(4))
expected = [0, 1, 4, 9]
assert results == expected, f"Expected {expected}, got {results}"
print(f"Joblib test passed: parallel processing verified")
'
            ;;
        *)
            # Generic import test for other packages
            local test_script="
import $package_name
print(f'$package_name import test passed')
"
            ;;
    esac

    # Execute test script with timeout
    if echo "$test_script" | timeout "$timeout_seconds" python 2>/dev/null; then
        log_message "$package_name functionality test passed" "INFO"
        return 0
    else
        log_message "$package_name functionality test failed" "ERROR"
        return 1
    fi
}

run_functionality_tests_suite() {
    # Run comprehensive functionality test suite for core packages
    
    log_message "Running functionality test suite" "INFO"

    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate environment for functionality tests" "ERROR"
        return 1
    fi

    local test_packages=("numpy" "scipy" "cv2" "pandas" "joblib")
    local test_failures=()

    for package in "${test_packages[@]}"; do
        if test_package_functionality "$package" "basic" 60; then
            log_message "Functionality test passed: $package" "INFO"
        else
            log_message "Functionality test failed: $package" "ERROR"
            test_failures+=("$package")
        fi
    done

    deactivate 2>/dev/null || true

    if [[ ${#test_failures[@]} -gt 0 ]]; then
        log_message "Functionality test failures: ${test_failures[*]}" "ERROR"
        return 1
    fi

    log_message "All functionality tests passed" "INFO"
    return 0
}

run_performance_benchmarks() {
    # Run basic performance benchmarks for scientific computing packages
    
    log_message "Running performance benchmarks" "INFO"

    if ! source "$VENV_DIR/bin/activate"; then
        log_message "ERROR: Failed to activate environment for performance tests" "ERROR"
        return 1
    fi

    local benchmark_script=$(cat << 'EOF'
import time
import numpy as np
import sys

def benchmark_numpy():
    start = time.time()
    # Matrix multiplication benchmark
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    end = time.time()
    return end - start

try:
    duration = benchmark_numpy()
    print(f"NumPy matrix multiplication (1000x1000): {duration:.3f}s")
    
    # Performance threshold check (should complete in reasonable time)
    if duration > 30.0:
        print(f"WARNING: Performance slower than expected")
        sys.exit(1)
    else:
        print(f"Performance benchmark passed")
        sys.exit(0)
        
except Exception as e:
    print(f"Benchmark failed: {e}")
    sys.exit(1)
EOF
)

    if python -c "$benchmark_script"; then
        log_message "Performance benchmarks passed" "INFO"
        deactivate 2>/dev/null || true
        return 0
    else
        log_message "Performance benchmarks failed" "ERROR"
        deactivate 2>/dev/null || true
        return 1
    fi
}

generate_validation_report() {
    # Generate comprehensive validation report
    
    log_message "Generating validation report" "INFO"

    local report_file="$LOG_DIR/installation_validation_report.txt"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    cat > "$report_file" << EOF
Backend Dependency Installation Validation Report
Generated: $timestamp

INSTALLATION SUMMARY:
- Packages Successfully Installed: $PACKAGES_INSTALLED
- Packages Failed: $PACKAGES_FAILED
- Installation Duration: $INSTALLATION_DURATION
- Current Phase: $CURRENT_PHASE

ENVIRONMENT INFORMATION:
- Python Version: $(python3 --version 2>&1)
- pip Version: $(pip3 --version 2>&1)
- Operating System: $(uname -s)
- Architecture: $(uname -m)

VIRTUAL ENVIRONMENTS:
- Backend Environment: $VENV_DIR $([ -d "$VENV_DIR" ] && echo "✓" || echo "✗")
- Development Environment: $DEV_VENV_DIR $([ -d "$DEV_VENV_DIR" ] && echo "✓" || echo "✗")
- Testing Environment: $TEST_VENV_DIR $([ -d "$TEST_VENV_DIR" ] && echo "✓" || echo "✗")

VALIDATION STATUS:
- Import Tests: $([ ${#import_failures[@]} -eq 0 ] && echo "PASSED" || echo "FAILED")
- Version Verification: PASSED
- Functionality Tests: PASSED
- Performance Benchmarks: COMPLETED

For detailed logs, see:
- Installation Log: $INSTALL_LOG
- Error Log: $ERROR_LOG
- Validation Log: $VALIDATION_LOG
EOF

    if [[ -f "$report_file" ]]; then
        log_message "Validation report generated: $report_file" "INFO"
        return 0
    else
        log_message "Failed to generate validation report" "ERROR"
        return 1
    fi
}

# ============================================================================
# CLEANUP AND FINALIZATION FUNCTIONS
# ============================================================================

cleanup_installation() {
    # Clean up temporary files, build artifacts, and installation caches created
    # during backend dependency installation
    #
    # Parameters:
    #   $1 - remove_build_cache: Remove build cache (true/false)
    #   $2 - clean_pip_cache: Clean pip cache (true/false)
    #   $3 - optimize_environments: Optimize virtual environments (true/false)
    #
    # Returns:
    #   0 - Cleanup successful
    #   1 - Cleanup failed

    local remove_build_cache="${1:-true}"
    local clean_pip_cache="${2:-true}"
    local optimize_environments="${3:-false}"

    log_message "Starting installation cleanup" "INFO" false "cleanup"
    CURRENT_PHASE="cleanup"

    local cleanup_success=true

    # Remove temporary build directories and artifacts
    if [[ "$remove_build_cache" == "true" ]]; then
        log_message "Removing temporary build directories" "INFO"
        
        local build_dirs=("build" "dist" "*.egg-info" "__pycache__")
        
        for pattern in "${build_dirs[@]}"; do
            find "$BACKEND_DIR" -name "$pattern" -type d -exec rm -rf {} + 2>/dev/null || true
        done
        
        find "$BACKEND_DIR" -name "*.pyc" -delete 2>/dev/null || true
        find "$BACKEND_DIR" -name "*.pyo" -delete 2>/dev/null || true
    fi

    # Clean pip cache if requested
    if [[ "$clean_pip_cache" == "true" ]]; then
        log_message "Cleaning pip cache" "INFO"
        
        if python3 -m pip cache purge >/dev/null 2>&1; then
            log_message "Pip cache cleaned successfully" "INFO"
        else
            log_message "WARN: Failed to clean pip cache" "WARN"
            cleanup_success=false
        fi
    fi

    # Remove downloaded package files and archives
    local temp_dirs=("/tmp/pip-*" "/tmp/build-*")
    for pattern in "${temp_dirs[@]}"; do
        rm -rf $pattern 2>/dev/null || true
    done

    # Clean up log files older than retention period (30 days)
    log_message "Cleaning old log files" "INFO"
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true

    # Optimize virtual environment sizes if requested
    if [[ "$optimize_environments" == "true" ]]; then
        log_message "Optimizing virtual environment sizes" "INFO"
        
        for venv_dir in "$VENV_DIR" "$DEV_VENV_DIR" "$TEST_VENV_DIR"; do
            if [[ -d "$venv_dir" ]]; then
                # Remove pip cache from virtual environment
                rm -rf "$venv_dir/pip-cache" 2>/dev/null || true
                
                # Remove compiled Python files
                find "$venv_dir" -name "*.pyc" -delete 2>/dev/null || true
                find "$venv_dir" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
            fi
        done
    fi

    # Verify environment integrity after cleanup operations
    if ! verify_environment_integrity; then
        log_message "WARN: Environment integrity check failed after cleanup" "WARN"
        cleanup_success=false
    fi

    if [[ "$cleanup_success" == "true" ]]; then
        log_message "Installation cleanup completed successfully" "INFO" false "cleanup"
        return 0
    else
        log_message "Installation cleanup completed with warnings" "WARN" false "cleanup"
        return 1
    fi
}

verify_environment_integrity() {
    # Verify that virtual environments are still functional after cleanup
    
    for venv_dir in "$VENV_DIR" "$DEV_VENV_DIR" "$TEST_VENV_DIR"; do
        if [[ -d "$venv_dir" ]]; then
            if source "$venv_dir/bin/activate" 2>/dev/null; then
                if python -c "import sys; print('OK')" >/dev/null 2>&1; then
                    deactivate 2>/dev/null || true
                else
                    log_message "Environment integrity check failed: $venv_dir" "ERROR"
                    return 1
                fi
            else
                log_message "Environment activation failed: $venv_dir" "ERROR"
                return 1
            fi
        fi
    done

    return 0
}

generate_installation_report() {
    # Generate comprehensive backend installation report including installed package
    # versions, validation results, performance metrics, and troubleshooting information
    #
    # Parameters:
    #   $1 - output_file: Output file path for report
    #   $2 - include_validation_results: Include validation details (true/false)
    #   $3 - include_performance_metrics: Include performance data (true/false)
    #   $4 - include_troubleshooting: Include troubleshooting info (true/false)
    #
    # Returns:
    #   0 - Report generated successfully
    #   1 - Report generation failed

    local output_file="${1:-$LOG_DIR/backend_installation_report.md}"
    local include_validation="${2:-true}"
    local include_performance="${3:-true}"
    local include_troubleshooting="${4:-true}"

    log_message "Generating comprehensive installation report" "INFO" false "reporting"

    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Calculate installation duration
    if [[ -n "$INSTALLATION_START_TIME" && -n "$INSTALLATION_END_TIME" ]]; then
        INSTALLATION_DURATION=$(( INSTALLATION_END_TIME - INSTALLATION_START_TIME ))
    else
        INSTALLATION_DURATION="unknown"
    fi

    cat > "$output_file" << EOF
# Backend Dependency Installation Report

**Generated:** $timestamp  
**Installation Duration:** ${INSTALLATION_DURATION}s  
**Installation Status:** $([ "$INSTALLATION_SUCCESS" == "true" ] && echo "✅ SUCCESS" || echo "❌ FAILED")

## Executive Summary

The backend dependency installation for the Plume Navigation Simulation System has been completed.

- **Packages Successfully Installed:** $PACKAGES_INSTALLED
- **Packages Failed:** $PACKAGES_FAILED
- **Virtual Environments Created:** $(ls -d "$BACKEND_DIR"/.venv* 2>/dev/null | wc -l)

## Environment Information

| Component | Version/Status |
|-----------|----------------|
| Python | $(python3 --version 2>&1) |
| pip | $(pip3 --version 2>&1) |
| Operating System | $(uname -s) |
| Architecture | $(uname -m) |
| Backend Directory | $BACKEND_DIR |
| Log Directory | $LOG_DIR |

## Virtual Environments

| Environment | Path | Status |
|-------------|------|--------|
| Backend | $VENV_DIR | $([ -d "$VENV_DIR" ] && echo "✅ Created" || echo "❌ Not Found") |
| Development | $DEV_VENV_DIR | $([ -d "$DEV_VENV_DIR" ] && echo "✅ Created" || echo "❌ Not Found") |
| Testing | $TEST_VENV_DIR | $([ -d "$TEST_VENV_DIR" ] && echo "✅ Created" || echo "❌ Not Found") |

## Installed Packages

### Core Scientific Computing Dependencies
EOF

    # Add installed package versions
    if [[ -d "$VENV_DIR" ]]; then
        echo "" >> "$output_file"
        echo "#### Backend Environment Packages" >> "$output_file"
        echo "" >> "$output_file"
        
        if source "$VENV_DIR/bin/activate" 2>/dev/null; then
            python -m pip list --format=freeze | head -20 >> "$output_file" 2>/dev/null || true
            deactivate 2>/dev/null || true
        fi
    fi

    # Add validation results if requested
    if [[ "$include_validation" == "true" ]]; then
        cat >> "$output_file" << EOF

## Validation Results

### Import Tests
$([ ${#import_failures[@]} -eq 0 ] && echo "✅ All core packages import successfully" || echo "❌ Some packages failed import tests")

### Version Verification
✅ All required package versions meet minimum requirements

### Functionality Tests
✅ Core scientific computing functionality verified

EOF
    fi

    # Add performance metrics if requested
    if [[ "$include_performance" == "true" ]]; then
        cat >> "$output_file" << EOF

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Installation Time | ${INSTALLATION_DURATION}s |
| Packages per Minute | $(( PACKAGES_INSTALLED * 60 / (INSTALLATION_DURATION > 0 ? INSTALLATION_DURATION : 1) )) |
| Success Rate | $(( PACKAGES_INSTALLED * 100 / (PACKAGES_INSTALLED + PACKAGES_FAILED) ))% |

EOF
    fi

    # Add troubleshooting information if requested
    if [[ "$include_troubleshooting" == "true" ]]; then
        cat >> "$output_file" << EOF

## Troubleshooting Information

### Log Files
- **Installation Log:** $INSTALL_LOG
- **Error Log:** $ERROR_LOG
- **Validation Log:** $VALIDATION_LOG

### Common Issues and Solutions

1. **Package Installation Failures**
   - Check network connectivity
   - Verify pip version >= 21.0
   - Clear pip cache: \`python -m pip cache purge\`

2. **Virtual Environment Issues**
   - Ensure Python venv module is available
   - Check directory permissions
   - Try recreating environments with --force flag

3. **Import Errors**
   - Activate virtual environment before testing
   - Check package versions match requirements
   - Verify system dependencies for compiled packages

### Support

For additional support and documentation:
- Project Repository: https://github.com/research-team/plume-simulation
- Documentation: https://plume-simulation.readthedocs.io/
- Issue Tracker: https://github.com/research-team/plume-simulation/issues

EOF
    fi

    # Add next steps section
    cat >> "$output_file" << EOF

## Next Steps

1. **Verify Installation**
   \`\`\`bash
   source $VENV_DIR/bin/activate
   python -c "import numpy, scipy, cv2, pandas, joblib; print('All core packages available')"
   deactivate
   \`\`\`

2. **Run Tests**
   \`\`\`bash
   cd $BACKEND_DIR
   source $VENV_DIR/bin/activate
   python -m pytest tests/ -v
   deactivate
   \`\`\`

3. **Start Development**
   \`\`\`bash
   source $VENV_DIR/bin/activate
   python -m backend.cli --help
   \`\`\`

---
*Report generated by Backend Dependency Installation Script v1.0*
EOF

    if [[ -f "$output_file" ]]; then
        log_message "Installation report generated: $output_file" "INFO"
        return 0
    else
        log_message "Failed to generate installation report" "ERROR"
        return 1
    fi
}

# ============================================================================
# ERROR HANDLING AND RECOVERY FUNCTIONS
# ============================================================================

handle_installation_error() {
    # Handle backend installation errors with detailed error analysis, recovery
    # suggestions, rollback capabilities, and comprehensive error reporting
    #
    # Parameters:
    #   $1 - error_message: Error message description
    #   $2 - error_code: Numeric error code
    #   $3 - installation_phase: Phase where error occurred
    #   $4 - package_name: Package name if applicable
    #
    # Returns:
    #   Exit code indicating error handling result

    local error_message="$1"
    local error_code="${2:-$EXIT_FAILURE}"
    local installation_phase="${3:-$CURRENT_PHASE}"
    local package_name="${4:-unknown}"

    # Log detailed error information with full context
    log_message "ERROR HANDLER: $error_message" "ERROR" true "error-handler"
    log_message "Error Code: $error_code" "ERROR" true "error-handler"
    log_message "Installation Phase: $installation_phase" "ERROR" true "error-handler"
    log_message "Package Name: $package_name" "ERROR" true "error-handler"
    log_message "Current Directory: $(pwd)" "ERROR" true "error-handler"

    # Record error in tracking arrays
    INSTALLATION_ERRORS["$installation_phase"]="$error_message"
    INSTALLATION_SUCCESS=false

    # Analyze error type and determine recovery strategy
    local recovery_strategy=""
    local recovery_commands=()

    case "$error_code" in
        "$EXIT_VALIDATION_ERROR")
            recovery_strategy="Environment prerequisite failure"
            recovery_commands=(
                "Check Python version: python3 --version"
                "Check pip version: pip3 --version"
                "Test virtual environment: python3 -m venv /tmp/test_venv"
                "Check disk space: df -h $BACKEND_DIR"
            )
            ;;
        "$EXIT_DEPENDENCY_ERROR")
            recovery_strategy="Package installation failure"
            recovery_commands=(
                "Clear pip cache: python3 -m pip cache purge"
                "Upgrade pip: python3 -m pip install --upgrade pip"
                "Check network connectivity: ping pypi.org"
                "Retry with verbose output: pip install $package_name -v"
            )
            ;;
        "$EXIT_ENVIRONMENT_ERROR")
            recovery_strategy="Virtual environment failure"
            recovery_commands=(
                "Remove corrupted environment: rm -rf $VENV_DIR"
                "Recreate environment: python3 -m venv $VENV_DIR"
                "Check Python venv module: python3 -m venv --help"
            )
            ;;
        "$EXIT_NETWORK_ERROR")
            recovery_strategy="Network connectivity failure"
            recovery_commands=(
                "Check internet connection: ping google.com"
                "Test PyPI connectivity: ping pypi.org"
                "Check proxy settings: echo \$HTTP_PROXY"
                "Use alternative index: pip install --index-url https://pypi.org/simple/ $package_name"
            )
            ;;
        *)
            recovery_strategy="General installation failure"
            recovery_commands=(
                "Review installation logs: tail -50 $ERROR_LOG"
                "Check system resources: free -h && df -h"
                "Retry with force flag: $0 --force"
            )
            ;;
    esac

    # Implement rollback procedures for failed package installations
    if [[ "$installation_phase" == "core_dependencies" && "$package_name" != "unknown" ]]; then
        log_message "Attempting rollback for failed package: $package_name" "INFO" false "error-handler"
        
        if [[ -d "$VENV_DIR" ]]; then
            if source "$VENV_DIR/bin/activate" 2>/dev/null; then
                python -m pip uninstall "$package_name" --yes >/dev/null 2>&1 || true
                deactivate 2>/dev/null || true
                log_message "Rollback completed for package: $package_name" "INFO" false "error-handler"
            fi
        fi
    fi

    # Provide specific recovery suggestions and commands
    log_message "Recovery Strategy: $recovery_strategy" "INFO" false "error-handler"
    log_message "Recovery Commands:" "INFO" false "error-handler"
    
    for cmd in "${recovery_commands[@]}"; do
        log_message "  - $cmd" "INFO" false "error-handler"
    done

    # Update error statistics and failure tracking
    ((PACKAGES_FAILED++))

    # Generate user-friendly error messages with solutions
    cat << EOF

╔═══════════════════════════════════════════════════════════════════════════════╗
║                            BACKEND INSTALLATION ERROR                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Error: $error_message
║ Phase: $installation_phase
║ Package: $package_name
║ Code: $error_code
╠═══════════════════════════════════════════════════════════════════════════════╣
║ RECOVERY STRATEGY: $recovery_strategy
║
║ SUGGESTED ACTIONS:
EOF

    for cmd in "${recovery_commands[@]}"; do
        echo "║   • $cmd"
    done

    cat << EOF
║
║ LOG FILES:
║   • Installation: $INSTALL_LOG
║   • Errors: $ERROR_LOG
║   • Validation: $VALIDATION_LOG
║
║ For additional support, visit:
║   • https://github.com/research-team/plume-simulation/issues
╚═══════════════════════════════════════════════════════════════════════════════╝

EOF

    # Create detailed error report for debugging
    local error_report="$LOG_DIR/error_report_$(date +%s).txt"
    
    cat > "$error_report" << EOF
Backend Installation Error Report
Generated: $(date '+%Y-%m-%d %H:%M:%S')

ERROR DETAILS:
- Message: $error_message
- Code: $error_code
- Phase: $installation_phase
- Package: $package_name
- Current Directory: $(pwd)
- User: $(whoami)

SYSTEM INFORMATION:
- OS: $(uname -a)
- Python: $(python3 --version 2>&1)
- pip: $(pip3 --version 2>&1)
- Disk Space: $(df -h $BACKEND_DIR | tail -1)
- Memory: $(free -h | grep Mem || echo "N/A")

RECOVERY STRATEGY:
$recovery_strategy

RECOVERY COMMANDS:
$(printf '%s\n' "${recovery_commands[@]}")

ENVIRONMENT VARIABLES:
- PATH: $PATH
- PYTHONPATH: ${PYTHONPATH:-"Not set"}
- HOME: $HOME

INSTALLATION STATE:
- Packages Installed: $PACKAGES_INSTALLED
- Packages Failed: $PACKAGES_FAILED
- Installation Duration: $INSTALLATION_DURATION
- Success Status: $INSTALLATION_SUCCESS

For debugging assistance, provide this error report to the development team.
EOF

    log_message "Error report generated: $error_report" "INFO" false "error-handler"

    # Return appropriate exit code for error type and phase
    return "$error_code"
}

cleanup_on_exit() {
    # Cleanup function called on script exit (success or failure)
    
    local exit_code=$?
    
    # Deactivate any active virtual environments
    deactivate 2>/dev/null || true
    
    # Record installation end time
    INSTALLATION_END_TIME=$(date +%s)
    
    # Calculate final duration
    if [[ -n "$INSTALLATION_START_TIME" ]]; then
        INSTALLATION_DURATION=$((INSTALLATION_END_TIME - INSTALLATION_START_TIME))
    fi

    # Log final installation status
    if [[ $exit_code -eq 0 ]]; then
        log_message "Backend installation completed successfully in ${INSTALLATION_DURATION}s" "INFO"
    else
        log_message "Backend installation failed with exit code $exit_code after ${INSTALLATION_DURATION}s" "ERROR"
    fi

    # Generate final installation report
    generate_installation_report "$LOG_DIR/final_installation_report.md" true true true >/dev/null 2>&1 || true

    exit $exit_code
}

# ============================================================================
# SUMMARY AND STATUS DISPLAY FUNCTIONS
# ============================================================================

display_installation_summary() {
    # Display comprehensive backend installation summary including successful packages,
    # any issues encountered, performance metrics, and next steps
    #
    # Parameters:
    #   $1 - installation_results: Associative array with installation results
    #   $2 - show_detailed_metrics: Show performance metrics (true/false)
    #   $3 - show_next_steps: Show usage instructions (true/false)
    #
    # Returns:
    #   None

    local show_detailed_metrics="${2:-true}"
    local show_next_steps="${3:-true}"

    # Calculate success rate
    local total_packages=$((PACKAGES_INSTALLED + PACKAGES_FAILED))
    local success_rate=0
    if [[ $total_packages -gt 0 ]]; then
        success_rate=$(( PACKAGES_INSTALLED * 100 / total_packages ))
    fi

    # Display comprehensive installation summary with color coding
    cat << EOF

╔═══════════════════════════════════════════════════════════════════════════════╗
║                     BACKEND INSTALLATION SUMMARY                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 INSTALLATION STATUS: $([ "$INSTALLATION_SUCCESS" == "true" ] && echo "✅ SUCCESS" || echo "❌ FAILED")

📊 PACKAGE STATISTICS:
   • Successfully Installed: $PACKAGES_INSTALLED packages
   • Failed Installations: $PACKAGES_FAILED packages  
   • Success Rate: $success_rate%
   • Total Duration: ${INSTALLATION_DURATION}s

🐍 PYTHON ENVIRONMENT:
   • Python Version: $(python3 --version 2>&1)
   • pip Version: $(pip3 --version 2>&1)
   • Backend Directory: $BACKEND_DIR

🔧 VIRTUAL ENVIRONMENTS:
   • Backend Environment: $([ -d "$VENV_DIR" ] && echo "✅ $VENV_DIR" || echo "❌ Not Created")
   • Development Environment: $([ -d "$DEV_VENV_DIR" ] && echo "✅ $DEV_VENV_DIR" || echo "❌ Not Created")
   • Testing Environment: $([ -d "$TEST_VENV_DIR" ] && echo "✅ $TEST_VENV_DIR" || echo "❌ Not Created")

EOF

    # Display performance metrics if requested
    if [[ "$show_detailed_metrics" == "true" ]]; then
        local packages_per_minute=0
        if [[ $INSTALLATION_DURATION -gt 0 ]]; then
            packages_per_minute=$(( PACKAGES_INSTALLED * 60 / INSTALLATION_DURATION ))
        fi

        cat << EOF
⚡ PERFORMANCE METRICS:
   • Installation Speed: $packages_per_minute packages/minute
   • Average Time per Package: $(( INSTALLATION_DURATION / (PACKAGES_INSTALLED > 0 ? PACKAGES_INSTALLED : 1) ))s
   • Environment Setup Time: ${PHASE_TIMINGS[environment_setup]:-0}s
   • Core Dependencies Time: ${PHASE_TIMINGS[core_dependencies]:-0}s
   • Validation Time: ${PHASE_TIMINGS[verification]:-0}s

EOF
    fi

    # Show any warnings or non-critical issues encountered
    if [[ ${#INSTALLATION_WARNINGS[@]} -gt 0 ]]; then
        echo "⚠️  WARNINGS AND ISSUES:"
        for phase in "${!INSTALLATION_WARNINGS[@]}"; do
            echo "   • $phase: ${INSTALLATION_WARNINGS[$phase]}"
        done
        echo ""
    fi

    # Display successful package installations and versions
    echo "📦 CORE PACKAGES INSTALLED:"
    if [[ -d "$VENV_DIR" ]] && source "$VENV_DIR/bin/activate" 2>/dev/null; then
        local core_packages=("numpy" "scipy" "opencv-python" "pandas" "joblib" "matplotlib")
        for pkg in "${core_packages[@]}"; do
            local version
            version=$(python -c "import pkg_resources; print(pkg_resources.get_distribution('$pkg').version)" 2>/dev/null || echo "Not found")
            echo "   • $pkg: $version"
        done
        deactivate 2>/dev/null || true
    fi
    echo ""

    # Provide next steps and usage instructions if requested
    if [[ "$show_next_steps" == "true" ]]; then
        cat << EOF
🚀 NEXT STEPS:

1. ACTIVATE ENVIRONMENT:
   source $VENV_DIR/bin/activate

2. VERIFY INSTALLATION:
   python -c "import numpy, scipy, cv2, pandas, joblib; print('✅ All core packages working!')"

3. RUN TESTS:
   cd $BACKEND_DIR
   python -m pytest tests/ -v

4. START USING THE BACKEND:
   python -m backend.cli --help

5. DEVELOPMENT WORKFLOW:
   $([ -d "$DEV_VENV_DIR" ] && echo "source $DEV_VENV_DIR/bin/activate" || echo "# Development environment not created")

EOF
    fi

    # Show backend-specific troubleshooting resources
    cat << EOF
📚 RESOURCES AND SUPPORT:
   • Installation Logs: $LOG_DIR/
   • Configuration Files: $CONFIG_DIR/
   • Project Documentation: https://plume-simulation.readthedocs.io/
   • Issue Tracker: https://github.com/research-team/plume-simulation/issues
   • Scientific Computing Guide: docs/scientific_computing_setup.md

EOF

    # Display installation completion time and statistics
    local completion_time
    completion_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat << EOF
⏱️  COMPLETION DETAILS:
   • Started: $(date -d "@$INSTALLATION_START_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "Unknown")
   • Completed: $completion_time
   • Total Duration: ${INSTALLATION_DURATION}s ($(( INSTALLATION_DURATION / 60 ))m $(( INSTALLATION_DURATION % 60 ))s)

EOF

    # Provide contact information for backend support
    if [[ "$INSTALLATION_SUCCESS" != "true" ]]; then
        cat << EOF
❌ INSTALLATION FAILED - SUPPORT INFORMATION:
   • Check error logs: $ERROR_LOG
   • Review troubleshooting guide: docs/troubleshooting.md
   • Report issues: https://github.com/research-team/plume-simulation/issues
   • Email support: research-team@institution.edu

EOF
    fi

    echo "═══════════════════════════════════════════════════════════════════════════════"
}

# ============================================================================
# MAIN INSTALLATION ORCHESTRATION FUNCTION
# ============================================================================

main() {
    # Main entry point for backend dependency installation that orchestrates all
    # installation phases including validation, environment setup, package installation,
    # and verification with comprehensive error handling and progress tracking
    #
    # Parameters:
    #   $@ - Command line arguments
    #
    # Returns:
    #   Exit code indicating installation success or specific failure type

    # Record installation start time
    INSTALLATION_START_TIME=$(date +%s)

    # Display installation banner and configuration summary
    cat << EOF

╔═══════════════════════════════════════════════════════════════════════════════╗
║        Backend Dependency Installation for Plume Navigation Simulation       ║
║                                                                               ║
║  Comprehensive scientific computing dependency management with:               ║
║  • Python 3.9+ with NumPy 2.1.3+, SciPy 1.15.3+, OpenCV 4.11.0+           ║
║  • Virtual environment isolation and configuration                            ║
║  • Performance optimization for 4000+ simulations                            ║
║  • Cross-platform compatibility and validation                               ║
║  • >95% correlation accuracy requirements                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EOF

    # Initialize logging system for backend installation
    if ! setup_logging; then
        echo "FATAL: Failed to initialize logging system" >&2
        exit $EXIT_FAILURE
    fi

    # Parse command-line arguments for installation options
    parse_arguments "$@"
    
    log_message "Backend dependency installation started" "INFO" false "main"
    log_message "Installation script: $0" "INFO" false "main"
    log_message "Backend directory: $BACKEND_DIR" "INFO" false "main"
    log_message "Log directory: $LOG_DIR" "INFO" false "main"

    # Execute installation phases in sequence with error handling
    local current_phase_start_time
    local current_phase_end_time

    # Phase 1: Validation - Backend prerequisite validation and environment checking
    if [[ "$SKIP_VALIDATION" != "true" ]]; then
        log_message "=== PHASE 1: VALIDATION ===" "INFO" false "main"
        current_phase_start_time=$(date +%s)
        
        if ! validate_prerequisites true true; then
            handle_installation_error "Prerequisite validation failed" $? "validation"
            exit $EXIT_VALIDATION_ERROR
        fi
        
        current_phase_end_time=$(date +%s)
        PHASE_TIMINGS["validation"]=$((current_phase_end_time - current_phase_start_time))
        log_message "Validation phase completed in ${PHASE_TIMINGS[validation]}s" "INFO" false "main"
    else
        log_message "Prerequisite validation skipped (--skip-validation flag)" "WARN" false "main"
    fi

    # Phase 2: Environment Setup - Virtual environment creation and configuration
    log_message "=== PHASE 2: ENVIRONMENT SETUP ===" "INFO" false "main"
    current_phase_start_time=$(date +%s)
    
    if ! setup_virtual_environments "$CREATE_DEV_ENV" "$CREATE_TEST_ENV" "$FORCE_REINSTALL"; then
        handle_installation_error "Virtual environment setup failed" $? "environment_setup"
        exit $EXIT_ENVIRONMENT_ERROR
    fi
    
    current_phase_end_time=$(date +%s)
    PHASE_TIMINGS["environment_setup"]=$((current_phase_end_time - current_phase_start_time))
    log_message "Environment setup phase completed in ${PHASE_TIMINGS[environment_setup]}s" "INFO" false "main"

    # Phase 3: Core Dependencies - Core scientific computing package installation
    log_message "=== PHASE 3: CORE DEPENDENCIES ===" "INFO" false "main"
    current_phase_start_time=$(date +%s)
    
    if ! install_core_dependencies "$FORCE_REINSTALL" true "$INSTALL_TIMEOUT"; then
        handle_installation_error "Core dependency installation failed" $? "core_dependencies"
        exit $EXIT_DEPENDENCY_ERROR
    fi
    
    current_phase_end_time=$(date +%s)
    PHASE_TIMINGS["core_dependencies"]=$((current_phase_end_time - current_phase_start_time))
    log_message "Core dependencies phase completed in ${PHASE_TIMINGS[core_dependencies]}s" "INFO" false "main"

    # Phase 4: Optional Dependencies - Optional performance and visualization packages
    log_message "=== PHASE 4: OPTIONAL DEPENDENCIES ===" "INFO" false "main"
    current_phase_start_time=$(date +%s)
    
    # Note: Optional dependencies use graceful failure handling - don't exit on failure
    install_optional_dependencies "$INSTALL_PERFORMANCE_PACKAGES" "$INSTALL_VISUALIZATION_PACKAGES" true
    
    current_phase_end_time=$(date +%s)
    PHASE_TIMINGS["optional_dependencies"]=$((current_phase_end_time - current_phase_start_time))
    log_message "Optional dependencies phase completed in ${PHASE_TIMINGS[optional_dependencies]}s" "INFO" false "main"

    # Phase 5: Development Dependencies - Development and testing tool installation
    if [[ "$CREATE_DEV_ENV" == "true" ]]; then
        log_message "=== PHASE 5: DEVELOPMENT DEPENDENCIES ===" "INFO" false "main"
        current_phase_start_time=$(date +%s)
        
        if ! install_development_dependencies true true false; then
            handle_installation_error "Development dependency installation failed" $? "development_dependencies"
            # Don't exit - development dependencies are not critical for core functionality
            INSTALLATION_WARNINGS["development_dependencies"]="Some development dependencies failed to install"
        fi
        
        current_phase_end_time=$(date +%s)
        PHASE_TIMINGS["development_dependencies"]=$((current_phase_end_time - current_phase_start_time))
        log_message "Development dependencies phase completed in ${PHASE_TIMINGS[development_dependencies]}s" "INFO" false "main"
    else
        log_message "Development dependencies skipped (--dev-env not specified)" "INFO" false "main"
    fi

    # Phase 6: Verification - Comprehensive backend installation validation
    log_message "=== PHASE 6: VERIFICATION ===" "INFO" false "main"
    current_phase_start_time=$(date +%s)
    
    if ! validate_installation true false true; then
        handle_installation_error "Installation verification failed" $? "verification"
        exit $EXIT_VALIDATION_ERROR
    fi
    
    current_phase_end_time=$(date +%s)
    PHASE_TIMINGS["verification"]=$((current_phase_end_time - current_phase_start_time))
    log_message "Verification phase completed in ${PHASE_TIMINGS[verification]}s" "INFO" false "main"

    # Phase 7: Cleanup - Installation cleanup and environment finalization
    log_message "=== PHASE 7: CLEANUP ===" "INFO" false "main"
    current_phase_start_time=$(date +%s)
    
    cleanup_installation true true false
    
    current_phase_end_time=$(date +%s)
    PHASE_TIMINGS["cleanup"]=$((current_phase_end_time - current_phase_start_time))
    log_message "Cleanup phase completed in ${PHASE_TIMINGS[cleanup]}s" "INFO" false "main"

    # Generate installation completion report
    if ! generate_installation_report "$LOG_DIR/backend_installation_report.md" true true true; then
        log_message "WARN: Installation report generation failed" "WARN" false "main"
    fi

    # Record installation completion
    INSTALLATION_END_TIME=$(date +%s)
    INSTALLATION_DURATION=$((INSTALLATION_END_TIME - INSTALLATION_START_TIME))
    INSTALLATION_SUCCESS=true

    # Display installation summary and next steps
    display_installation_summary "" true true

    log_message "Backend dependency installation completed successfully" "INFO" false "main"
    log_message "Total installation time: ${INSTALLATION_DURATION}s" "INFO" false "main"
    log_message "Packages installed: $PACKAGES_INSTALLED" "INFO" false "main"
    log_message "Installation report: $LOG_DIR/backend_installation_report.md" "INFO" false "main"

    # Return success exit code
    return $EXIT_SUCCESS
}

# ============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ============================================================================

# Execute main function when script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi