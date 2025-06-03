#!/bin/bash

# Multi-level cache cleanup and optimization script for plume navigation simulation system
# Handles Level 1 (memory), Level 2 (disk), and Level 3 (result) cache coordination
# Provides comprehensive maintenance with performance optimization and system health monitoring

# Global Script Configuration
readonly SCRIPT_NAME="clean_cache.sh"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DESCRIPTION="Multi-level cache cleanup and optimization script"
readonly CACHE_BASE_DIR="${HOME}/.cache/plume_simulation"
readonly LOG_DIR="logs"
readonly CONFIG_DIR="src/backend/config"
readonly PYTHON_EXECUTABLE="python3"
readonly BACKEND_MODULE="src.backend"

# Default Configuration Parameters
readonly DEFAULT_CLEANUP_MODE="standard"
readonly DEFAULT_PRESERVE_HOT_DATA=true
readonly DEFAULT_FORCE_CLEANUP=false
readonly DEFAULT_TARGET_UTILIZATION=0.7
readonly DEFAULT_OPTIMIZE_AFTER_CLEANUP=true
readonly CLEANUP_TIMEOUT=1800
readonly PROGRESS_UPDATE_INTERVAL=5

# Color Scheme for Scientific Computing Interface
readonly COLOR_GREEN="\033[92m"
readonly COLOR_YELLOW="\033[93m"
readonly COLOR_RED="\033[91m"
readonly COLOR_BLUE="\033[94m"
readonly COLOR_CYAN="\033[96m"
readonly COLOR_BOLD="\033[1m"
readonly COLOR_RESET="\033[0m"

# Exit Status Codes
readonly EXIT_SUCCESS=0
readonly EXIT_ERROR=1
readonly EXIT_TIMEOUT=2
readonly EXIT_INVALID_ARGS=3

# Runtime Variables
CLEANUP_MODE="$DEFAULT_CLEANUP_MODE"
PRESERVE_HOT_DATA="$DEFAULT_PRESERVE_HOT_DATA"
FORCE_CLEANUP="$DEFAULT_FORCE_CLEANUP"
TARGET_UTILIZATION="$DEFAULT_TARGET_UTILIZATION"
OPTIMIZE_AFTER_CLEANUP="$DEFAULT_OPTIMIZE_AFTER_CLEANUP"
VERBOSE_MODE=false
DRY_RUN=false
LOG_FILE=""
SCRIPT_PID=$$
START_TIME=""
CLEANUP_STATS=""

# Cache Status Tracking
LEVEL1_STATUS=""
LEVEL2_STATUS=""
LEVEL3_STATUS=""
TOTAL_SPACE_FREED=0
PERFORMANCE_IMPROVEMENT=0

#######################################
# Display comprehensive usage information including command-line options,
# cleanup modes, and examples for cache cleanup operations
# Globals:
#   All script configuration variables
# Arguments:
#   None
# Returns:
#   None
#######################################
print_usage() {
    cat << EOF
${COLOR_BOLD}${COLOR_BLUE}========================================${COLOR_RESET}
${COLOR_BOLD}${SCRIPT_NAME} v${SCRIPT_VERSION}${COLOR_RESET}
${COLOR_CYAN}${SCRIPT_DESCRIPTION}${COLOR_RESET}
${COLOR_BOLD}${COLOR_BLUE}========================================${COLOR_RESET}

${COLOR_BOLD}SYNOPSIS:${COLOR_RESET}
    ${SCRIPT_NAME} [OPTIONS]

${COLOR_BOLD}DESCRIPTION:${COLOR_RESET}
    Comprehensive cache cleanup utility for multi-level caching architecture
    in plume navigation simulation system. Manages Level 1 (memory), Level 2 
    (disk), and Level 3 (result) caches with intelligent optimization.

${COLOR_BOLD}OPTIONS:${COLOR_RESET}
    ${COLOR_GREEN}-m, --mode MODE${COLOR_RESET}          Cleanup mode: standard, aggressive, conservative
                                Default: ${DEFAULT_CLEANUP_MODE}
    
    ${COLOR_GREEN}-t, --target RATIO${COLOR_RESET}       Target utilization ratio (0.0-1.0)
                                Default: ${DEFAULT_TARGET_UTILIZATION}
    
    ${COLOR_GREEN}-p, --preserve-hot${COLOR_RESET}       Preserve frequently accessed data
                                Default: ${DEFAULT_PRESERVE_HOT_DATA}
    
    ${COLOR_GREEN}-f, --force${COLOR_RESET}              Force cleanup without confirmation
                                Default: ${DEFAULT_FORCE_CLEANUP}
    
    ${COLOR_GREEN}-o, --optimize${COLOR_RESET}           Optimize cache coordination after cleanup
                                Default: ${DEFAULT_OPTIMIZE_AFTER_CLEANUP}
    
    ${COLOR_GREEN}-n, --dry-run${COLOR_RESET}            Show what would be cleaned without executing
    
    ${COLOR_GREEN}-v, --verbose${COLOR_RESET}            Enable verbose logging and progress output
    
    ${COLOR_GREEN}-h, --help${COLOR_RESET}               Display this help message

${COLOR_BOLD}CLEANUP MODES:${COLOR_RESET}
    ${COLOR_CYAN}standard${COLOR_RESET}     - Balanced cleanup maintaining performance thresholds
    ${COLOR_CYAN}aggressive${COLOR_RESET}   - Maximum space reclamation with minimal data preservation
    ${COLOR_CYAN}conservative${COLOR_RESET} - Minimal cleanup preserving maximum cache effectiveness

${COLOR_BOLD}EXAMPLES:${COLOR_RESET}
    # Standard cleanup with optimization
    ${COLOR_GREEN}./${SCRIPT_NAME}${COLOR_RESET}
    
    # Aggressive cleanup targeting 50% utilization
    ${COLOR_GREEN}./${SCRIPT_NAME} --mode aggressive --target 0.5${COLOR_RESET}
    
    # Conservative cleanup preserving hot data
    ${COLOR_GREEN}./${SCRIPT_NAME} --mode conservative --preserve-hot${COLOR_RESET}
    
    # Dry run to preview cleanup operations
    ${COLOR_GREEN}./${SCRIPT_NAME} --dry-run --verbose${COLOR_RESET}

${COLOR_BOLD}EXIT CODES:${COLOR_RESET}
    ${EXIT_SUCCESS} - Successful completion
    ${EXIT_ERROR} - General error during execution
    ${EXIT_TIMEOUT} - Operation timeout exceeded
    ${EXIT_INVALID_ARGS} - Invalid command line arguments

${COLOR_BOLD}CACHE ARCHITECTURE:${COLOR_RESET}
    ${COLOR_CYAN}Level 1${COLOR_RESET}: In-memory caching for active simulation data
    ${COLOR_CYAN}Level 2${COLOR_RESET}: Disk-based caching for normalized video data
    ${COLOR_CYAN}Level 3${COLOR_RESET}: Result caching for completed simulations

${COLOR_BOLD}PERFORMANCE TARGETS:${COLOR_RESET}
    Cache hit rate: ≥0.8, Memory fragmentation: ≤0.1
    Processing time: <7.2s average per simulation
    Batch capacity: 4000+ simulations within 8 hours

${COLOR_BOLD}TROUBLESHOOTING:${COLOR_RESET}
    - Ensure Python 3.9+ is available and backend module is importable
    - Verify cache directories exist and are writable
    - Check system resources (disk space, memory) before cleanup
    - Review log files in ${LOG_DIR}/ for detailed error information
    - Use --dry-run to preview operations without execution

EOF
}

#######################################
# Print colored text to console with ANSI color codes for enhanced
# readability and status indication following scientific computing color scheme
# Globals:
#   Color constants
# Arguments:
#   $1 - message: Text message to display
#   $2 - color: Color code (optional, defaults to reset)
#   $3 - bold: Enable bold formatting (optional, defaults to false)
# Returns:
#   None
#######################################
print_colored() {
    local message="$1"
    local color="${2:-$COLOR_RESET}"
    local bold="${3:-false}"
    
    # Check if terminal supports colors
    if [[ ! -t 1 ]] || [[ "${TERM:-}" == "dumb" ]]; then
        echo "$message"
        return
    fi
    
    local format_string=""
    if [[ "$bold" == "true" ]]; then
        format_string="${COLOR_BOLD}${color}"
    else
        format_string="${color}"
    fi
    
    printf "${format_string}%s${COLOR_RESET}\n" "$message"
}

#######################################
# Log messages with timestamp, severity level, and structured format
# for audit trail and debugging support
# Globals:
#   LOG_FILE, LOG_DIR
# Arguments:
#   $1 - level: Log level (INFO, WARN, ERROR, DEBUG)
#   $2 - message: Log message content
#   $3 - component: Component name (optional)
# Returns:
#   None
#######################################
log_message() {
    local level="$1"
    local message="$2"
    local component="${3:-CACHE_CLEANUP}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local formatted_message="[$timestamp] [$level] [$component] $message"
    
    # Ensure log directory exists
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null
        echo "$formatted_message" >> "$LOG_FILE"
    fi
    
    # Display on console with color coding based on level
    case "$level" in
        "ERROR")
            print_colored "$formatted_message" "$COLOR_RED"
            ;;
        "WARN")
            print_colored "$formatted_message" "$COLOR_YELLOW"
            ;;
        "INFO")
            if [[ "$VERBOSE_MODE" == "true" ]]; then
                print_colored "$formatted_message" "$COLOR_BLUE"
            fi
            ;;
        "DEBUG")
            if [[ "$VERBOSE_MODE" == "true" ]]; then
                print_colored "$formatted_message" "$COLOR_CYAN"
            fi
            ;;
        *)
            echo "$formatted_message"
            ;;
    esac
}

#######################################
# Validate script execution environment including Python availability,
# module imports, cache directories, and system resources
# Globals:
#   PYTHON_EXECUTABLE, BACKEND_MODULE, CACHE_BASE_DIR, CONFIG_DIR
# Arguments:
#   None
# Returns:
#   0 for success, non-zero for validation failures
#######################################
validate_environment() {
    local validation_errors=0
    
    log_message "INFO" "Starting environment validation"
    
    # Check Python 3.9+ availability and version compatibility
    if ! command -v "$PYTHON_EXECUTABLE" &> /dev/null; then
        log_message "ERROR" "Python executable not found: $PYTHON_EXECUTABLE"
        ((validation_errors++))
    else
        local python_version
        python_version=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$python_version >= 3.9" | bc -l 2>/dev/null || echo "0") -eq 0 ]]; then
            log_message "ERROR" "Python version $python_version < 3.9 required"
            ((validation_errors++))
        else
            log_message "INFO" "Python version validated: $python_version"
        fi
    fi
    
    # Validate backend module can be imported
    if ! $PYTHON_EXECUTABLE -c "import $BACKEND_MODULE" 2>/dev/null; then
        log_message "ERROR" "Cannot import backend module: $BACKEND_MODULE"
        log_message "INFO" "Ensure PYTHONPATH includes project root directory"
        ((validation_errors++))
    else
        log_message "INFO" "Backend module import validated"
    fi
    
    # Verify cache directories exist and are accessible
    if [[ ! -d "$CACHE_BASE_DIR" ]]; then
        log_message "WARN" "Cache base directory does not exist: $CACHE_BASE_DIR"
        if ! mkdir -p "$CACHE_BASE_DIR" 2>/dev/null; then
            log_message "ERROR" "Cannot create cache directory: $CACHE_BASE_DIR"
            ((validation_errors++))
        else
            log_message "INFO" "Created cache base directory"
        fi
    else
        log_message "INFO" "Cache directory validated: $CACHE_BASE_DIR"
    fi
    
    # Check cache directory permissions
    if [[ ! -w "$CACHE_BASE_DIR" ]]; then
        log_message "ERROR" "Cache directory not writable: $CACHE_BASE_DIR"
        ((validation_errors++))
    fi
    
    # Validate configuration files are present and readable
    if [[ -d "$CONFIG_DIR" ]] && [[ -r "$CONFIG_DIR" ]]; then
        log_message "INFO" "Configuration directory validated: $CONFIG_DIR"
    else
        log_message "WARN" "Configuration directory not accessible: $CONFIG_DIR"
    fi
    
    # Check available disk space (minimum 1GB recommended)
    local available_space
    available_space=$(df "$CACHE_BASE_DIR" | awk 'NR==2 {print $4}' 2>/dev/null || echo "0")
    if [[ $available_space -lt 1048576 ]]; then  # 1GB in KB
        log_message "WARN" "Low disk space available: ${available_space}KB"
    else
        log_message "INFO" "Sufficient disk space available: ${available_space}KB"
    fi
    
    # Test cache manager initialization
    local test_result
    test_result=$($PYTHON_EXECUTABLE -c "
from $BACKEND_MODULE.cache_manager import CacheManager
try:
    cm = CacheManager()
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR: Import failed")
    
    if [[ "$test_result" != "SUCCESS" ]]; then
        log_message "ERROR" "Cache manager initialization failed: $test_result"
        ((validation_errors++))
    else
        log_message "INFO" "Cache manager initialization validated"
    fi
    
    # Report validation summary
    if [[ $validation_errors -eq 0 ]]; then
        print_colored "✓ Environment validation completed successfully" "$COLOR_GREEN" "true"
        log_message "INFO" "Environment validation passed"
        return 0
    else
        print_colored "✗ Environment validation failed with $validation_errors errors" "$COLOR_RED" "true"
        log_message "ERROR" "Environment validation failed: $validation_errors errors"
        return 1
    fi
}

#######################################
# Parse and validate command-line arguments with comprehensive error
# checking and default value assignment
# Globals:
#   All configuration variables
# Arguments:
#   $@ - Command line arguments array
# Returns:
#   0 for valid arguments, non-zero for parsing errors
#######################################
parse_arguments() {
    local args=("$@")
    
    # Initialize with default values
    CLEANUP_MODE="$DEFAULT_CLEANUP_MODE"
    PRESERVE_HOT_DATA="$DEFAULT_PRESERVE_HOT_DATA"
    FORCE_CLEANUP="$DEFAULT_FORCE_CLEANUP"
    TARGET_UTILIZATION="$DEFAULT_TARGET_UTILIZATION"
    OPTIMIZE_AFTER_CLEANUP="$DEFAULT_OPTIMIZE_AFTER_CLEANUP"
    
    # Parse command-line options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                if [[ -z "$2" ]]; then
                    log_message "ERROR" "Mode option requires an argument"
                    return $EXIT_INVALID_ARGS
                fi
                case "$2" in
                    standard|aggressive|conservative)
                        CLEANUP_MODE="$2"
                        ;;
                    *)
                        log_message "ERROR" "Invalid cleanup mode: $2"
                        return $EXIT_INVALID_ARGS
                        ;;
                esac
                shift 2
                ;;
            -t|--target)
                if [[ -z "$2" ]]; then
                    log_message "ERROR" "Target option requires an argument"
                    return $EXIT_INVALID_ARGS
                fi
                if [[ $(echo "$2 >= 0.0 && $2 <= 1.0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
                    TARGET_UTILIZATION="$2"
                else
                    log_message "ERROR" "Target utilization must be between 0.0 and 1.0: $2"
                    return $EXIT_INVALID_ARGS
                fi
                shift 2
                ;;
            -p|--preserve-hot)
                PRESERVE_HOT_DATA=true
                shift
                ;;
            --no-preserve-hot)
                PRESERVE_HOT_DATA=false
                shift
                ;;
            -f|--force)
                FORCE_CLEANUP=true
                shift
                ;;
            -o|--optimize)
                OPTIMIZE_AFTER_CLEANUP=true
                shift
                ;;
            --no-optimize)
                OPTIMIZE_AFTER_CLEANUP=false
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE_MODE=true
                shift
                ;;
            -h|--help)
                print_usage
                exit $EXIT_SUCCESS
                ;;
            *)
                log_message "ERROR" "Unknown option: $1"
                print_usage
                return $EXIT_INVALID_ARGS
                ;;
        esac
    done
    
    # Validate argument combinations
    if [[ "$CLEANUP_MODE" == "conservative" && "$TARGET_UTILIZATION" < 0.8 ]]; then
        log_message "WARN" "Conservative mode with low target utilization may be ineffective"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_message "INFO" "Dry run mode enabled - no actual cleanup will be performed"
        FORCE_CLEANUP=true  # Skip confirmations in dry run
    fi
    
    log_message "INFO" "Arguments parsed successfully"
    return 0
}

#######################################
# Check current cache system status including utilization, performance
# metrics, and health indicators before cleanup operations
# Globals:
#   LEVEL1_STATUS, LEVEL2_STATUS, LEVEL3_STATUS
# Arguments:
#   None
# Returns:
#   0 for healthy cache, non-zero for issues detected
#######################################
check_cache_status() {
    local status_issues=0
    
    print_colored "Analyzing cache system status..." "$COLOR_BLUE" "true"
    log_message "INFO" "Starting cache status analysis"
    
    # Initialize Python cache manager interface and collect statistics
    local cache_stats
    cache_stats=$($PYTHON_EXECUTABLE -c "
import sys
import json
from $BACKEND_MODULE.cache_manager import CacheManager

try:
    cm = CacheManager()
    stats = cm.get_comprehensive_stats()
    
    # Format statistics for shell processing
    status = {
        'level1': {
            'utilization': stats.get('level1_utilization', 0.0),
            'fragmentation': stats.get('level1_fragmentation', 0.0),
            'hit_rate': stats.get('level1_hit_rate', 0.0),
            'size_mb': stats.get('level1_size_mb', 0)
        },
        'level2': {
            'utilization': stats.get('level2_utilization', 0.0),
            'file_count': stats.get('level2_file_count', 0),
            'size_gb': stats.get('level2_size_gb', 0.0),
            'hit_rate': stats.get('level2_hit_rate', 0.0)
        },
        'level3': {
            'utilization': stats.get('level3_utilization', 0.0),
            'result_count': stats.get('level3_result_count', 0),
            'size_gb': stats.get('level3_size_gb', 0.0),
            'hit_rate': stats.get('level3_hit_rate', 0.0)
        },
        'overall': {
            'health_score': stats.get('overall_health', 1.0),
            'performance_index': stats.get('performance_index', 1.0)
        }
    }
    
    print(json.dumps(status))
    
except Exception as e:
    print(json.dumps({'error': str(e)}))
    sys.exit(1)
" 2>/dev/null)
    
    if [[ $? -ne 0 ]] || [[ -z "$cache_stats" ]]; then
        log_message "ERROR" "Failed to retrieve cache statistics"
        return 1
    fi
    
    # Parse JSON statistics (simplified shell parsing)
    local l1_util l1_frag l1_hit l2_util l2_hit l3_util l3_hit health_score
    
    # Extract key metrics using basic string parsing
    l1_util=$(echo "$cache_stats" | grep -o '"level1_utilization":[0-9.]*' | cut -d: -f2 || echo "0.0")
    l1_frag=$(echo "$cache_stats" | grep -o '"level1_fragmentation":[0-9.]*' | cut -d: -f2 || echo "0.0")
    l1_hit=$(echo "$cache_stats" | grep -o '"level1_hit_rate":[0-9.]*' | cut -d: -f2 || echo "0.0")
    l2_hit=$(echo "$cache_stats" | grep -o '"level2_hit_rate":[0-9.]*' | cut -d: -f2 || echo "0.0")
    l3_hit=$(echo "$cache_stats" | grep -o '"level3_hit_rate":[0-9.]*' | cut -d: -f2 || echo "0.0")
    health_score=$(echo "$cache_stats" | grep -o '"health_score":[0-9.]*' | cut -d: -f2 || echo "1.0")
    
    # Display cache status summary
    print_colored "═══ Cache System Status Report ═══" "$COLOR_CYAN" "true"
    
    printf "${COLOR_BOLD}Level 1 (Memory Cache):${COLOR_RESET}\n"
    printf "  Utilization: %.1f%% " "$(echo "$l1_util * 100" | bc -l 2>/dev/null || echo "0")"
    if [[ $(echo "$l1_util > 0.9" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(High)${COLOR_RESET}\n"
        ((status_issues++))
    elif [[ $(echo "$l1_util > 0.7" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_YELLOW}(Moderate)${COLOR_RESET}\n"
    else
        printf "${COLOR_GREEN}(Normal)${COLOR_RESET}\n"
    fi
    
    printf "  Fragmentation: %.1f%% " "$(echo "$l1_frag * 100" | bc -l 2>/dev/null || echo "0")"
    if [[ $(echo "$l1_frag > 0.1" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(High)${COLOR_RESET}\n"
        ((status_issues++))
    else
        printf "${COLOR_GREEN}(Normal)${COLOR_RESET}\n"
    fi
    
    printf "  Hit Rate: %.1f%% " "$(echo "$l1_hit * 100" | bc -l 2>/dev/null || echo "0")"
    if [[ $(echo "$l1_hit < 0.8" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(Below Threshold)${COLOR_RESET}\n"
        ((status_issues++))
    else
        printf "${COLOR_GREEN}(Healthy)${COLOR_RESET}\n"
    fi
    
    printf "\n${COLOR_BOLD}Level 2 (Disk Cache):${COLOR_RESET}\n"
    printf "  Hit Rate: %.1f%% " "$(echo "$l2_hit * 100" | bc -l 2>/dev/null || echo "0")"
    if [[ $(echo "$l2_hit < 0.8" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(Below Threshold)${COLOR_RESET}\n"
        ((status_issues++))
    else
        printf "${COLOR_GREEN}(Healthy)${COLOR_RESET}\n"
    fi
    
    printf "\n${COLOR_BOLD}Level 3 (Result Cache):${COLOR_RESET}\n"
    printf "  Hit Rate: %.1f%% " "$(echo "$l3_hit * 100" | bc -l 2>/dev/null || echo "0")"
    if [[ $(echo "$l3_hit < 0.8" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(Below Threshold)${COLOR_RESET}\n"
        ((status_issues++))
    else
        printf "${COLOR_GREEN}(Healthy)${COLOR_RESET}\n"
    fi
    
    printf "\n${COLOR_BOLD}Overall Health Score:${COLOR_RESET} %.2f " "$health_score"
    if [[ $(echo "$health_score < 0.7" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_RED}(Poor)${COLOR_RESET}\n"
        ((status_issues++))
    elif [[ $(echo "$health_score < 0.9" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        printf "${COLOR_YELLOW}(Fair)${COLOR_RESET}\n"
    else
        printf "${COLOR_GREEN}(Excellent)${COLOR_RESET}\n"
    fi
    
    # Store status for cleanup operations
    LEVEL1_STATUS="$l1_util,$l1_frag,$l1_hit"
    LEVEL2_STATUS="$l2_hit"
    LEVEL3_STATUS="$l3_hit"
    
    log_message "INFO" "Cache status analysis completed: $status_issues issues detected"
    
    if [[ $status_issues -gt 0 ]]; then
        print_colored "⚠ Cache system requires cleanup and optimization" "$COLOR_YELLOW" "true"
        return 1
    else
        print_colored "✓ Cache system is operating within optimal parameters" "$COLOR_GREEN" "true"
        return 0
    fi
}

#######################################
# Execute comprehensive cache cleanup operations across all cache levels
# with progress monitoring and error handling
# Globals:
#   DRY_RUN, CLEANUP_MODE, TARGET_UTILIZATION, PRESERVE_HOT_DATA
# Arguments:
#   None
# Returns:
#   0 for successful cleanup, non-zero for cleanup failures
#######################################
execute_cache_cleanup() {
    local cleanup_errors=0
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_colored "=== DRY RUN: Cache Cleanup Preview ===" "$COLOR_CYAN" "true"
        log_message "INFO" "Starting dry run cache cleanup preview"
    else
        print_colored "=== Executing Cache Cleanup Operations ===" "$COLOR_BLUE" "true"
        log_message "INFO" "Starting cache cleanup execution"
    fi
    
    # Display cleanup configuration
    printf "\n${COLOR_BOLD}Cleanup Configuration:${COLOR_RESET}\n"
    printf "  Mode: ${COLOR_CYAN}%s${COLOR_RESET}\n" "$CLEANUP_MODE"
    printf "  Target Utilization: ${COLOR_CYAN}%.1f%%${COLOR_RESET}\n" "$(echo "$TARGET_UTILIZATION * 100" | bc -l)"
    printf "  Preserve Hot Data: ${COLOR_CYAN}%s${COLOR_RESET}\n" "$PRESERVE_HOT_DATA"
    printf "  Optimize After Cleanup: ${COLOR_CYAN}%s${COLOR_RESET}\n" "$OPTIMIZE_AFTER_CLEANUP"
    printf "\n"
    
    # Execute cleanup through Python cache manager
    local cleanup_result
    cleanup_result=$($PYTHON_EXECUTABLE -c "
import sys
import json
from $BACKEND_MODULE.cache_manager import CacheManager

try:
    cm = CacheManager()
    
    # Configure cleanup parameters
    config = {
        'mode': '$CLEANUP_MODE',
        'target_utilization': float('$TARGET_UTILIZATION'),
        'preserve_hot_data': $([[ '$PRESERVE_HOT_DATA' == 'true' ]] && echo 'True' || echo 'False'),
        'dry_run': $([[ '$DRY_RUN' == 'true' ]] && echo 'True' || echo 'False'),
        'optimize_after': $([[ '$OPTIMIZE_AFTER_CLEANUP' == 'true' ]] && echo 'True' || echo 'False')
    }
    
    # Execute cleanup with progress callback
    def progress_callback(operation, total, completed, details=None):
        progress_data = {
            'operation': operation,
            'total': total,
            'completed': completed,
            'details': details or {}
        }
        print(f'PROGRESS:{json.dumps(progress_data)}', flush=True)
    
    result = cm.execute_comprehensive_cleanup(config, progress_callback)
    
    # Format results for shell processing
    cleanup_summary = {
        'success': result.get('success', False),
        'space_freed_mb': result.get('space_freed_mb', 0),
        'files_cleaned': result.get('files_cleaned', 0),
        'performance_improvement': result.get('performance_improvement', 0.0),
        'cleanup_duration': result.get('cleanup_duration', 0.0),
        'level1_cleaned': result.get('level1_cleaned', 0),
        'level2_cleaned': result.get('level2_cleaned', 0),
        'level3_cleaned': result.get('level3_cleaned', 0),
        'errors': result.get('errors', [])
    }
    
    print(f'RESULT:{json.dumps(cleanup_summary)}')
    
except Exception as e:
    error_data = {'success': False, 'error': str(e)}
    print(f'RESULT:{json.dumps(error_data)}')
    sys.exit(1)
" 2>&1)
    
    # Process cleanup output and progress updates
    local final_result=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^PROGRESS: ]]; then
            local progress_json="${line#PROGRESS:}"
            # Parse progress and display
            local operation=$(echo "$progress_json" | grep -o '"operation":"[^"]*"' | cut -d'"' -f4)
            local total=$(echo "$progress_json" | grep -o '"total":[0-9]*' | cut -d: -f2)
            local completed=$(echo "$progress_json" | grep -o '"completed":[0-9]*' | cut -d: -f2)
            
            if [[ -n "$operation" && -n "$total" && -n "$completed" ]]; then
                monitor_cleanup_progress "$operation" "$total" "$completed"
            fi
        elif [[ "$line" =~ ^RESULT: ]]; then
            final_result="${line#RESULT:}"
        else
            # Regular output from Python script
            if [[ "$VERBOSE_MODE" == "true" ]]; then
                echo "$line"
            fi
        fi
    done <<< "$cleanup_result"
    
    # Process final cleanup results
    if [[ -z "$final_result" ]]; then
        log_message "ERROR" "No cleanup result received from cache manager"
        return 1
    fi
    
    # Parse cleanup results
    local success=$(echo "$final_result" | grep -o '"success":[^,}]*' | cut -d: -f2 | tr -d ' ')
    local space_freed=$(echo "$final_result" | grep -o '"space_freed_mb":[0-9.]*' | cut -d: -f2)
    local files_cleaned=$(echo "$final_result" | grep -o '"files_cleaned":[0-9]*' | cut -d: -f2)
    local perf_improvement=$(echo "$final_result" | grep -o '"performance_improvement":[0-9.]*' | cut -d: -f2)
    
    TOTAL_SPACE_FREED="${space_freed:-0}"
    PERFORMANCE_IMPROVEMENT="${perf_improvement:-0}"
    
    # Display cleanup summary
    printf "\n${COLOR_BOLD}═══ Cleanup Summary ═══${COLOR_RESET}\n"
    
    if [[ "$success" == "true" ]]; then
        print_colored "✓ Cache cleanup completed successfully" "$COLOR_GREEN" "true"
        printf "  Space Freed: ${COLOR_CYAN}%.1f MB${COLOR_RESET}\n" "${TOTAL_SPACE_FREED}"
        printf "  Files Processed: ${COLOR_CYAN}%s${COLOR_RESET}\n" "${files_cleaned:-0}"
        printf "  Performance Improvement: ${COLOR_CYAN}%.1f%%${COLOR_RESET}\n" "$(echo "${PERFORMANCE_IMPROVEMENT} * 100" | bc -l 2>/dev/null || echo "0")"
        log_message "INFO" "Cache cleanup completed successfully"
        return 0
    else
        print_colored "✗ Cache cleanup encountered errors" "$COLOR_RED" "true"
        local error_msg=$(echo "$final_result" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
        if [[ -n "$error_msg" ]]; then
            log_message "ERROR" "Cleanup failed: $error_msg"
        fi
        return 1
    fi
}

#######################################
# Monitor cache cleanup progress with real-time updates, performance metrics,
# and estimated completion time
# Globals:
#   PROGRESS_UPDATE_INTERVAL
# Arguments:
#   $1 - operation_type: Type of operation being performed
#   $2 - total_items: Total number of items to process
#   $3 - completed_items: Number of items completed
# Returns:
#   None
#######################################
monitor_cleanup_progress() {
    local operation_type="$1"
    local total_items="$2"
    local completed_items="$3"
    
    # Calculate progress metrics
    local progress_percent=0
    if [[ $total_items -gt 0 ]]; then
        progress_percent=$(echo "scale=1; $completed_items * 100 / $total_items" | bc -l 2>/dev/null || echo "0")
    fi
    
    # Determine terminal width for responsive formatting
    local term_width
    term_width=$(tput cols 2>/dev/null || echo "80")
    local bar_width=$((term_width - 40))  # Reserve space for text
    if [[ $bar_width -lt 20 ]]; then
        bar_width=20
    fi
    
    # Generate ASCII progress bar
    local filled_width=$((bar_width * completed_items / total_items))
    if [[ $total_items -eq 0 ]]; then
        filled_width=0
    fi
    
    local progress_bar=""
    for ((i=0; i<filled_width; i++)); do
        progress_bar+="█"
    done
    for ((i=filled_width; i<bar_width; i++)); do
        progress_bar+="░"
    done
    
    # Color coding for different progress states
    local color="$COLOR_BLUE"
    if [[ $(echo "$progress_percent >= 100" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        color="$COLOR_GREEN"
    elif [[ $(echo "$progress_percent >= 75" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        color="$COLOR_CYAN"
    elif [[ $(echo "$progress_percent >= 25" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        color="$COLOR_YELLOW"
    fi
    
    # Display progress without scrolling (carriage return)
    printf "\r${COLOR_BOLD}%s:${COLOR_RESET} [${color}%s${COLOR_RESET}] %6.1f%% (%d/%d)" \
        "$operation_type" "$progress_bar" "$progress_percent" "$completed_items" "$total_items"
    
    # Add newline when complete
    if [[ $completed_items -eq $total_items ]]; then
        printf "\n"
    fi
}

#######################################
# Generate comprehensive cleanup report with statistics, performance impact,
# and recommendations for future maintenance
# Globals:
#   TOTAL_SPACE_FREED, PERFORMANCE_IMPROVEMENT, START_TIME
# Arguments:
#   None
# Returns:
#   0 for successful report generation, non-zero for errors
#######################################
generate_cleanup_report() {
    local report_errors=0
    local end_time=$(date '+%s')
    local duration=$((end_time - START_TIME))
    
    print_colored "Generating comprehensive cleanup report..." "$COLOR_BLUE"
    log_message "INFO" "Starting cleanup report generation"
    
    # Get post-cleanup cache statistics
    local post_cleanup_stats
    post_cleanup_stats=$($PYTHON_EXECUTABLE -c "
import json
from $BACKEND_MODULE.cache_manager import CacheManager

try:
    cm = CacheManager()
    stats = cm.get_comprehensive_stats()
    recommendations = cm.generate_maintenance_recommendations()
    
    report_data = {
        'post_cleanup_stats': stats,
        'recommendations': recommendations,
        'optimization_suggestions': cm.get_optimization_suggestions()
    }
    
    print(json.dumps(report_data))
    
except Exception as e:
    print(json.dumps({'error': str(e)}))
" 2>/dev/null)
    
    if [[ $? -ne 0 ]] || [[ -z "$post_cleanup_stats" ]]; then
        log_message "ERROR" "Failed to generate post-cleanup statistics"
        ((report_errors++))
    fi
    
    # Create report timestamp
    local report_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="${LOG_DIR}/cleanup_report_$(date '+%Y%m%d_%H%M%S').txt"
    
    # Ensure log directory exists
    mkdir -p "$LOG_DIR" 2>/dev/null
    
    # Generate comprehensive report
    cat > "$report_file" << EOF
═══════════════════════════════════════════════════════════════════
              PLUME SIMULATION CACHE CLEANUP REPORT
═══════════════════════════════════════════════════════════════════

Report Generated: $report_timestamp
Script Version: $SCRIPT_VERSION
Cleanup Duration: ${duration}s

CLEANUP CONFIGURATION:
  Mode: $CLEANUP_MODE
  Target Utilization: $(echo "$TARGET_UTILIZATION * 100" | bc -l)%
  Preserve Hot Data: $PRESERVE_HOT_DATA
  Force Cleanup: $FORCE_CLEANUP
  Optimize After Cleanup: $OPTIMIZE_AFTER_CLEANUP

CLEANUP RESULTS:
  Total Space Freed: ${TOTAL_SPACE_FREED} MB
  Performance Improvement: $(echo "$PERFORMANCE_IMPROVEMENT * 100" | bc -l)%
  Operation Status: $(if [[ $report_errors -eq 0 ]]; then echo "SUCCESS"; else echo "PARTIAL"; fi)

EOF
    
    # Add detailed statistics if available
    if [[ -n "$post_cleanup_stats" ]] && [[ "$post_cleanup_stats" != *"error"* ]]; then
        echo "POST-CLEANUP CACHE STATISTICS:" >> "$report_file"
        echo "$post_cleanup_stats" | python3 -m json.tool >> "$report_file" 2>/dev/null || echo "Statistics formatting failed" >> "$report_file"
    fi
    
    # Display report summary
    printf "\n${COLOR_BOLD}═══ Cleanup Report Summary ═══${COLOR_RESET}\n"
    printf "  Duration: ${COLOR_CYAN}%d seconds${COLOR_RESET}\n" "$duration"
    printf "  Space Freed: ${COLOR_GREEN}%.1f MB${COLOR_RESET}\n" "$TOTAL_SPACE_FREED"
    printf "  Performance Gain: ${COLOR_GREEN}%.1f%%${COLOR_RESET}\n" "$(echo "$PERFORMANCE_IMPROVEMENT * 100" | bc -l 2>/dev/null || echo "0")"
    printf "  Report Saved: ${COLOR_CYAN}%s${COLOR_RESET}\n" "$report_file"
    
    # Provide maintenance recommendations
    printf "\n${COLOR_BOLD}Maintenance Recommendations:${COLOR_RESET}\n"
    
    if [[ $(echo "$PERFORMANCE_IMPROVEMENT > 0.1" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        print_colored "• Regular cleanup significantly improves performance" "$COLOR_GREEN"
        print_colored "• Consider scheduling automated cleanup operations" "$COLOR_BLUE"
    fi
    
    if [[ $(echo "$TOTAL_SPACE_FREED > 1000" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        print_colored "• Substantial space was reclaimed - monitor cache growth" "$COLOR_YELLOW"
        print_colored "• Review cache retention policies for optimization" "$COLOR_BLUE"
    fi
    
    print_colored "• Next recommended cleanup: 1-2 weeks" "$COLOR_CYAN"
    print_colored "• Monitor cache hit rates and adjust policies as needed" "$COLOR_CYAN"
    
    log_message "INFO" "Cleanup report generated successfully: $report_file"
    
    if [[ $report_errors -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

#######################################
# Handle cleanup errors with classification, recovery attempts, and detailed
# error reporting for troubleshooting
# Globals:
#   LOG_FILE
# Arguments:
#   $1 - error_type: Classification of error (validation, execution, timeout)
#   $2 - error_message: Detailed error description
#   $3 - component: Component where error occurred
# Returns:
#   0 for recoverable error, non-zero for critical failure
#######################################
handle_cleanup_error() {
    local error_type="$1"
    local error_message="$2"
    local component="${3:-UNKNOWN}"
    local recovery_attempted=false
    
    log_message "ERROR" "[$error_type] $error_message" "$component"
    
    # Classify error severity and determine recovery strategy
    case "$error_type" in
        "validation")
            print_colored "✗ Validation Error: $error_message" "$COLOR_RED" "true"
            print_colored "Recommendation: Check system requirements and configuration" "$COLOR_YELLOW"
            return 1  # Critical - cannot continue
            ;;
        "permission")
            print_colored "✗ Permission Error: $error_message" "$COLOR_RED" "true"
            print_colored "Recommendation: Check file/directory permissions" "$COLOR_YELLOW"
            print_colored "  sudo chown -R \$USER: $CACHE_BASE_DIR" "$COLOR_CYAN"
            return 1  # Critical - cannot continue
            ;;
        "timeout")
            print_colored "⚠ Operation Timeout: $error_message" "$COLOR_YELLOW" "true"
            print_colored "Recommendation: Retry with aggressive cleanup mode" "$COLOR_BLUE"
            return 0  # Recoverable
            ;;
        "resource")
            print_colored "⚠ Resource Error: $error_message" "$COLOR_YELLOW" "true"
            print_colored "Recommendation: Free system resources and retry" "$COLOR_BLUE"
            
            # Attempt basic resource cleanup
            if command -v sync &> /dev/null; then
                sync
                recovery_attempted=true
                log_message "INFO" "Attempted filesystem sync for recovery"
            fi
            return 0  # Recoverable
            ;;
        "cache_corruption")
            print_colored "⚠ Cache Integrity Error: $error_message" "$COLOR_YELLOW" "true"
            print_colored "Recommendation: Force rebuild corrupted cache sections" "$COLOR_BLUE"
            return 0  # Recoverable with force cleanup
            ;;
        *)
            print_colored "✗ Unknown Error: $error_message" "$COLOR_RED" "true"
            print_colored "Recommendation: Review logs and contact support" "$COLOR_YELLOW"
            return 1  # Unknown - treat as critical
            ;;
    esac
}

#######################################
# Clean up script execution resources including temporary files, Python
# processes, and lock files before exit
# Globals:
#   SCRIPT_PID, LOG_FILE
# Arguments:
#   None
# Returns:
#   None
#######################################
cleanup_script_resources() {
    log_message "INFO" "Starting script resource cleanup"
    
    # Terminate any running Python cache manager processes spawned by this script
    local python_pids
    python_pids=$(pgrep -f "$BACKEND_MODULE.cache_manager" 2>/dev/null || true)
    if [[ -n "$python_pids" ]]; then
        for pid in $python_pids; do
            if [[ "$pid" != "$SCRIPT_PID" ]]; then
                log_message "INFO" "Terminating Python process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Remove temporary files created during execution
    local temp_files=(
        "/tmp/cache_cleanup_$$_*"
        "/tmp/plume_sim_cleanup_*"
        "${CACHE_BASE_DIR}/.cleanup_lock"
    )
    
    for pattern in "${temp_files[@]}"; do
        if compgen -G "$pattern" > /dev/null 2>&1; then
            rm -f $pattern 2>/dev/null || true
            log_message "DEBUG" "Removed temporary files: $pattern"
        fi
    done
    
    # Release file locks and cleanup lock files
    if [[ -f "${CACHE_BASE_DIR}/.cleanup_lock" ]]; then
        rm -f "${CACHE_BASE_DIR}/.cleanup_lock" 2>/dev/null || true
        log_message "DEBUG" "Released cleanup lock file"
    fi
    
    # Close log files and flush pending writes
    if [[ -n "$LOG_FILE" ]] && [[ -f "$LOG_FILE" ]]; then
        sync 2>/dev/null || true
        log_message "INFO" "Script cleanup completed successfully"
    fi
    
    # Reset terminal color settings
    printf "${COLOR_RESET}"
    
    # Clear environment variables set by script
    unset CLEANUP_MODE PRESERVE_HOT_DATA FORCE_CLEANUP TARGET_UTILIZATION
    unset OPTIMIZE_AFTER_CLEANUP VERBOSE_MODE DRY_RUN
    unset LEVEL1_STATUS LEVEL2_STATUS LEVEL3_STATUS
    unset TOTAL_SPACE_FREED PERFORMANCE_IMPROVEMENT
}

#######################################
# Handle interrupt signals (SIGINT, SIGTERM) with graceful cleanup and
# proper exit status reporting
# Globals:
#   All script variables
# Arguments:
#   $1 - signal_number: Signal number received
# Returns:
#   None (exits script)
#######################################
signal_handler() {
    local signal_number="$1"
    local signal_name=""
    
    case "$signal_number" in
        2) signal_name="SIGINT" ;;
        15) signal_name="SIGTERM" ;;
        *) signal_name="SIG$signal_number" ;;
    esac
    
    log_message "WARN" "Received signal $signal_name ($signal_number) - initiating graceful shutdown"
    print_colored "\n⚠ Cleanup operation interrupted by signal $signal_name" "$COLOR_YELLOW" "true"
    
    # Display interruption message to user
    print_colored "Performing graceful cleanup before exit..." "$COLOR_BLUE"
    
    # Initiate graceful cleanup of cache operations
    if command -v $PYTHON_EXECUTABLE &> /dev/null; then
        $PYTHON_EXECUTABLE -c "
try:
    from $BACKEND_MODULE.cache_manager import CacheManager
    cm = CacheManager()
    cm.emergency_stop()
    print('Cache operations stopped gracefully')
except:
    pass
" 2>/dev/null || true
    fi
    
    # Generate partial cleanup report if possible
    if [[ -n "$START_TIME" ]]; then
        print_colored "Generating partial cleanup report..." "$COLOR_CYAN"
        generate_cleanup_report || true
    fi
    
    # Clean up script resources
    cleanup_script_resources
    
    # Exit with signal-based exit code
    local exit_code=$((128 + signal_number))
    print_colored "Script terminated by signal $signal_name (exit code: $exit_code)" "$COLOR_RED"
    exit $exit_code
}

#######################################
# Main script execution function orchestrating cache cleanup workflow with
# comprehensive error handling and progress reporting
# Globals:
#   All script configuration and runtime variables
# Arguments:
#   $@ - script_arguments: Command line arguments
# Returns:
#   Exit code indicating script execution status
#######################################
main() {
    local script_arguments=("$@")
    START_TIME=$(date '+%s')
    
    # Setup signal handlers for graceful interruption
    trap 'signal_handler 2' SIGINT
    trap 'signal_handler 15' SIGTERM
    
    # Initialize logging
    LOG_FILE="${LOG_DIR}/cache_cleanup_$(date '+%Y%m%d_%H%M%S').log"
    mkdir -p "$LOG_DIR" 2>/dev/null
    
    # Display script header
    print_colored "╔══════════════════════════════════════════════════════════════════╗" "$COLOR_BLUE" "true"
    print_colored "║           PLUME SIMULATION CACHE CLEANUP UTILITY v$SCRIPT_VERSION           ║" "$COLOR_BLUE" "true"
    print_colored "║      Multi-Level Cache Maintenance and Performance Optimization     ║" "$COLOR_BLUE" "true"
    print_colored "╚══════════════════════════════════════════════════════════════════╝" "$COLOR_BLUE" "true"
    printf "\n"
    
    log_message "INFO" "Starting cache cleanup script v$SCRIPT_VERSION"
    log_message "INFO" "Command line: $0 ${script_arguments[*]}"
    
    # Parse and validate command-line arguments
    if ! parse_arguments "${script_arguments[@]}"; then
        handle_cleanup_error "validation" "Invalid command line arguments" "ARGUMENT_PARSER"
        cleanup_script_resources
        exit $EXIT_INVALID_ARGS
    fi
    
    # Validate execution environment and dependencies
    print_colored "Validating execution environment..." "$COLOR_BLUE"
    if ! validate_environment; then
        handle_cleanup_error "validation" "Environment validation failed" "ENVIRONMENT"
        cleanup_script_resources
        exit $EXIT_ERROR
    fi
    
    # Check current cache status and health
    if ! check_cache_status; then
        log_message "WARN" "Cache status check indicates issues requiring cleanup"
    fi
    
    # Request user confirmation unless forced or dry run
    if [[ "$FORCE_CLEANUP" != "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        printf "\n${COLOR_YELLOW}Proceed with cache cleanup? [y/N]: ${COLOR_RESET}"
        read -r user_confirmation
        if [[ ! "$user_confirmation" =~ ^[Yy]$ ]]; then
            print_colored "Cache cleanup cancelled by user" "$COLOR_CYAN"
            log_message "INFO" "User cancelled cleanup operation"
            cleanup_script_resources
            exit $EXIT_SUCCESS
        fi
    fi
    
    # Execute cache cleanup operations with monitoring
    if ! execute_cache_cleanup; then
        handle_cleanup_error "execution" "Cache cleanup operations failed" "CLEANUP_EXECUTOR"
        cleanup_script_resources
        exit $EXIT_ERROR
    fi
    
    # Generate comprehensive cleanup report
    print_colored "\nGenerating cleanup report and analysis..." "$COLOR_BLUE"
    if ! generate_cleanup_report; then
        handle_cleanup_error "execution" "Cleanup report generation failed" "REPORT_GENERATOR"
        # Continue - report failure is not critical
    fi
    
    # Display completion summary
    printf "\n${COLOR_BOLD}${COLOR_GREEN}╔═══════════════════════════════════════════════════════════════════╗${COLOR_RESET}\n"
    printf "${COLOR_BOLD}${COLOR_GREEN}║                    CLEANUP COMPLETED SUCCESSFULLY                    ║${COLOR_RESET}\n"
    printf "${COLOR_BOLD}${COLOR_GREEN}╚═══════════════════════════════════════════════════════════════════╝${COLOR_RESET}\n"
    
    local end_time=$(date '+%s')
    local total_duration=$((end_time - START_TIME))
    
    printf "\n${COLOR_BOLD}Final Summary:${COLOR_RESET}\n"
    printf "  Total Duration: ${COLOR_CYAN}%d seconds${COLOR_RESET}\n" "$total_duration"
    printf "  Space Reclaimed: ${COLOR_GREEN}%.1f MB${COLOR_RESET}\n" "$TOTAL_SPACE_FREED"
    printf "  Performance Improvement: ${COLOR_GREEN}%.1f%%${COLOR_RESET}\n" "$(echo "$PERFORMANCE_IMPROVEMENT * 100" | bc -l 2>/dev/null || echo "0")"
    printf "  Cache System Status: ${COLOR_GREEN}Optimized${COLOR_RESET}\n"
    printf "  Log File: ${COLOR_CYAN}%s${COLOR_RESET}\n" "$LOG_FILE"
    
    log_message "INFO" "Cache cleanup completed successfully in ${total_duration}s"
    
    # Clean up script resources and temporary files
    cleanup_script_resources
    
    # Exit with appropriate status code
    exit $EXIT_SUCCESS
}

# Execute main function if script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi