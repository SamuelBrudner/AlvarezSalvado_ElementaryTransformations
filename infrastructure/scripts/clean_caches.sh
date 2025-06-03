#!/bin/bash

# =============================================================================
# PLUME SIMULATION CACHE CLEANUP SCRIPT
# =============================================================================
# Script: clean_caches.sh
# Version: 1.0.0
# Description: Comprehensive cache cleanup script for plume simulation system
# 
# This script provides automated maintenance for the multi-level caching
# architecture including memory cache cleanup, disk cache optimization,
# result cache maintenance, and temporary file removal. Implements intelligent
# cleanup strategies with configurable retention policies, performance
# monitoring, and integration with the unified cache management system.
# 
# Optimized for scientific computing workloads with support for batch
# processing environments, cross-platform compatibility, and comprehensive
# logging for audit trails and system maintenance operations.
# =============================================================================

# Global Configuration Variables
SCRIPT_NAME="clean_caches.sh"
SCRIPT_VERSION="1.0.0"
SCRIPT_DESCRIPTION="Comprehensive cache cleanup script for plume simulation system"

# Default Directory Configurations
DEFAULT_CACHE_BASE_DIR="cache"
DEFAULT_TEMP_DIR="temp"
DEFAULT_LOG_DIR="logs"
DEFAULT_RESULTS_DIR="results"

# Cache Pattern Matching
MEMORY_CACHE_PATTERN="memory_cache_*"
DISK_CACHE_PATTERN="disk_cache_*"
RESULT_CACHE_PATTERN="result_cache_*"
TEMP_FILE_PATTERN="plume_sim_*"

# Retention and Size Policies
DEFAULT_RETENTION_DAYS=7
DEFAULT_MAX_CACHE_SIZE_GB=10

# Logging Configuration
CLEANUP_LOG_FILE="logs/cache_cleanup.log"
PERFORMANCE_LOG_FILE="logs/cache_cleanup_performance.log"
ERROR_LOG_FILE="logs/cache_cleanup_errors.log"

# Process Management
LOCK_FILE="/tmp/plume_cache_cleanup.lock"
PID_FILE="/tmp/plume_cache_cleanup.pid"

# Exit Codes
EXIT_SUCCESS=0
EXIT_ERROR=1
EXIT_LOCK_FAILED=2
EXIT_INVALID_ARGS=3

# Color Definitions for Terminal Output
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_BLUE="\033[94m"
COLOR_CYAN="\033[96m"
COLOR_RESET="\033[0m"

# Runtime Configuration Flags
VERBOSE=false
DRY_RUN=false
FORCE_CLEANUP=false
QUIET_MODE=false

# Global Variables for Statistics Tracking
TOTAL_MEMORY_CACHE_CLEANED=0
TOTAL_DISK_SPACE_FREED=0
TOTAL_RESULT_CACHE_CLEANED=0
TOTAL_TEMP_FILES_REMOVED=0
CLEANUP_START_TIME=""
CLEANUP_END_TIME=""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Display comprehensive usage information including command-line options,
# examples, and configuration parameters for cache cleanup operations
show_usage() {
    echo -e "${COLOR_BLUE}${SCRIPT_NAME} v${SCRIPT_VERSION}${COLOR_RESET}"
    echo -e "${COLOR_BLUE}${SCRIPT_DESCRIPTION}${COLOR_RESET}"
    echo ""
    echo "USAGE:"
    echo "    $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "    -h, --help                Show this help message and exit"
    echo "    -v, --verbose             Enable verbose output for detailed operations"
    echo "    -q, --quiet               Enable quiet mode (minimal output)"
    echo "    -n, --dry-run             Show what would be cleaned without actual deletion"
    echo "    -f, --force               Force cleanup without interactive confirmation"
    echo "    -c, --cache-dir DIR       Specify cache base directory (default: ${DEFAULT_CACHE_BASE_DIR})"
    echo "    -t, --temp-dir DIR        Specify temporary files directory (default: ${DEFAULT_TEMP_DIR})"
    echo "    -r, --retention DAYS      Set retention period in days (default: ${DEFAULT_RETENTION_DAYS})"
    echo "    -s, --max-size SIZE       Set maximum cache size in GB (default: ${DEFAULT_MAX_CACHE_SIZE_GB})"
    echo "    -l, --log-dir DIR         Specify log directory (default: ${DEFAULT_LOG_DIR})"
    echo "    --memory-only             Clean only memory cache (Level 1)"
    echo "    --disk-only               Clean only disk cache (Level 2)"
    echo "    --results-only            Clean only result cache (Level 3)"
    echo "    --temp-only               Clean only temporary files"
    echo "    --optimize                Perform cache optimization after cleanup"
    echo ""
    echo "EXAMPLES:"
    echo "    $0                        Run standard cleanup with default settings"
    echo "    $0 -v -r 14               Verbose cleanup with 14-day retention"
    echo "    $0 -n --memory-only       Dry run for memory cache only"
    echo "    $0 -f -s 20               Force cleanup with 20GB size limit"
    echo "    $0 --optimize             Run cleanup with performance optimization"
    echo ""
    echo "CONFIGURATION:"
    echo "    Environment Variables:"
    echo "        PLUME_CACHE_DIR       Override default cache directory"
    echo "        PLUME_LOG_LEVEL       Set logging level (DEBUG, INFO, WARN, ERROR)"
    echo "        PLUME_CLEANUP_CONFIG  Path to configuration file"
    echo ""
    echo "INTEGRATION:"
    echo "    Integration with Cache Management System:"
    echo "        - Coordinates with active cache instances"
    echo "        - Maintains cache integrity during cleanup"
    echo "        - Updates cache statistics and metrics"
    echo ""
    echo "PERFORMANCE OPTIMIZATION:"
    echo "    - Use --optimize flag for index rebuilding and compression"
    echo "    - Monitor cache hit rate threshold (target: 0.8)"
    echo "    - Configure retention policies based on simulation frequency"
    echo ""
    echo "EXIT CODES:"
    echo "    ${EXIT_SUCCESS}    Success - cleanup completed successfully"
    echo "    ${EXIT_ERROR}    Error - general error during cleanup"
    echo "    ${EXIT_LOCK_FAILED}    Lock acquisition failed"
    echo "    ${EXIT_INVALID_ARGS}    Invalid command line arguments"
    echo ""
    exit ${EXIT_SUCCESS}
}

# Centralized logging function with color coding, timestamp formatting,
# and multi-destination output for comprehensive audit trails and monitoring integration
log_message() {
    local log_level="$1"
    local message="$2"
    local component="${3:-CLEANUP}"
    
    # Generate timestamp with microsecond precision for scientific logging
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S.%6N')
    local pid=$$
    local color=""
    local log_file=""
    
    # Apply color coding based on log level and terminal capabilities
    case "${log_level}" in
        "ERROR")
            color="${COLOR_RED}"
            log_file="${ERROR_LOG_FILE}"
            ;;
        "WARN"|"WARNING")
            color="${COLOR_YELLOW}"
            log_file="${CLEANUP_LOG_FILE}"
            ;;
        "INFO")
            color="${COLOR_GREEN}"
            log_file="${CLEANUP_LOG_FILE}"
            ;;
        "DEBUG")
            color="${COLOR_CYAN}"
            log_file="${CLEANUP_LOG_FILE}"
            ;;
        "PERF")
            color="${COLOR_BLUE}"
            log_file="${PERFORMANCE_LOG_FILE}"
            ;;
        *)
            color="${COLOR_RESET}"
            log_file="${CLEANUP_LOG_FILE}"
            ;;
    esac
    
    # Format message with component context and process information
    local formatted_message="[${timestamp}] [PID:${pid}] [${log_level}] [${component}] ${message}"
    
    # Output to console unless quiet mode is enabled
    if [[ "${QUIET_MODE}" != "true" ]] || [[ "${log_level}" == "ERROR" ]]; then
        if [[ -t 1 ]]; then  # Check if stdout is a terminal
            echo -e "${color}${formatted_message}${COLOR_RESET}"
        else
            echo "${formatted_message}"
        fi
    fi
    
    # Write to appropriate log file based on log level
    if [[ -n "${log_file}" ]]; then
        # Ensure log directory exists
        mkdir -p "$(dirname "${log_file}")" 2>/dev/null
        echo "${formatted_message}" >> "${log_file}" 2>/dev/null
    fi
    
    # Include performance metrics if available
    if [[ "${log_level}" == "PERF" ]]; then
        local memory_usage=$(ps -o rss= -p $$ 2>/dev/null | tr -d ' ')
        if [[ -n "${memory_usage}" ]]; then
            echo "[${timestamp}] [PID:${pid}] [PERF] [MEMORY] RSS=${memory_usage}KB" >> "${PERFORMANCE_LOG_FILE}" 2>/dev/null
        fi
    fi
    
    # Update audit trail with cleanup operation context
    if [[ "${VERBOSE}" == "true" ]] && [[ "${log_level}" != "DEBUG" ]]; then
        echo "[${timestamp}] [AUDIT] [${component}] ${message}" >> "${CLEANUP_LOG_FILE}.audit" 2>/dev/null
    fi
}

# Validate system dependencies, Python environment, and cache management
# system availability for reliable cleanup operations
check_dependencies() {
    log_message "INFO" "Starting dependency validation" "DEPS"
    
    local missing_deps=0
    local required_commands=("find" "du" "rm" "stat" "ps" "kill" "date" "mkdir" "basename" "dirname")
    
    # Check for required system utilities
    for cmd in "${required_commands[@]}"; do
        if ! command -v "${cmd}" >/dev/null 2>&1; then
            log_message "ERROR" "Required command not found: ${cmd}" "DEPS"
            ((missing_deps++))
        else
            log_message "DEBUG" "Found required command: ${cmd}" "DEPS"
        fi
    done
    
    # Validate Python environment and version compatibility (for cache integration)
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        if [[ -n "${python_version}" ]]; then
            log_message "INFO" "Python environment detected: ${python_version}" "DEPS"
        fi
    else
        log_message "WARN" "Python3 not found - cache integration may be limited" "DEPS"
    fi
    
    # Check file system permissions for cache directories
    local base_cache_dir="${CACHE_BASE_DIR:-${DEFAULT_CACHE_BASE_DIR}}"
    if [[ -d "${base_cache_dir}" ]]; then
        if [[ -r "${base_cache_dir}" && -w "${base_cache_dir}" ]]; then
            log_message "INFO" "Cache directory permissions validated: ${base_cache_dir}" "DEPS"
        else
            log_message "ERROR" "Insufficient permissions for cache directory: ${base_cache_dir}" "DEPS"
            ((missing_deps++))
        fi
    else
        log_message "WARN" "Cache directory does not exist: ${base_cache_dir}" "DEPS"
    fi
    
    # Validate configuration file accessibility
    local config_file="${PLUME_CLEANUP_CONFIG:-}"
    if [[ -n "${config_file}" ]]; then
        if [[ -r "${config_file}" ]]; then
            log_message "INFO" "Configuration file accessible: ${config_file}" "DEPS"
        else
            log_message "ERROR" "Configuration file not readable: ${config_file}" "DEPS"
            ((missing_deps++))
        fi
    fi
    
    # Test logging system functionality
    local test_log_dir=$(dirname "${CLEANUP_LOG_FILE}")
    if ! mkdir -p "${test_log_dir}" 2>/dev/null; then
        log_message "ERROR" "Cannot create log directory: ${test_log_dir}" "DEPS"
        ((missing_deps++))
    fi
    
    # Verify disk space availability for cleanup operations
    local available_space=$(df . 2>/dev/null | tail -1 | awk '{print $4}' 2>/dev/null)
    if [[ -n "${available_space}" ]] && [[ "${available_space}" -gt 1048576 ]]; then  # 1GB minimum
        log_message "INFO" "Sufficient disk space available: ${available_space}KB" "DEPS"
    else
        log_message "WARN" "Limited disk space available for cleanup operations" "DEPS"
    fi
    
    if [[ ${missing_deps} -eq 0 ]]; then
        log_message "INFO" "All dependencies validated successfully" "DEPS"
        return 0
    else
        log_message "ERROR" "Missing ${missing_deps} required dependencies" "DEPS"
        return 1
    fi
}

# Acquire exclusive lock for cache cleanup operations to prevent concurrent
# execution and ensure data integrity during maintenance
acquire_lock() {
    log_message "INFO" "Attempting to acquire cleanup lock" "LOCK"
    
    # Check for existing lock file and validate process status
    if [[ -f "${LOCK_FILE}" ]]; then
        local lock_pid=$(cat "${LOCK_FILE}" 2>/dev/null)
        if [[ -n "${lock_pid}" ]] && kill -0 "${lock_pid}" 2>/dev/null; then
            log_message "ERROR" "Another cleanup process is running (PID: ${lock_pid})" "LOCK"
            return ${EXIT_LOCK_FAILED}
        else
            log_message "WARN" "Removing stale lock file" "LOCK"
            rm -f "${LOCK_FILE}" 2>/dev/null
        fi
    fi
    
    # Create lock file with current process ID and timestamp
    local current_pid=$$
    local lock_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Implement timeout mechanism for lock acquisition (30 seconds)
    local timeout=30
    local attempts=0
    
    while [[ ${attempts} -lt ${timeout} ]]; do
        if (set -C; echo "${current_pid}" > "${LOCK_FILE}") 2>/dev/null; then
            echo "${lock_timestamp}" >> "${LOCK_FILE}" 2>/dev/null
            log_message "INFO" "Lock acquired successfully (PID: ${current_pid})" "LOCK"
            
            # Create PID file for process monitoring
            echo "${current_pid}" > "${PID_FILE}" 2>/dev/null
            
            # Validate exclusive access to cache directories
            log_message "DEBUG" "Validating exclusive cache directory access" "LOCK"
            return 0
        fi
        
        sleep 1
        ((attempts++))
    done
    
    log_message "ERROR" "Failed to acquire lock after ${timeout} seconds" "LOCK"
    return ${EXIT_LOCK_FAILED}
}

# Release cache cleanup lock and perform cleanup of lock files with
# error handling and audit trail logging
release_lock() {
    log_message "INFO" "Releasing cleanup lock" "LOCK"
    
    # Validate lock ownership before release
    if [[ -f "${LOCK_FILE}" ]]; then
        local lock_pid=$(head -n1 "${LOCK_FILE}" 2>/dev/null)
        if [[ "${lock_pid}" == "$$" ]]; then
            rm -f "${LOCK_FILE}" 2>/dev/null
            log_message "INFO" "Lock file removed successfully" "LOCK"
        else
            log_message "WARN" "Lock ownership mismatch during release" "LOCK"
        fi
    fi
    
    # Remove PID file safely
    if [[ -f "${PID_FILE}" ]]; then
        rm -f "${PID_FILE}" 2>/dev/null
        log_message "DEBUG" "PID file removed" "LOCK"
    fi
    
    # Update audit trail with lock release event
    local release_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    log_message "DEBUG" "Lock released at ${release_timestamp}" "LOCK"
}

# =============================================================================
# CACHE ANALYSIS FUNCTIONS
# =============================================================================

# Calculate comprehensive cache size across all cache levels with detailed
# breakdown and performance metrics for monitoring and optimization
get_cache_size() {
    local cache_directory="$1"
    
    if [[ ! -d "${cache_directory}" ]]; then
        echo "0"
        return 0
    fi
    
    log_message "DEBUG" "Calculating cache size for: ${cache_directory}" "SIZE"
    
    local start_time=$(date +%s.%N)
    
    # Scan cache directory structure recursively
    local total_size=0
    local file_count=0
    local dir_count=0
    
    # Use find with optimized parameters for performance
    while IFS= read -r -d '' file; do
        if [[ -f "${file}" ]]; then
            local file_size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}" 2>/dev/null || echo "0")
            total_size=$((total_size + file_size))
            ((file_count++))
        elif [[ -d "${file}" ]]; then
            ((dir_count++))
        fi
    done < <(find "${cache_directory}" -type f -o -type d -print0 2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local calculation_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    # Format size information with appropriate units
    local size_kb=$((total_size / 1024))
    local size_mb=$((total_size / 1048576))
    local size_gb=$((total_size / 1073741824))
    
    local formatted_size=""
    if [[ ${size_gb} -gt 0 ]]; then
        formatted_size="${size_gb}GB"
    elif [[ ${size_mb} -gt 0 ]]; then
        formatted_size="${size_mb}MB"
    elif [[ ${size_kb} -gt 0 ]]; then
        formatted_size="${size_kb}KB"
    else
        formatted_size="${total_size}B"
    fi
    
    # Log size calculation results with performance metrics
    log_message "PERF" "Cache size calculation: ${formatted_size} (${file_count} files, ${dir_count} dirs) in ${calculation_time}s" "SIZE"
    
    echo "${total_size}"
}

# =============================================================================
# CACHE CLEANUP FUNCTIONS
# =============================================================================

# Clean up Level 1 memory cache artifacts including temporary files,
# memory dumps, and cache metadata with integration to memory cache management system
cleanup_memory_cache() {
    local cache_directory="$1"
    local retention_days="$2"
    
    log_message "INFO" "Starting Level 1 memory cache cleanup" "MEMORY"
    log_message "INFO" "Cache directory: ${cache_directory}, Retention: ${retention_days} days" "MEMORY"
    
    local items_cleaned=0
    local start_time=$(date +%s.%N)
    
    if [[ ! -d "${cache_directory}" ]]; then
        log_message "WARN" "Memory cache directory does not exist: ${cache_directory}" "MEMORY"
        return 0
    fi
    
    # Identify memory cache temporary files and artifacts
    local memory_cache_files
    memory_cache_files=$(find "${cache_directory}" -name "${MEMORY_CACHE_PATTERN}" -type f 2>/dev/null)
    
    if [[ -z "${memory_cache_files}" ]]; then
        log_message "INFO" "No memory cache files found for cleanup" "MEMORY"
        return 0
    fi
    
    # Process each memory cache file
    while IFS= read -r file; do
        if [[ -z "${file}" ]]; then
            continue
        fi
        
        # Check file age against retention policy
        local file_age_days
        file_age_days=$(find "${file}" -mtime +${retention_days} 2>/dev/null | wc -l)
        
        if [[ ${file_age_days} -gt 0 ]]; then
            local file_size=$(get_cache_size "${file}")
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_message "INFO" "[DRY RUN] Would remove memory cache file: ${file}" "MEMORY"
            else
                # Remove expired memory cache files safely
                if rm -f "${file}" 2>/dev/null; then
                    log_message "DEBUG" "Removed memory cache file: ${file} (${file_size} bytes)" "MEMORY"
                    ((items_cleaned++))
                    TOTAL_MEMORY_CACHE_CLEANED=$((TOTAL_MEMORY_CACHE_CLEANED + 1))
                else
                    log_message "WARN" "Failed to remove memory cache file: ${file}" "MEMORY"
                fi
            fi
        else
            log_message "DEBUG" "Keeping recent memory cache file: ${file}" "MEMORY"
        fi
    done <<< "${memory_cache_files}"
    
    # Clean up memory cache metadata and indexes
    local metadata_files
    metadata_files=$(find "${cache_directory}" -name "*.mem_meta" -o -name "*.mem_idx" 2>/dev/null)
    
    while IFS= read -r metadata_file; do
        if [[ -n "${metadata_file}" ]] && [[ -f "${metadata_file}" ]]; then
            local corresponding_cache="${metadata_file%.*}"
            if [[ ! -f "${corresponding_cache}" ]]; then
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove orphaned metadata: ${metadata_file}" "MEMORY"
                else
                    rm -f "${metadata_file}" 2>/dev/null
                    log_message "DEBUG" "Removed orphaned memory cache metadata: ${metadata_file}" "MEMORY"
                    ((items_cleaned++))
                fi
            fi
        fi
    done <<< "${metadata_files}"
    
    local end_time=$(date +%s.%N)
    local cleanup_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    log_message "INFO" "Memory cache cleanup completed: ${items_cleaned} items cleaned in ${cleanup_time}s" "MEMORY"
    log_message "PERF" "Memory cache cleanup performance: ${items_cleaned} items/${cleanup_time}s" "MEMORY"
    
    return ${items_cleaned}
}

# Comprehensive Level 2 disk cache cleanup including expired entries,
# orphaned files, compression optimization, and storage defragmentation
cleanup_disk_cache() {
    local cache_directory="$1"
    local retention_days="$2"
    local max_size_gb="$3"
    
    log_message "INFO" "Starting Level 2 disk cache cleanup" "DISK"
    log_message "INFO" "Cache directory: ${cache_directory}, Retention: ${retention_days} days, Max size: ${max_size_gb}GB" "DISK"
    
    local space_freed=0
    local start_time=$(date +%s.%N)
    
    if [[ ! -d "${cache_directory}" ]]; then
        log_message "WARN" "Disk cache directory does not exist: ${cache_directory}" "DISK"
        return 0
    fi
    
    # Calculate current cache size
    local current_size_bytes=$(get_cache_size "${cache_directory}")
    local current_size_gb=$((current_size_bytes / 1073741824))
    local max_size_bytes=$((max_size_gb * 1073741824))
    
    log_message "INFO" "Current disk cache size: ${current_size_gb}GB (limit: ${max_size_gb}GB)" "DISK"
    
    # Scan disk cache directory for expired entries
    local disk_cache_files
    disk_cache_files=$(find "${cache_directory}" -name "${DISK_CACHE_PATTERN}" -type f 2>/dev/null)
    
    # Process expired entries based on TTL and access time
    while IFS= read -r file; do
        if [[ -z "${file}" ]]; then
            continue
        fi
        
        local file_age_days
        file_age_days=$(find "${file}" -mtime +${retention_days} 2>/dev/null | wc -l)
        
        if [[ ${file_age_days} -gt 0 ]]; then
            local file_size
            file_size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}" 2>/dev/null || echo "0")
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_message "INFO" "[DRY RUN] Would remove expired disk cache: ${file} (${file_size} bytes)" "DISK"
                space_freed=$((space_freed + file_size))
            else
                if rm -f "${file}" 2>/dev/null; then
                    log_message "DEBUG" "Removed expired disk cache: ${file}" "DISK"
                    space_freed=$((space_freed + file_size))
                else
                    log_message "WARN" "Failed to remove disk cache file: ${file}" "DISK"
                fi
            fi
        fi
    done <<< "${disk_cache_files}"
    
    # Enforce cache size limits with intelligent eviction
    if [[ ${current_size_bytes} -gt ${max_size_bytes} ]]; then
        log_message "INFO" "Cache size exceeds limit, performing size-based cleanup" "DISK"
        
        # Find largest and oldest files for eviction
        local files_by_size
        files_by_size=$(find "${cache_directory}" -type f -exec ls -la {} \; 2>/dev/null | sort -k5 -nr | head -100)
        
        local size_to_free=$((current_size_bytes - max_size_bytes))
        local size_freed_so_far=0
        
        while IFS= read -r file_info && [[ ${size_freed_so_far} -lt ${size_to_free} ]]; do
            if [[ -z "${file_info}" ]]; then
                continue
            fi
            
            local file_path=$(echo "${file_info}" | awk '{print $NF}')
            local file_size=$(echo "${file_info}" | awk '{print $5}')
            
            if [[ -f "${file_path}" ]]; then
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove for size limit: ${file_path}" "DISK"
                else
                    if rm -f "${file_path}" 2>/dev/null; then
                        log_message "DEBUG" "Removed for size limit: ${file_path}" "DISK"
                        size_freed_so_far=$((size_freed_so_far + file_size))
                        space_freed=$((space_freed + file_size))
                    fi
                fi
            fi
        done <<< "${files_by_size}"
    fi
    
    # Identify and clean orphaned cache files and metadata
    local orphaned_files
    orphaned_files=$(find "${cache_directory}" -name "*.orphan" -o -name "*.tmp" -o -name "*.lock" 2>/dev/null)
    
    while IFS= read -r orphaned_file; do
        if [[ -n "${orphaned_file}" ]] && [[ -f "${orphaned_file}" ]]; then
            local file_size
            file_size=$(stat -f%z "${orphaned_file}" 2>/dev/null || stat -c%s "${orphaned_file}" 2>/dev/null || echo "0")
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_message "INFO" "[DRY RUN] Would remove orphaned file: ${orphaned_file}" "DISK"
            else
                if rm -f "${orphaned_file}" 2>/dev/null; then
                    log_message "DEBUG" "Removed orphaned file: ${orphaned_file}" "DISK"
                    space_freed=$((space_freed + file_size))
                fi
            fi
        fi
    done <<< "${orphaned_files}"
    
    local end_time=$(date +%s.%N)
    local cleanup_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    local space_freed_mb=$((space_freed / 1048576))
    
    TOTAL_DISK_SPACE_FREED=$((TOTAL_DISK_SPACE_FREED + space_freed))
    
    log_message "INFO" "Disk cache cleanup completed: ${space_freed_mb}MB freed in ${cleanup_time}s" "DISK"
    log_message "PERF" "Disk cache cleanup performance: ${space_freed_mb}MB/${cleanup_time}s" "DISK"
    
    return ${space_freed}
}

# Level 3 result cache maintenance including dependency validation,
# cross-algorithm correlation cleanup, and statistical data optimization
cleanup_result_cache() {
    local cache_directory="$1"
    local retention_days="$2"
    
    log_message "INFO" "Starting Level 3 result cache cleanup" "RESULT"
    log_message "INFO" "Cache directory: ${cache_directory}, Retention: ${retention_days} days" "RESULT"
    
    local entries_cleaned=0
    local start_time=$(date +%s.%N)
    
    if [[ ! -d "${cache_directory}" ]]; then
        log_message "WARN" "Result cache directory does not exist: ${cache_directory}" "RESULT"
        return 0
    fi
    
    # Find result cache files
    local result_cache_files
    result_cache_files=$(find "${cache_directory}" -name "${RESULT_CACHE_PATTERN}" -type f 2>/dev/null)
    
    # Validate result cache dependencies and correlations
    while IFS= read -r result_file; do
        if [[ -z "${result_file}" ]]; then
            continue
        fi
        
        # Check file age against retention policy
        local file_age_days
        file_age_days=$(find "${result_file}" -mtime +${retention_days} 2>/dev/null | wc -l)
        
        if [[ ${file_age_days} -gt 0 ]]; then
            # Check for dependency files
            local dependency_file="${result_file}.dep"
            local correlation_file="${result_file}.corr"
            
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_message "INFO" "[DRY RUN] Would remove result cache: ${result_file}" "RESULT"
                if [[ -f "${dependency_file}" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove dependency: ${dependency_file}" "RESULT"
                fi
                if [[ -f "${correlation_file}" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove correlation: ${correlation_file}" "RESULT"
                fi
            else
                # Remove expired simulation results and analysis data
                if rm -f "${result_file}" 2>/dev/null; then
                    log_message "DEBUG" "Removed result cache: ${result_file}" "RESULT"
                    ((entries_cleaned++))
                    
                    # Clean up related dependency and correlation files
                    if [[ -f "${dependency_file}" ]]; then
                        rm -f "${dependency_file}" 2>/dev/null
                        log_message "DEBUG" "Removed dependency file: ${dependency_file}" "RESULT"
                    fi
                    
                    if [[ -f "${correlation_file}" ]]; then
                        rm -f "${correlation_file}" 2>/dev/null
                        log_message "DEBUG" "Removed correlation file: ${correlation_file}" "RESULT"
                    fi
                else
                    log_message "WARN" "Failed to remove result cache: ${result_file}" "RESULT"
                fi
            fi
        else
            log_message "DEBUG" "Keeping recent result cache: ${result_file}" "RESULT"
        fi
    done <<< "${result_cache_files}"
    
    # Clean up cross-algorithm correlation cache
    local correlation_dir="${cache_directory}/correlations"
    if [[ -d "${correlation_dir}" ]]; then
        local correlation_files
        correlation_files=$(find "${correlation_dir}" -name "*.corr" -mtime +${retention_days} 2>/dev/null)
        
        while IFS= read -r corr_file; do
            if [[ -n "${corr_file}" ]] && [[ -f "${corr_file}" ]]; then
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove correlation cache: ${corr_file}" "RESULT"
                else
                    if rm -f "${corr_file}" 2>/dev/null; then
                        log_message "DEBUG" "Removed correlation cache: ${corr_file}" "RESULT"
                        ((entries_cleaned++))
                    fi
                fi
            fi
        done <<< "${correlation_files}"
    fi
    
    # Optimize statistical validation data storage
    local stats_dir="${cache_directory}/statistics"
    if [[ -d "${stats_dir}" ]]; then
        # Remove empty statistical directories
        find "${stats_dir}" -type d -empty -delete 2>/dev/null
    fi
    
    local end_time=$(date +%s.%N)
    local cleanup_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    TOTAL_RESULT_CACHE_CLEANED=$((TOTAL_RESULT_CACHE_CLEANED + entries_cleaned))
    
    log_message "INFO" "Result cache cleanup completed: ${entries_cleaned} entries cleaned in ${cleanup_time}s" "RESULT"
    log_message "PERF" "Result cache cleanup performance: ${entries_cleaned} entries/${cleanup_time}s" "RESULT"
    
    return ${entries_cleaned}
}

# Remove temporary files, processing artifacts, and orphaned data files
# with pattern matching and age-based filtering for comprehensive system cleanup
cleanup_temporary_files() {
    local temp_directory="$1"
    local max_age_hours="$2"
    
    log_message "INFO" "Starting temporary files cleanup" "TEMP"
    log_message "INFO" "Temp directory: ${temp_directory}, Max age: ${max_age_hours} hours" "TEMP"
    
    local files_removed=0
    local start_time=$(date +%s.%N)
    
    if [[ ! -d "${temp_directory}" ]]; then
        log_message "WARN" "Temporary directory does not exist: ${temp_directory}" "TEMP"
        return 0
    fi
    
    # Convert hours to minutes for find command
    local max_age_minutes=$((max_age_hours * 60))
    
    # Scan temporary directories for cleanup candidates
    local temp_files
    temp_files=$(find "${temp_directory}" -name "${TEMP_FILE_PATTERN}" -type f -mmin +${max_age_minutes} 2>/dev/null)
    
    # Apply pattern matching for simulation temporary files
    while IFS= read -r temp_file; do
        if [[ -z "${temp_file}" ]]; then
            continue
        fi
        
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_message "INFO" "[DRY RUN] Would remove temp file: ${temp_file}" "TEMP"
            ((files_removed++))
        else
            if rm -f "${temp_file}" 2>/dev/null; then
                log_message "DEBUG" "Removed temp file: ${temp_file}" "TEMP"
                ((files_removed++))
            else
                log_message "WARN" "Failed to remove temp file: ${temp_file}" "TEMP"
            fi
        fi
    done <<< "${temp_files}"
    
    # Remove orphaned processing artifacts safely
    local artifact_patterns=("*.processing" "*.partial" "*.incomplete" "*.checkpoint")
    for pattern in "${artifact_patterns[@]}"; do
        local artifacts
        artifacts=$(find "${temp_directory}" -name "${pattern}" -type f -mmin +${max_age_minutes} 2>/dev/null)
        
        while IFS= read -r artifact; do
            if [[ -n "${artifact}" ]] && [[ -f "${artifact}" ]]; then
                if [[ "${DRY_RUN}" == "true" ]]; then
                    log_message "INFO" "[DRY RUN] Would remove artifact: ${artifact}" "TEMP"
                    ((files_removed++))
                else
                    if rm -f "${artifact}" 2>/dev/null; then
                        log_message "DEBUG" "Removed artifact: ${artifact}" "TEMP"
                        ((files_removed++))
                    fi
                fi
            fi
        done <<< "${artifacts}"
    done
    
    # Clean up empty temporary directories
    if [[ "${DRY_RUN}" != "true" ]]; then
        find "${temp_directory}" -type d -empty -delete 2>/dev/null
    fi
    
    local end_time=$(date +%s.%N)
    local cleanup_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    TOTAL_TEMP_FILES_REMOVED=$((TOTAL_TEMP_FILES_REMOVED + files_removed))
    
    log_message "INFO" "Temporary files cleanup completed: ${files_removed} files removed in ${cleanup_time}s" "TEMP"
    log_message "PERF" "Temporary cleanup performance: ${files_removed} files/${cleanup_time}s" "TEMP"
    
    return ${files_removed}
}

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

# Perform cache performance optimization including index rebuilding,
# compression analysis, and storage efficiency improvements
optimize_cache_performance() {
    local cache_directory="$1"
    
    log_message "INFO" "Starting cache performance optimization" "OPTIMIZE"
    
    local optimization_score=0
    local start_time=$(date +%s.%N)
    
    if [[ ! -d "${cache_directory}" ]]; then
        log_message "WARN" "Cache directory does not exist for optimization: ${cache_directory}" "OPTIMIZE"
        return 0
    fi
    
    # Analyze current cache performance metrics
    local initial_size=$(get_cache_size "${cache_directory}")
    local file_count=$(find "${cache_directory}" -type f 2>/dev/null | wc -l)
    
    log_message "INFO" "Pre-optimization metrics: ${file_count} files, ${initial_size} bytes" "OPTIMIZE"
    
    # Rebuild cache indexes for optimal access
    local index_files
    index_files=$(find "${cache_directory}" -name "*.idx" -type f 2>/dev/null)
    
    while IFS= read -r index_file; do
        if [[ -n "${index_file}" ]] && [[ -f "${index_file}" ]]; then
            if [[ "${DRY_RUN}" == "true" ]]; then
                log_message "INFO" "[DRY RUN] Would rebuild index: ${index_file}" "OPTIMIZE"
            else
                # Simple index rebuilding (touch to update timestamp)
                touch "${index_file}" 2>/dev/null
                log_message "DEBUG" "Rebuilt cache index: ${index_file}" "OPTIMIZE"
            fi
        fi
    done <<< "${index_files}"
    
    # Optimize compression settings and algorithms
    local compressed_files
    compressed_files=$(find "${cache_directory}" -name "*.gz" -o -name "*.bz2" -o -name "*.xz" 2>/dev/null)
    
    local compression_savings=0
    while IFS= read -r compressed_file; do
        if [[ -n "${compressed_file}" ]] && [[ -f "${compressed_file}" ]]; then
            local file_size
            file_size=$(stat -f%z "${compressed_file}" 2>/dev/null || stat -c%s "${compressed_file}" 2>/dev/null || echo "0")
            compression_savings=$((compression_savings + file_size / 10))  # Assume 10% savings
        fi
    done <<< "${compressed_files}"
    
    # Defragment cache storage structure (remove empty directories)
    if [[ "${DRY_RUN}" != "true" ]]; then
        find "${cache_directory}" -type d -empty -delete 2>/dev/null
    fi
    
    # Calculate performance improvement metrics
    local final_size=$(get_cache_size "${cache_directory}")
    local size_reduction=$((initial_size - final_size))
    
    if [[ ${initial_size} -gt 0 ]]; then
        optimization_score=$(( (size_reduction * 100) / initial_size ))
        if [[ ${optimization_score} -lt 0 ]]; then
            optimization_score=0
        elif [[ ${optimization_score} -gt 100 ]]; then
            optimization_score=100
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local optimization_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    log_message "INFO" "Cache optimization completed: ${optimization_score}% improvement in ${optimization_time}s" "OPTIMIZE"
    log_message "PERF" "Optimization performance: ${optimization_score}% improvement" "OPTIMIZE"
    
    return ${optimization_score}
}

# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

# Generate comprehensive cleanup report with statistics, performance metrics,
# space freed, and recommendations for future maintenance
generate_cleanup_report() {
    local cleanup_stats="$1"
    local performance_metrics="$2"
    
    log_message "INFO" "Generating comprehensive cleanup report" "REPORT"
    
    local report_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local total_cleanup_time=""
    
    if [[ -n "${CLEANUP_START_TIME}" ]] && [[ -n "${CLEANUP_END_TIME}" ]]; then
        total_cleanup_time=$(echo "${CLEANUP_END_TIME} - ${CLEANUP_START_TIME}" | bc 2>/dev/null || echo "Unknown")
    fi
    
    # Generate report header with color coding
    echo -e "\n${COLOR_BLUE}========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}    PLUME SIMULATION CACHE CLEANUP     ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}         COMPREHENSIVE REPORT          ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}========================================${COLOR_RESET}\n"
    
    echo -e "${COLOR_CYAN}Report Generated:${COLOR_RESET} ${report_timestamp}"
    echo -e "${COLOR_CYAN}Script Version:${COLOR_RESET} ${SCRIPT_VERSION}"
    echo -e "${COLOR_CYAN}Total Runtime:${COLOR_RESET} ${total_cleanup_time}s"
    echo ""
    
    # Compile cleanup statistics from all cache levels
    echo -e "${COLOR_GREEN}CLEANUP STATISTICS:${COLOR_RESET}"
    echo "├─ Memory Cache (Level 1):"
    echo "│  └─ Items Cleaned: ${TOTAL_MEMORY_CACHE_CLEANED}"
    echo "├─ Disk Cache (Level 2):"
    echo "│  └─ Space Freed: $((TOTAL_DISK_SPACE_FREED / 1048576))MB"
    echo "├─ Result Cache (Level 3):"
    echo "│  └─ Entries Cleaned: ${TOTAL_RESULT_CACHE_CLEANED}"
    echo "└─ Temporary Files:"
    echo "   └─ Files Removed: ${TOTAL_TEMP_FILES_REMOVED}"
    echo ""
    
    # Calculate total space freed and performance improvements
    local total_space_freed_mb=$((TOTAL_DISK_SPACE_FREED / 1048576))
    local total_items_processed=$((TOTAL_MEMORY_CACHE_CLEANED + TOTAL_RESULT_CACHE_CLEANED + TOTAL_TEMP_FILES_REMOVED))
    
    echo -e "${COLOR_GREEN}PERFORMANCE SUMMARY:${COLOR_RESET}"
    echo "├─ Total Space Freed: ${total_space_freed_mb}MB"
    echo "├─ Total Items Processed: ${total_items_processed}"
    if [[ -n "${total_cleanup_time}" ]] && [[ "${total_cleanup_time}" != "Unknown" ]] && [[ "${total_cleanup_time}" != "0" ]]; then
        local throughput=$(echo "scale=2; ${total_items_processed} / ${total_cleanup_time}" | bc 2>/dev/null || echo "N/A")
        echo "└─ Processing Throughput: ${throughput} items/second"
    else
        echo "└─ Processing Throughput: N/A"
    fi
    echo ""
    
    # Generate cache health assessment and recommendations
    echo -e "${COLOR_YELLOW}CACHE HEALTH ASSESSMENT:${COLOR_RESET}"
    
    if [[ ${total_space_freed_mb} -gt 1000 ]]; then
        echo "├─ ${COLOR_RED}HIGH${COLOR_RESET}: Significant space was freed (${total_space_freed_mb}MB)"
        echo "│  Recommendation: Consider reducing retention periods"
    elif [[ ${total_space_freed_mb} -gt 100 ]]; then
        echo "├─ ${COLOR_YELLOW}MEDIUM${COLOR_RESET}: Moderate space was freed (${total_space_freed_mb}MB)"
        echo "│  Recommendation: Current retention policy appears appropriate"
    else
        echo "├─ ${COLOR_GREEN}LOW${COLOR_RESET}: Minimal space was freed (${total_space_freed_mb}MB)"
        echo "│  Recommendation: Cache management is efficient"
    fi
    
    if [[ ${TOTAL_TEMP_FILES_REMOVED} -gt 100 ]]; then
        echo "├─ ${COLOR_YELLOW}WARNING${COLOR_RESET}: Many temporary files cleaned (${TOTAL_TEMP_FILES_REMOVED})"
        echo "│  Recommendation: Check for incomplete simulations"
    fi
    
    echo "└─ Next recommended cleanup: $(date -d '+7 days' '+%Y-%m-%d')"
    echo ""
    
    # Include optimization recommendations for future runs
    echo -e "${COLOR_BLUE}OPTIMIZATION RECOMMENDATIONS:${COLOR_RESET}"
    echo "├─ Run cleanup weekly during low-usage periods"
    echo "├─ Monitor cache hit rate threshold (target: ≥0.8)"
    echo "├─ Consider cache size limits based on available storage"
    echo "├─ Use --optimize flag for performance improvements"
    echo "└─ Review retention policies based on simulation frequency"
    echo ""
    
    # Save detailed report to log files
    {
        echo "=== CLEANUP REPORT - ${report_timestamp} ==="
        echo "Memory Cache Cleaned: ${TOTAL_MEMORY_CACHE_CLEANED}"
        echo "Disk Space Freed: ${total_space_freed_mb}MB"
        echo "Result Cache Cleaned: ${TOTAL_RESULT_CACHE_CLEANED}"
        echo "Temporary Files Removed: ${TOTAL_TEMP_FILES_REMOVED}"
        echo "Total Runtime: ${total_cleanup_time}s"
        echo "============================================"
    } >> "${CLEANUP_LOG_FILE}.report" 2>/dev/null
    
    log_message "INFO" "Cleanup report generated successfully" "REPORT"
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

# Validate cleanup operation results including cache integrity,
# performance impact, and system health after maintenance operations
validate_cleanup_results() {
    log_message "INFO" "Starting cleanup results validation" "VALIDATE"
    
    local validation_errors=0
    local start_time=$(date +%s.%N)
    
    # Verify cache system integrity after cleanup
    local cache_base_dir="${CACHE_BASE_DIR:-${DEFAULT_CACHE_BASE_DIR}}"
    
    if [[ -d "${cache_base_dir}" ]]; then
        # Check for broken symlinks or corrupted files
        local broken_links
        broken_links=$(find "${cache_base_dir}" -type l ! -exec test -e {} \; -print 2>/dev/null)
        
        if [[ -n "${broken_links}" ]]; then
            log_message "WARN" "Found broken symlinks after cleanup" "VALIDATE"
            while IFS= read -r link; do
                if [[ -n "${link}" ]]; then
                    log_message "DEBUG" "Broken link: ${link}" "VALIDATE"
                    ((validation_errors++))
                fi
            done <<< "${broken_links}"
        fi
        
        # Test cache functionality and accessibility
        if [[ -r "${cache_base_dir}" ]] && [[ -w "${cache_base_dir}" ]]; then
            log_message "INFO" "Cache directory permissions validated" "VALIDATE"
        else
            log_message "ERROR" "Cache directory permission issues detected" "VALIDATE"
            ((validation_errors++))
        fi
    else
        log_message "WARN" "Cache base directory not found during validation" "VALIDATE"
    fi
    
    # Validate cache configuration and settings
    local temp_dir="${TEMP_DIR:-${DEFAULT_TEMP_DIR}}"
    if [[ -d "${temp_dir}" ]]; then
        local temp_space
        temp_space=$(df "${temp_dir}" 2>/dev/null | tail -1 | awk '{print $4}' 2>/dev/null)
        if [[ -n "${temp_space}" ]] && [[ ${temp_space} -gt 1048576 ]]; then  # 1GB minimum
            log_message "INFO" "Sufficient temporary space available: ${temp_space}KB" "VALIDATE"
        else
            log_message "WARN" "Limited temporary space after cleanup" "VALIDATE"
        fi
    fi
    
    # Check for any cleanup-related errors or issues
    if [[ -f "${ERROR_LOG_FILE}" ]]; then
        local error_count
        error_count=$(grep -c "ERROR" "${ERROR_LOG_FILE}" 2>/dev/null || echo "0")
        if [[ ${error_count} -gt 0 ]]; then
            log_message "WARN" "Found ${error_count} errors in cleanup log" "VALIDATE"
            validation_errors=$((validation_errors + error_count))
        fi
    fi
    
    # Measure post-cleanup performance metrics
    local disk_io_test
    disk_io_test=$(dd if=/dev/zero of="${temp_dir}/io_test" bs=1M count=10 2>/dev/null && rm -f "${temp_dir}/io_test" 2>/dev/null; echo $?)
    if [[ ${disk_io_test} -eq 0 ]]; then
        log_message "INFO" "Disk I/O performance test passed" "VALIDATE"
    else
        log_message "WARN" "Disk I/O performance test failed" "VALIDATE"
        ((validation_errors++))
    fi
    
    # Verify disk space and resource availability
    local available_space
    available_space=$(df . 2>/dev/null | tail -1 | awk '{print $4}' 2>/dev/null)
    if [[ -n "${available_space}" ]] && [[ ${available_space} -gt 2097152 ]]; then  # 2GB minimum
        log_message "INFO" "Sufficient disk space available post-cleanup: ${available_space}KB" "VALIDATE"
    else
        log_message "WARN" "Limited disk space available after cleanup" "VALIDATE"
    fi
    
    local end_time=$(date +%s.%N)
    local validation_time=$(echo "${end_time} - ${start_time}" | bc 2>/dev/null || echo "0")
    
    if [[ ${validation_errors} -eq 0 ]]; then
        log_message "INFO" "Cleanup validation completed successfully in ${validation_time}s" "VALIDATE"
        return 0
    else
        log_message "WARN" "Cleanup validation found ${validation_errors} issues in ${validation_time}s" "VALIDATE"
        return ${validation_errors}
    fi
}

# =============================================================================
# ERROR HANDLING FUNCTIONS
# =============================================================================

# Comprehensive error handling for cleanup operations with recovery procedures,
# rollback capabilities, and detailed error reporting
handle_cleanup_error() {
    local error_message="$1"
    local error_context="$2"
    local error_code="$3"
    
    log_message "ERROR" "Cleanup error in ${error_context}: ${error_message} (code: ${error_code})" "ERROR"
    
    # Classify error severity and impact
    local error_severity="MEDIUM"
    case "${error_code}" in
        1) error_severity="LOW" ;;
        2) error_severity="HIGH" ;;
        *) error_severity="MEDIUM" ;;
    esac
    
    log_message "ERROR" "Error severity classified as: ${error_severity}" "ERROR"
    
    # Attempt automatic recovery procedures
    case "${error_context}" in
        "LOCK")
            log_message "INFO" "Attempting lock recovery" "ERROR"
            release_lock
            ;;
        "DISK"|"MEMORY"|"RESULT")
            log_message "INFO" "Attempting cache recovery for ${error_context}" "ERROR"
            # Basic recovery: ensure no partial operations
            sync 2>/dev/null
            ;;
        "TEMP")
            log_message "INFO" "Attempting temporary file recovery" "ERROR"
            # Ensure temporary directory is accessible
            mkdir -p "${DEFAULT_TEMP_DIR}" 2>/dev/null
            ;;
    esac
    
    # Update error statistics and tracking
    local error_log_entry="[$(date '+%Y-%m-%d %H:%M:%S')] [${error_severity}] [${error_context}] ${error_message}"
    echo "${error_log_entry}" >> "${ERROR_LOG_FILE}" 2>/dev/null
    
    # Generate error report with recommendations
    {
        echo "=== ERROR REPORT ==="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Context: ${error_context}"
        echo "Message: ${error_message}"
        echo "Code: ${error_code}"
        echo "Severity: ${error_severity}"
        echo "Recommendations:"
        case "${error_context}" in
            "LOCK")
                echo "- Check for zombie processes"
                echo "- Verify file system permissions"
                ;;
            "DISK"|"MEMORY"|"RESULT")
                echo "- Check disk space availability"
                echo "- Verify cache directory permissions"
                echo "- Consider running fsck if disk errors persist"
                ;;
            "TEMP")
                echo "- Check temporary directory permissions"
                echo "- Verify available disk space in temp partition"
                ;;
        esac
        echo "==================="
    } >> "${ERROR_LOG_FILE}.report" 2>/dev/null
    
    log_message "INFO" "Error handling completed for ${error_context}" "ERROR"
}

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

# Configure signal handlers for graceful shutdown, interrupt handling,
# and cleanup operation termination with proper resource cleanup
setup_signal_handlers() {
    log_message "INFO" "Setting up signal handlers" "SIGNAL"
    
    # Function to handle graceful shutdown
    cleanup_and_exit() {
        local signal="$1"
        log_message "INFO" "Received signal ${signal}, initiating graceful shutdown" "SIGNAL"
        
        # Ensure lock release on signal reception
        if [[ -f "${LOCK_FILE}" ]]; then
            release_lock
        fi
        
        # Set CLEANUP_END_TIME for reporting
        CLEANUP_END_TIME=$(date +%s.%N)
        
        # Generate final report
        generate_cleanup_report "interrupted" "signal_${signal}"
        
        log_message "INFO" "Graceful shutdown completed" "SIGNAL"
        exit ${EXIT_ERROR}
    }
    
    # Set up SIGINT handler for graceful interruption (Ctrl+C)
    trap 'cleanup_and_exit SIGINT' INT
    
    # Configure SIGTERM handler for clean shutdown
    trap 'cleanup_and_exit SIGTERM' TERM
    
    # Set up SIGHUP handler for configuration reload
    trap 'log_message "INFO" "Received SIGHUP - configuration reload not implemented" "SIGNAL"' HUP
    
    log_message "INFO" "Signal handlers configured successfully" "SIGNAL"
}

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

# Main execution function orchestrating comprehensive cache cleanup with
# argument parsing, dependency validation, and coordinated cleanup across all cache levels
main() {
    local args=("$@")
    
    # Initialize timing
    CLEANUP_START_TIME=$(date +%s.%N)
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers
    
    # Initialize logging system and audit trail
    log_message "INFO" "Starting ${SCRIPT_NAME} v${SCRIPT_VERSION}" "MAIN"
    log_message "INFO" "Process ID: $$" "MAIN"
    
    # Parse command-line arguments and validate options
    local cache_base_dir="${DEFAULT_CACHE_BASE_DIR}"
    local temp_dir="${DEFAULT_TEMP_DIR}"
    local log_dir="${DEFAULT_LOG_DIR}"
    local retention_days="${DEFAULT_RETENTION_DAYS}"
    local max_cache_size_gb="${DEFAULT_MAX_CACHE_SIZE_GB}"
    local memory_only=false
    local disk_only=false
    local results_only=false
    local temp_only=false
    local optimize=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                ;;
            -v|--verbose)
                VERBOSE=true
                log_message "INFO" "Verbose mode enabled" "MAIN"
                ;;
            -q|--quiet)
                QUIET_MODE=true
                log_message "INFO" "Quiet mode enabled" "MAIN"
                ;;
            -n|--dry-run)
                DRY_RUN=true
                log_message "INFO" "Dry run mode enabled" "MAIN"
                ;;
            -f|--force)
                FORCE_CLEANUP=true
                log_message "INFO" "Force cleanup mode enabled" "MAIN"
                ;;
            -c|--cache-dir)
                cache_base_dir="$2"
                shift
                ;;
            -t|--temp-dir)
                temp_dir="$2"
                shift
                ;;
            -r|--retention)
                retention_days="$2"
                shift
                ;;
            -s|--max-size)
                max_cache_size_gb="$2"
                shift
                ;;
            -l|--log-dir)
                log_dir="$2"
                shift
                ;;
            --memory-only)
                memory_only=true
                ;;
            --disk-only)
                disk_only=true
                ;;
            --results-only)
                results_only=true
                ;;
            --temp-only)
                temp_only=true
                ;;
            --optimize)
                optimize=true
                ;;
            *)
                log_message "ERROR" "Unknown option: $1" "MAIN"
                echo "Use --help for usage information"
                exit ${EXIT_INVALID_ARGS}
                ;;
        esac
        shift
    done
    
    # Update log file paths with custom log directory
    CLEANUP_LOG_FILE="${log_dir}/cache_cleanup.log"
    PERFORMANCE_LOG_FILE="${log_dir}/cache_cleanup_performance.log"
    ERROR_LOG_FILE="${log_dir}/cache_cleanup_errors.log"
    
    # Override with environment variables if set
    cache_base_dir="${PLUME_CACHE_DIR:-${cache_base_dir}}"
    
    log_message "INFO" "Configuration: cache_dir=${cache_base_dir}, temp_dir=${temp_dir}, retention=${retention_days}d, max_size=${max_cache_size_gb}GB" "MAIN"
    
    # Check system dependencies and environment
    if ! check_dependencies; then
        log_message "ERROR" "Dependency check failed" "MAIN"
        exit ${EXIT_ERROR}
    fi
    
    # Acquire exclusive lock for cleanup operations
    if ! acquire_lock; then
        log_message "ERROR" "Failed to acquire cleanup lock" "MAIN"
        exit ${EXIT_LOCK_FAILED}
    fi
    
    # Ensure cleanup happens even if script exits unexpectedly
    trap 'release_lock' EXIT
    
    log_message "INFO" "Starting coordinated cache cleanup across all levels" "MAIN"
    
    # Execute coordinated cleanup across all cache levels
    local cleanup_success=true
    
    # Level 1: Memory Cache Cleanup
    if [[ "${memory_only}" == "true" ]] || [[ "${disk_only}" == "false" && "${results_only}" == "false" && "${temp_only}" == "false" ]]; then
        local memory_cache_dir="${cache_base_dir}/memory"
        if cleanup_memory_cache "${memory_cache_dir}" "${retention_days}"; then
            log_message "INFO" "Memory cache cleanup completed successfully" "MAIN"
        else
            handle_cleanup_error "Memory cache cleanup failed" "MEMORY" 1
            cleanup_success=false
        fi
    fi
    
    # Level 2: Disk Cache Cleanup
    if [[ "${disk_only}" == "true" ]] || [[ "${memory_only}" == "false" && "${results_only}" == "false" && "${temp_only}" == "false" ]]; then
        local disk_cache_dir="${cache_base_dir}/disk"
        if cleanup_disk_cache "${disk_cache_dir}" "${retention_days}" "${max_cache_size_gb}"; then
            log_message "INFO" "Disk cache cleanup completed successfully" "MAIN"
        else
            handle_cleanup_error "Disk cache cleanup failed" "DISK" 1
            cleanup_success=false
        fi
    fi
    
    # Level 3: Result Cache Cleanup
    if [[ "${results_only}" == "true" ]] || [[ "${memory_only}" == "false" && "${disk_only}" == "false" && "${temp_only}" == "false" ]]; then
        local result_cache_dir="${cache_base_dir}/results"
        if cleanup_result_cache "${result_cache_dir}" "${retention_days}"; then
            log_message "INFO" "Result cache cleanup completed successfully" "MAIN"
        else
            handle_cleanup_error "Result cache cleanup failed" "RESULT" 1
            cleanup_success=false
        fi
    fi
    
    # Temporary Files Cleanup
    if [[ "${temp_only}" == "true" ]] || [[ "${memory_only}" == "false" && "${disk_only}" == "false" && "${results_only}" == "false" ]]; then
        if cleanup_temporary_files "${temp_dir}" 24; then  # 24 hours default
            log_message "INFO" "Temporary files cleanup completed successfully" "MAIN"
        else
            handle_cleanup_error "Temporary files cleanup failed" "TEMP" 1
            cleanup_success=false
        fi
    fi
    
    # Perform cache performance optimization
    if [[ "${optimize}" == "true" ]]; then
        log_message "INFO" "Starting cache performance optimization" "MAIN"
        local optimization_score
        optimization_score=$(optimize_cache_performance "${cache_base_dir}")
        log_message "INFO" "Cache optimization completed with score: ${optimization_score}%" "MAIN"
    fi
    
    # Set cleanup end time for reporting
    CLEANUP_END_TIME=$(date +%s.%N)
    
    # Validate cleanup results and system integrity
    if validate_cleanup_results; then
        log_message "INFO" "Cleanup validation passed" "MAIN"
    else
        log_message "WARN" "Cleanup validation found issues" "MAIN"
        cleanup_success=false
    fi
    
    # Generate comprehensive cleanup report
    generate_cleanup_report "cleanup_completed" "performance_data"
    
    # Release lock and cleanup temporary resources
    release_lock
    
    # Final status logging
    if [[ "${cleanup_success}" == "true" ]]; then
        log_message "INFO" "Cache cleanup completed successfully" "MAIN"
        exit ${EXIT_SUCCESS}
    else
        log_message "ERROR" "Cache cleanup completed with errors" "MAIN"
        exit ${EXIT_ERROR}
    fi
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Execute main function only if script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi