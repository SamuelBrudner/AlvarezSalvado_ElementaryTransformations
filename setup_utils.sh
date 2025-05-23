#!/bin/bash
# Utility functions for setup scripts

# Colors for log output
COLOR_INFO="\033[1;34m"
COLOR_SUCCESS="\033[1;32m"
COLOR_WARNING="\033[1;33m"
COLOR_ERROR="\033[1;31m"
COLOR_RESET="\033[0m"

log() {
    local level="$1"
    shift
    local color="$COLOR_INFO"
    case "$level" in
        INFO) color="$COLOR_INFO" ;;
        SUCCESS) color="$COLOR_SUCCESS" ;;
        WARNING) color="$COLOR_WARNING" ;;
        ERROR) color="$COLOR_ERROR" ;;
    esac
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${color}${level}${COLOR_RESET}: $*"
}

section() {
    echo
    echo "---- $* ----"
}

error() {
    log ERROR "$*" >&2
    exit 1
}

run_command_verbose() {
    log INFO "Running: $*"
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        log ERROR "Command failed with status $status: $*"
    fi
    return $status
}

