#!/bin/bash
# Utility functions for setup scripts

log() {
  local level="$1"
  shift
  local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

  # Define colors
  local color_reset="\e[0m"
  local color_info="\e[34m"
  local color_success="\e[32m"
  local color_warning="\e[33m"
  local color_error="\e[31m"

  # Set color based on log level
  local color="$color_info"
  case "$level" in
    INFO) color="$color_info" ;;
    SUCCESS) color="$color_success" ;;
    WARNING) color="$color_warning" ;;
    ERROR) color="$color_error" ;;
  esac

  # Log to stderr for errors, stdout for everything else
  if [ "$level" = "ERROR" ]; then
    echo -e "${color}[$timestamp] [$level] $*${color_reset}" >&2
  else
    echo -e "${color}[$timestamp] [$level] $*${color_reset}"
  fi
}

section() {
  echo
  log INFO "---- $* ----"
}

error() {
  log ERROR "$*"
  exit 1
}

run_command_verbose() {
  log INFO "Running: $*"
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    log ERROR "Command failed with status $status: $*"
    return $status
  fi
  return 0
}
