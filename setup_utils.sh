#!/usr/bin/env bash
set -euo pipefail

section() {
  echo -e "\n\033[1;34m==> $*\033[0m"
}

log() {
  local level="$1"; shift
  local color="0"
  case "$level" in
    INFO) color="34";;
    SUCCESS) color="32";;
    WARNING) color="33";;
    ERROR) color="31";;
  esac
  printf "\033[%sm[%s]\033[0m %s\n" "$color" "$level" "$*"
}

error() {
  log ERROR "$*"
  exit 1
}

run_command_verbose() {
  log INFO "Running: $*"
  "$@"
}
