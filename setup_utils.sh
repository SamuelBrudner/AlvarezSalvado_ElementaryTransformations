#!/bin/bash
# Utility functions for setup scripts

section() {
    echo
    echo "---- $* ----"
}

error() {
    echo "ERROR: $*" >&2
    exit 1
}

run_command_verbose() {
    echo "+ $*"
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Command failed with status $status: $*" >&2
    fi
    return $status
}

