#!/bin/bash

# Test script to verify paths configuration generation

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo -e "${GREEN}Created test directory: ${TEST_DIR}${NC}"

# Create a test template
TEMPLATE_FILE="${TEST_DIR}/project_paths.yaml.template"
OUTPUT_FILE="${TEST_DIR}/project_paths.yaml"

cat > "${TEMPLATE_FILE}" << 'EOL'
# Paths configuration
# This file is auto-generated during setup - do not edit directly

# Path to the Crimaldi HDF5 data file
crimaldi_hdf5: "${PROJECT_DIR}/data/10302017_10cms_bounded.hdf5"

# Temporary directory for processing
tmp_dir: "${TMPDIR:-/tmp}"

# Output directories (relative to project root)
output:
  raw: "data/raw"
  processed: "data/processed"
  figures: "figures"
EOL

# Simple variable substitution function
substitute_vars() {
    local content
    content=$(<"$1")
    
    # Replace ${VAR} or $VAR with their values
    while [[ $content =~ (\$\{([a-zA-Z_][a-zA-Z0-9_]*)(:[-=][^}]*)?\}|\$([a-zA-Z_][a-zA-Z0-9_]*)) ]]; do
        var_full=${BASH_REMATCH[0]}
        var_name=${BASH_REMATCH[2]:-${BASH_REMATCH[4]}}
        
        # Handle default values
        if [[ ${BASH_REMATCH[3]} =~ ^:- ]]; then
            default_value=${BASH_REMATCH[3]:2}
            var_value=${!var_name:-$default_value}
        else
            var_value=${!var_name}
        fi
        
        content=${content//"$var_full"/"$var_value"}
    done
    
    echo "$content"
}

# Test the paths configuration generation
echo -e "${GREEN}Testing paths configuration generation...${NC}"

# Set environment variables for testing
export PROJECT_DIR="/test/project"
export TMPDIR="/custom/tmp"

# Generate the config file
substitute_vars "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"

# Verify the output
echo -e "${GREEN}Checking for expected values in the generated file...${NC}"

# Check for expected values
if ! grep -q "crimaldi_hdf5: \"/test/project/data/10302017_10cms_bounded.hdf5\"" "${OUTPUT_FILE}"; then
    echo -e "${RED}✗ HDF5 path not set correctly${NC}"
    exit 1
fi

if ! grep -q "tmp_dir: \"/custom/tmp\"" "${OUTPUT_FILE}"; then
    echo -e "${RED}✗ TMPDIR not set correctly${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Paths configuration generated successfully${NC}"
echo -e "\nGenerated config at ${OUTPUT_FILE}:"
cat "${OUTPUT_FILE}"

# Clean up
rm -rf "${TEST_DIR}"
echo -e "\n${GREEN}✓ Test completed successfully!${NC}"
