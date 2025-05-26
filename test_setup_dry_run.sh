#!/bin/bash

# Test script to verify the setup script functionality

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo "Created test directory: ${TEST_DIR}"

# Copy necessary files to test directory
cp setup_env.sh "${TEST_DIR}/"
cp setup_utils.sh "${TEST_DIR}/"
cp -r configs "${TEST_DIR}/"  # Copy the entire configs directory

# Create a minimal environment file for testing
cat > "${TEST_DIR}/environment.yml" << 'EOL'
name: test-env
dependencies:
  - python=3.9
  - pip
  - pip:
    - pre-commit
EOL

# Go to test directory
cd "${TEST_DIR}"

# Run the setup script in dry-run mode
echo -e "\n${GREEN}Running setup script in dry-run mode...${NC}"
if bash -n ./setup_env.sh; then
    echo -e "${GREEN}✓ Setup script syntax check passed${NC}
"
else
    echo -e "${RED}✗ Setup script has syntax errors${NC}"
    exit 1
fi

# Test the paths configuration generation
echo -e "${GREEN}Testing paths configuration generation...${NC}"
./setup_env.sh --no-tests

# Check if paths configuration was created
if [ -f "paths.yaml" ]; then
    echo -e "${GREEN}✓ Paths configuration generated successfully${NC}"
    echo "Contents of paths.yaml:"
    cat "paths.yaml"
else
    echo -e "${RED}✗ Failed to generate paths configuration${NC}"
    exit 1
fi

# Clean up
cd - >/dev/null
rm -rf "${TEST_DIR}"
echo -e "\n${GREEN}✓ All tests completed successfully!${NC}"
