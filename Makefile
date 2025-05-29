# Makefile for the project
# Includes paths from configs/makefile_paths.mk which is generated during setup

SHELL := /bin/bash

# Include the generated paths file
-include configs/makefile_paths.mk

# Default paths (used if makefile_paths.mk is not available)
PROJECT_ROOT ?= $(CURDIR)
SCRIPTS_DIR ?= $(PROJECT_ROOT)/Code
OUTPUT_RAW ?= $(PROJECT_ROOT)/data/raw
OUTPUT_PROCESSED ?= $(PROJECT_ROOT)/data/processed
OUTPUT_FIGURES ?= $(PROJECT_ROOT)/figures

# Ensure output directories exist
$(shell mkdir -p $(OUTPUT_RAW) $(OUTPUT_PROCESSED) $(OUTPUT_FIGURES))

# Python and test configuration
# Use the pytest from the conda environment
CONDA_ENV_PATH ?= $(PWD)/dev_env
PYTHON ?= $(CONDA_ENV_PATH)/bin/python
PYTEST ?= $(CONDA_ENV_PATH)/bin/pytest
PYTEST_OPTS ?= -v

# Default target when you run just 'make'
help:
	@echo "Project: $(PROJECT_NAME)"
	@echo "Root directory: $(PROJECT_ROOT)"
	@echo ""
	@echo "Available targets:"
	@echo "  test          - Run tests with pytest"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  clean         - Clean up temporary files"
	@echo "  clean-all     - Clean all generated files and caches"
	@echo "  setup         - Run setup script to generate configuration"
	@echo "  help-paths    - Show configured paths"

# Show configured paths
help-paths:
	@echo "Project paths:"
	@echo "  Project root: $(PROJECT_ROOT)"
	@echo "  Scripts dir: $(SCRIPTS_DIR)"
	@echo "  Output dirs:"
	@echo "    - Raw: $(OUTPUT_RAW)"
	@echo "    - Processed: $(OUTPUT_PROCESSED)"
	@echo "    - Figures: $(OUTPUT_FIGURES)"

# Setup target to generate configuration
setup:
	@echo "Setting up project configuration..."
	@if [ -f "setup_env.sh" ]; then \
		echo "Running setup_env.sh..."; \
		source ./setup_env.sh; \
	else \
		echo "Error: setup_env.sh not found"; \
		exit 1; \
	fi

# Test target
test:
	@echo "Running tests..."
	cd "$(SCRIPTS_DIR)" && $(PYTHON) -m pytest $(PYTEST_OPTS)

# Test with coverage
test-cov:
	@echo "Running tests with coverage..."
	cd "$(SCRIPTS_DIR)" && \
	$(PYTHON) -m pytest --cov=. --cov-report=term-missing $(PYTEST_OPTS)

# Clean up generated files and caches
clean:
	@echo "Cleaning up..."
	find "$(PROJECT_ROOT)" -type d -name '__pycache__' -exec rm -rf {} +
	find "$(PROJECT_ROOT)" -type f -name '*.py[co]' -delete
	find "$(PROJECT_ROOT)" -type d -name '.pytest_cache' -exec rm -rf {} +
	find "$(PROJECT_ROOT)" -type d -name '.mypy_cache' -exec rm -rf {} +
	find "$(PROJECT_ROOT)" -type f -name '.coverage' -delete
	rm -rf "$(PROJECT_ROOT)/htmlcov/"
	rm -f "$(PROJECT_ROOT)/.coverage"

# Clean all generated files including outputs (be careful!)
clean-all: clean
	@echo "Removing all generated outputs..."
	rm -rf "$(OUTPUT_RAW)/*" "$(OUTPUT_PROCESSED)/*" "$(OUTPUT_FIGURES)/*"

.PHONY: help help-paths setup test test-cov clean clean-all
