.PHONY: help test test-cov lint format check-format type-check clean

# Project configuration
VENV_PATH := $(shell pwd)/dev_env
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip
PYTEST := $(VENV_PATH)/bin/pytest
MYPY := $(VENV_PATH)/mypy
BLACK := $(VENV_PATH)/black
ISORT := $(VENV_PATH)/isort
FLAKE8 := $(VENV_PATH)/flake8

# Help target to show available commands
help:
	@echo "Available targets:"
	@echo "  make test       - Run tests with pytest"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make lint       - Run all linters (black, isort, flake8, mypy)"
	@echo "  make format     - Format code with black and isort"
	@echo "  make check-format - Check code formatting without making changes"
	@echo "  make type-check - Run type checking with mypy"
	@echo "  make clean      - Clean up temporary files"

# Test targets
test:
	$(PYTEST) -v -s

test-cov:
	$(PYTEST) --cov=src --cov-report=term-missing -v

# Linting and formatting targets
lint: check-format type-check
	@echo "Running flake8..."
	$(FLAKE8) src tests

format:
	@echo "Running isort..."
	$(ISORT) src tests
	@echo "Running black..."
	$(BLACK) src tests

check-format:
	@echo "Running isort check..."
	$(ISORT) --check-only src tests
	@echo "Running black check..."
	$(BLACK) --check src tests

type-check:
	@echo "Running mypy..."
	$(MYPY) src tests

# Cleanup
clean:
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '.pytest_cache' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '.coverage' -delete
	@find . -type d -name 'htmlcov' -exec rm -rf {} +
