#!/usr/bin/env python3
"""
Plume Navigation Simulation Testing Framework Setup Configuration

This setup.py file configures the comprehensive testing framework for scientific
plume navigation simulation validation with cross-format compatibility, performance
benchmarking, and statistical analysis capabilities.

Features:
- Scientific computing test dependencies with >95% correlation accuracy requirements
- Performance validation with <7.2 seconds per simulation targets
- Comprehensive error handling across 4000+ batch simulation testing scenarios
- Cross-format compatibility testing for Crimaldi and custom plume formats
- Mock data generation and statistical validation frameworks
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Package metadata constants
PACKAGE_NAME = 'plume-simulation-test-framework'
PACKAGE_VERSION = '1.0.0'
PACKAGE_AUTHOR = 'Plume Navigation Test Framework Team'
PACKAGE_EMAIL = 'test-framework@institution.edu'
PACKAGE_DESCRIPTION = 'Comprehensive testing framework for scientific plume navigation simulation validation with cross-format compatibility and performance benchmarking'
PACKAGE_URL = 'https://github.com/research-team/plume-simulation/tree/main/src/test'
PYTHON_REQUIRES = '>=3.9'
SUPPORTED_PYTHON_VERSIONS = ['3.9', '3.10', '3.11', '3.12']

# Core testing and scientific computing dependencies
INSTALL_REQUIRES = [
    'pytest>=8.3.5',
    'pytest-cov>=5.0.0',
    'pytest-xdist>=3.6.0',
    'pytest-benchmark>=4.0.0',
    'pytest-mock>=3.12.0',
    'numpy>=2.1.3',
    'scipy>=1.15.3',
    'pandas>=2.2.0',
    'opencv-python>=4.11.0',
    'matplotlib>=3.9.0',
    'seaborn>=0.13.2',
    'statsmodels>=0.14.0',
    'hypothesis>=6.112.0',
    'factory-boy>=3.3.0',
    'faker>=26.0.0',
    'psutil>=5.9.0',
    'memory-profiler>=0.61.0',
    'jsonschema>=4.23.0',
    'tqdm>=4.65.0',
    'rich>=13.7.0'
]

# Optional dependency groups for enhanced functionality
EXTRAS_REQUIRE = {
    'dev': [
        'black>=24.0.0',
        'flake8>=7.0.0',
        'mypy>=1.11.0',
        'pre-commit>=3.0.0',
        'sphinx>=6.0.0',
        'sphinx-rtd-theme>=1.2.0'
    ],
    'performance': [
        'line-profiler>=4.1.0',
        'py-spy>=0.3.14',
        'numba>=0.58.0',
        'joblib>=1.6.0'
    ],
    'visualization': [
        'plotly>=5.17.0',
        'bokeh>=3.2.0',
        'ipywidgets>=8.0.0'
    ],
    'mock': [
        'responses>=0.25.0',
        'requests-mock>=1.11.0',
        'multiprocess>=0.70.16'
    ]
}

# Console script entry points for CLI tools
ENTRY_POINTS = {
    'console_scripts': [
        'plume-test=test.cli:main',
        'plume-test-validate=test.scripts.validate_test_environment:main',
        'plume-test-benchmark=test.scripts.run_performance_benchmarks:main',
        'plume-test-report=test.scripts.generate_test_report:main'
    ]
}

# PyPI classifiers for package categorization
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Testing',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Operating System :: OS Independent',
    'Environment :: Console',
    'Natural Language :: English'
]

# Keywords for package discovery
KEYWORDS = ['testing', 'plume-navigation', 'scientific-computing', 'cross-format-compatibility', 
           'performance-benchmarking', 'mock-testing', 'statistical-validation', 'pytest']

# Project URLs for documentation and source code
PROJECT_URLS = {
    'Documentation': 'https://plume-simulation-test.readthedocs.io/',
    'Source': 'https://github.com/research-team/plume-simulation/tree/main/src/test',
    'Tracker': 'https://github.com/research-team/plume-simulation/issues',
    'Test Reports': 'https://github.com/research-team/plume-simulation/actions'
}


def read_test_requirements(requirements_file='requirements.txt'):
    """
    Read and parse test requirements.txt file to extract testing package dependencies
    with version constraints for reproducible scientific computing test environments.
    
    Args:
        requirements_file (str): Path to requirements.txt file
        
    Returns:
        List[str]: List of test package requirements with version specifications
    """
    try:
        # Check if test requirements.txt file exists in the test package directory
        req_path = Path(__file__).parent / requirements_file
        if not req_path.exists():
            print(f"Warning: Requirements file {requirements_file} not found, using default dependencies")
            return INSTALL_REQUIRES
            
        # Read the requirements file line by line with UTF-8 encoding
        with open(req_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Filter out comments, empty lines, and development dependencies
        requirements = []
        for line in lines:
            # Strip whitespace and normalize requirement strings for testing packages
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Validate requirement format and version constraints for scientific computing
                if '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line:
                    requirements.append(line)
                else:
                    # Add default version constraint if none specified
                    requirements.append(f"{line}>=0.0.0")
        
        # Separate core testing requirements from optional performance dependencies
        core_requirements = [req for req in requirements if any(
            core_pkg in req for core_pkg in ['pytest', 'numpy', 'scipy', 'opencv', 'pandas']
        )]
        
        # Return list of clean requirement specifications for test framework
        return requirements if requirements else INSTALL_REQUIRES
        
    except (IOError, OSError) as e:
        print(f"Warning: Could not read requirements file {requirements_file}: {e}")
        return INSTALL_REQUIRES


def read_test_long_description(readme_file='README.md'):
    """
    Read and format the long description from test README.md file for test package
    metadata with proper markdown handling and testing framework documentation.
    
    Args:
        readme_file (str): Path to README.md file
        
    Returns:
        str: Formatted long description content for test package metadata
    """
    try:
        # Check if test README.md file exists in the test package directory
        readme_path = Path(__file__).parent / readme_file
        if not readme_path.exists():
            # Provide fallback description emphasizing testing framework capabilities
            return (
                "# Plume Navigation Simulation Testing Framework\n\n"
                "Comprehensive testing framework for scientific plume navigation simulation "
                "validation with cross-format compatibility and performance benchmarking.\n\n"
                "## Key Features\n\n"
                "- >95% correlation accuracy with reference implementations\n"
                "- <7.2 seconds per simulation performance validation\n"
                "- Cross-format compatibility testing for Crimaldi and custom plume formats\n"
                "- Comprehensive error handling across 4000+ batch simulation scenarios\n"
                "- Statistical validation and mock data generation frameworks\n\n"
                "## Testing Methodology\n\n"
                "The framework provides systematic validation processes for scientific "
                "computing environments with emphasis on numerical precision, reproducibility, "
                "and cross-platform compatibility for plume navigation research."
            )
            
        # Read the README.md file content with UTF-8 encoding
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean and format markdown content for PyPI display with testing focus
        # Handle any encoding issues or file read errors gracefully
        if content.strip():
            return content
        else:
            # Include testing methodology and scientific validation information
            return (
                "Comprehensive testing framework for plume navigation simulation validation "
                "with scientific computing dependencies and performance benchmarking capabilities."
            )
            
    except (IOError, OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read README file {readme_file}: {e}")
        # Return formatted long description string for test package
        return (
            "Testing framework for scientific plume navigation simulation with cross-format "
            "compatibility, performance validation, and statistical analysis capabilities."
        )


def get_test_package_version():
    """
    Get test package version using hardcoded fallback version for consistent
    version management across the testing framework.
    
    Returns:
        str: Test package version string
    """
    try:
        # Use hardcoded version string as primary source
        version = PACKAGE_VERSION
        
        # Validate version string format using semantic versioning
        version_parts = version.split('.')
        if len(version_parts) >= 3 and all(part.isdigit() for part in version_parts[:3]):
            # Log version extraction process for debugging test setup
            print(f"Using test framework version: {version}")
            
            # Ensure version compatibility with main backend package
            # Return validated version string for test framework
            return version
        else:
            raise ValueError(f"Invalid version format: {version}")
            
    except Exception as e:
        print(f"Warning: Version validation failed: {e}, using fallback")
        return '1.0.0'


def validate_test_python_version():
    """
    Validate that the current Python version meets the minimum requirements for
    the scientific computing test dependencies and testing framework functionality.
    
    Returns:
        bool: True if Python version is compatible with testing framework, False otherwise
    """
    try:
        # Get current Python version from sys.version_info
        current_version = sys.version_info
        
        # Check against minimum required Python version (3.9+) for testing
        min_major, min_minor = 3, 9
        current_major, current_minor = current_version.major, current_version.minor
        
        # Validate compatibility with scientific computing test libraries
        if current_major < min_major or (current_major == min_major and current_minor < min_minor):
            print(f"Error: Python {current_major}.{current_minor} is not supported. "
                  f"Minimum required: {min_major}.{min_minor}")
            return False
            
        # Check for known incompatible Python versions with pytest framework
        incompatible_versions = []  # Currently none for supported range
        version_tuple = (current_major, current_minor)
        
        if version_tuple in incompatible_versions:
            print(f"Error: Python {current_major}.{current_minor} has known compatibility issues")
            return False
            
        # Validate compatibility with performance monitoring tools
        supported_versions = [(3, 9), (3, 10), (3, 11), (3, 12)]
        if version_tuple not in supported_versions:
            print(f"Warning: Python {current_major}.{current_minor} compatibility not fully validated")
            
        # Log validation results for debugging test environment setup
        print(f"Python version {current_major}.{current_minor} validated for test framework")
        
        # Return compatibility status for test framework installation
        return True
        
    except Exception as e:
        print(f"Warning: Python version validation failed: {e}")
        return True  # Allow installation attempt


def setup_test_package_data():
    """
    Configure test package data including test fixtures, configuration files,
    mock data, and reference benchmarks for proper installation and distribution.
    
    Returns:
        Dict[str, List[str]]: Test package data configuration dictionary
    """
    # Define test fixture files to include in package (crimaldi_sample.avi, custom_sample.avi)
    test_fixtures = [
        'test_fixtures/**/*',
        'test_fixtures/crimaldi_sample.avi',
        'test_fixtures/custom_sample.avi',
        'test_fixtures/reference_data/*.npy',
        'test_fixtures/reference_data/*.json'
    ]
    
    # Specify test configuration files for algorithm, normalization, and simulation testing
    config_files = [
        'test_fixtures/config/*.json',
        'test_fixtures/config/algorithm_configs/*.yaml',
        'test_fixtures/config/normalization_configs/*.json',
        'test_fixtures/config/simulation_configs/*.yaml'
    ]
    
    # Include reference benchmark data files for validation testing
    reference_data = [
        'test_fixtures/reference_results/*.npy',
        'test_fixtures/reference_results/*.csv',
        'test_fixtures/benchmarks/*.json',
        'test_fixtures/validation_data/*.pkl'
    ]
    
    # Add mock data generation scripts and utilities
    mock_data = [
        'mocks/**/*.py',
        'mocks/data_generators/*.py',
        'mocks/simulation_mocks/*.py',
        'mocks/fixtures/*.json'
    ]
    
    # Configure test utility modules and helper functions
    utility_modules = [
        'utils/**/*.py',
        'utils/test_helpers/*.py',
        'utils/validation/*.py',
        'utils/performance/*.py'
    ]
    
    # Include pytest configuration files and test discovery settings
    pytest_configs = [
        'pytest.ini',
        'conftest.py',
        'pyproject.toml',
        'tox.ini'
    ]
    
    # Return comprehensive test package data dictionary
    package_data = {
        'test': (test_fixtures + config_files + reference_data + 
                mock_data + utility_modules + pytest_configs)
    }
    
    return package_data


def validate_test_dependencies(requirements_list):
    """
    Validate that all required test dependencies are compatible and meet the
    scientific computing requirements for the testing framework.
    
    Args:
        requirements_list (List[str]): List of package requirements
        
    Returns:
        bool: True if all test dependencies are valid and compatible, False otherwise
    """
    try:
        # Parse each requirement string and extract package name and version constraints
        validated_packages = []
        critical_packages = {
            'pytest': '8.3.5',
            'numpy': '2.1.3',
            'scipy': '1.15.3',
            'opencv-python': '4.11.0',
            'pandas': '2.2.0'
        }
        
        for requirement in requirements_list:
            # Extract package name from requirement string
            if '>=' in requirement:
                package_name = requirement.split('>=')[0].strip()
                min_version = requirement.split('>=')[1].strip()
            elif '==' in requirement:
                package_name = requirement.split('==')[0].strip()
                min_version = requirement.split('==')[1].strip()
            else:
                package_name = requirement.strip()
                min_version = None
                
            # Validate that pytest version supports required features for scientific testing
            if package_name == 'pytest' and min_version:
                if not min_version.startswith('8.'):
                    print(f"Warning: pytest version {min_version} may not support all required features")
                    
            # Check NumPy and SciPy compatibility for numerical precision requirements
            if package_name in ['numpy', 'scipy'] and min_version:
                major_version = min_version.split('.')[0]
                if package_name == 'numpy' and int(major_version) < 2:
                    print(f"Warning: NumPy version {min_version} may have precision limitations")
                    
            # Validate OpenCV version for video processing test capabilities
            if package_name == 'opencv-python' and min_version:
                major_version = min_version.split('.')[0]
                if int(major_version) < 4:
                    print(f"Warning: OpenCV version {min_version} may lack required video features")
                    
            validated_packages.append(package_name)
            
        # Check statistical analysis library versions for >95% correlation testing
        required_stats_packages = ['scipy', 'statsmodels', 'pandas']
        missing_stats = [pkg for pkg in required_stats_packages if pkg not in validated_packages]
        if missing_stats:
            print(f"Warning: Missing statistical packages: {missing_stats}")
            
        # Validate performance monitoring tool compatibility
        performance_packages = ['psutil', 'memory-profiler']
        missing_perf = [pkg for pkg in performance_packages if pkg not in validated_packages]
        if missing_perf:
            print(f"Warning: Missing performance monitoring packages: {missing_perf}")
            
        # Log any dependency conflicts or compatibility issues
        print(f"Validated {len(validated_packages)} test dependencies")
        
        # Return overall dependency validation status
        return len(missing_stats) == 0 and len(missing_perf) == 0
        
    except Exception as e:
        print(f"Warning: Dependency validation failed: {e}")
        return True  # Allow installation attempt


def create_test_entry_points():
    """
    Create console script entry points for test framework command-line utilities
    including validation, benchmarking, and reporting tools.
    
    Returns:
        Dict[str, List[str]]: Entry points configuration for test framework CLI tools
    """
    # Define main test runner entry point for comprehensive test execution
    main_entry = 'plume-test=test.cli:main'
    
    # Create test environment validation entry point for setup verification
    validate_entry = 'plume-test-validate=test.scripts.validate_test_environment:main'
    
    # Configure performance benchmark entry point for <7.2 seconds validation
    benchmark_entry = 'plume-test-benchmark=test.scripts.run_performance_benchmarks:main'
    
    # Set up test report generation entry point for scientific documentation
    report_entry = 'plume-test-report=test.scripts.generate_test_report:main'
    
    # Create cross-format compatibility testing entry point
    compatibility_entry = 'plume-test-compat=test.scripts.cross_format_validation:main'
    
    # Configure batch test execution entry point for 4000+ simulation testing
    batch_entry = 'plume-test-batch=test.scripts.batch_simulation_runner:main'
    
    # Return complete entry points configuration dictionary
    entry_points = {
        'console_scripts': [
            main_entry,
            validate_entry,
            benchmark_entry,
            report_entry,
            compatibility_entry,
            batch_entry
        ]
    }
    
    return entry_points


# Main setup configuration execution
if __name__ == '__main__':
    # Validate Python version compatibility before proceeding
    if not validate_test_python_version():
        sys.exit(1)
        
    # Read test requirements and validate dependencies
    requirements = read_test_requirements()
    if not validate_test_dependencies(requirements):
        print("Warning: Some dependency validation checks failed, proceeding with installation")
        
    # Get package version and read long description
    version = get_test_package_version()
    long_description = read_test_long_description()
    
    # Configure test package data and entry points
    package_data = setup_test_package_data()
    entry_points = create_test_entry_points()
    
    # Execute setup with comprehensive test framework configuration
    setup(
        # Package metadata
        name=PACKAGE_NAME,
        version=version,
        author=PACKAGE_AUTHOR,
        author_email=PACKAGE_EMAIL,
        description=PACKAGE_DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        url=PACKAGE_URL,
        project_urls=PROJECT_URLS,
        
        # Package discovery and structure
        packages=find_packages(),
        package_dir={'': '.'},
        package_data=package_data,
        include_package_data=True,
        zip_safe=False,
        
        # Python version requirements
        python_requires=PYTHON_REQUIRES,
        
        # Dependency management
        install_requires=requirements,
        extras_require=EXTRAS_REQUIRE,
        setup_requires=['setuptools>=65.0.0', 'wheel>=0.37.0'],
        
        # Entry points for CLI tools
        entry_points=entry_points,
        
        # Package classification
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        license='MIT',
        platforms='any',
        
        # Test configuration
        test_suite='test',
        tests_require=requirements + EXTRAS_REQUIRE.get('dev', []),
    )