"""
Python Package Setup Configuration for Plume Navigation Simulation Backend System

Setup configuration file for the plume navigation simulation backend system providing 
comprehensive package metadata, dependency specifications, entry point definitions, and 
installation configuration for scientific computing framework that handles automated 
normalization and calibration of plume recordings, batch simulation execution with 
4000+ simulation capabilities, and cross-platform compatibility with >95% correlation 
validation requirements.

This setup module implements modern Python packaging standards with setuptools integration 
for reproducible scientific computing environments, comprehensive dependency management 
for scientific libraries, and cross-platform installation optimization with performance 
monitoring and scientific reproducibility standards.

Key Features:
- Modern Python Packaging Standards with PEP 517/518 compliance
- Scientific Computing Dependencies Management with precise version constraints  
- Cross-Platform Compatibility Installation with video processing libraries
- Performance Requirements Infrastructure supporting 4000+ simulations within 8 hours
- Scientific Reproducibility Installation with >95% correlation accuracy requirements
- Command-Line Interface Integration with comprehensive CLI tools and workflow execution
- Comprehensive Package Data Management with configuration schemas and examples
- Optional Dependency Groups for development, documentation, performance, and visualization
- Platform-Specific Optimization with scientific computing library compatibility
- Quality Assurance Integration with testing frameworks and validation tools
"""

# External library imports with version specifications for modern Python packaging
import setuptools  # >=65.0.0 - Modern Python packaging framework with PEP 517/518 compliance and advanced features
from pathlib import Path  # 3.9+ - Cross-platform path handling for setup file operations and resource management
import os  # 3.9+ - Operating system interface for environment detection and platform-specific configuration
import sys  # 3.9+ - System-specific parameters and functions for Python version validation and platform detection
import ast  # 3.9+ - Abstract syntax tree parsing for version extraction without importing package
import re  # 3.9+ - Regular expression operations for dependency parsing and validation
import platform  # 3.9+ - Platform identification for cross-platform compatibility assessment
import logging  # 3.9+ - Logging for setup operations and error reporting
from typing import Dict, List, Any, Optional  # 3.9+ - Type hints for setup function interfaces

# Internal imports for package version information
from backend import __version__

# Global configuration constants and paths for package setup
HERE = Path(__file__).parent.resolve()
README_PATH = HERE / 'README.md'
REQUIREMENTS_PATH = HERE / 'requirements.txt'
VERSION_FILE = HERE / 'backend' / '__init__.py'
PACKAGE_NAME = 'plume-simulation-backend'
PACKAGE_VERSION = __version__
AUTHOR = 'Plume Simulation Research Team'
AUTHOR_EMAIL = 'research-team@institution.edu'
DESCRIPTION = 'Scientific computing framework for olfactory navigation algorithm evaluation and cross-format plume data processing'
LONG_DESCRIPTION = None  # Will be set by read_readme()
URL = 'https://github.com/research-team/plume-simulation'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.9'
CLASSIFIERS = None  # Will be set by get_package_classifiers()
KEYWORDS = None  # Will be set by get_package_keywords()
INSTALL_REQUIRES = None  # Will be set by read_requirements()
EXTRAS_REQUIRE = None  # Will be set by get_extras_require()
ENTRY_POINTS = None  # Will be set by get_entry_points()
PACKAGE_DATA = None  # Will be set by get_package_data()
INCLUDE_PACKAGE_DATA = True
ZIP_SAFE = False

# Setup logging for package installation and configuration operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_readme() -> str:
    """
    Read and return the contents of README.md file for long description in package 
    metadata with error handling for missing files and encoding issues to ensure 
    robust package installation across different environments.
    
    This function provides comprehensive README file handling with UTF-8 encoding 
    support, fallback descriptions for missing files, and detailed error logging 
    for scientific computing package installation reliability.
    
    Returns:
        str: Contents of README.md file or fallback description if file not found
        
    Raises:
        None: Function handles all exceptions gracefully with fallback descriptions
    """
    try:
        # Check if README.md file exists in the package directory
        if not README_PATH.exists():
            logger.warning(f"README.md file not found at {README_PATH}")
            return DESCRIPTION + "\n\nFor detailed documentation, please visit the project repository."
        
        # Read README.md file with UTF-8 encoding and error handling
        with open(README_PATH, 'r', encoding='utf-8', errors='replace') as readme_file:
            readme_content = readme_file.read().strip()
            
            # Validate README content length and format
            if len(readme_content) < 50:
                logger.warning(f"README.md file appears too short ({len(readme_content)} characters)")
                return DESCRIPTION + "\n\nPlease refer to the project repository for complete documentation."
            
            logger.info(f"Successfully read README.md file ({len(readme_content)} characters)")
            return readme_content
            
    except UnicodeDecodeError as e:
        logger.warning(f"README.md encoding error: {e}")
        return DESCRIPTION + "\n\nDocumentation encoding issue - please check project repository."
    
    except IOError as e:
        logger.warning(f"README.md I/O error: {e}")
        return DESCRIPTION + "\n\nDocumentation access issue - please check project repository."
    
    except Exception as e:
        logger.warning(f"Unexpected error reading README.md: {e}")
        return DESCRIPTION + "\n\nFor complete documentation, please visit the project repository."


def read_requirements() -> List[str]:
    """
    Read and parse requirements.txt file to extract package dependencies with version 
    constraints for scientific computing accuracy and cross-platform compatibility.
    
    This function provides comprehensive dependency parsing with comment filtering, 
    version constraint validation, and fallback dependency lists for reliable package 
    installation in scientific computing environments.
    
    Returns:
        List[str]: List of package dependencies with version constraints from requirements.txt
        
    Raises:
        None: Function handles all exceptions gracefully with fallback dependencies
    """
    try:
        # Check if requirements.txt file exists in the package directory
        if not REQUIREMENTS_PATH.exists():
            logger.warning(f"requirements.txt file not found at {REQUIREMENTS_PATH}")
            # Return minimal dependencies for basic functionality
            return [
                'numpy>=2.1.3',
                'scipy>=1.15.3', 
                'opencv-python>=4.11.0',
                'pandas>=2.2.0',
                'joblib>=1.6.0',
                'matplotlib>=3.9.0'
            ]
        
        # Read requirements.txt file line by line with UTF-8 encoding
        dependencies = []
        with open(REQUIREMENTS_PATH, 'r', encoding='utf-8', errors='replace') as req_file:
            for line_num, line in enumerate(req_file, 1):
                line = line.strip()
                
                # Filter out comments, empty lines, and development dependencies
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                # Skip development and testing dependencies
                if any(dev_keyword in line.lower() for dev_keyword in ['dev', 'test', 'doc', 'lint']):
                    continue
                
                # Parse package names and version constraints for scientific computing libraries
                if '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line or '~=' in line:
                    # Validate dependency format and version constraint syntax
                    if re.match(r'^[a-zA-Z0-9\-_.]+[<>=~!]+[0-9.]+.*$', line):
                        dependencies.append(line)
                        logger.debug(f"Added dependency: {line}")
                    else:
                        logger.warning(f"Invalid dependency format on line {line_num}: {line}")
                else:
                    # Add package without version constraint (not recommended for scientific computing)
                    if re.match(r'^[a-zA-Z0-9\-_.]+$', line):
                        dependencies.append(line)
                        logger.warning(f"Dependency without version constraint: {line}")
        
        # Validate that core scientific computing dependencies are included
        core_packages = ['numpy', 'scipy', 'opencv', 'pandas', 'joblib']
        found_packages = [dep.split('>=')[0].split('==')[0].lower() for dep in dependencies]
        
        for core_pkg in core_packages:
            if not any(core_pkg in found_pkg for found_pkg in found_packages):
                logger.warning(f"Core scientific package '{core_pkg}' not found in requirements")
        
        logger.info(f"Successfully parsed {len(dependencies)} dependencies from requirements.txt")
        return dependencies
        
    except UnicodeDecodeError as e:
        logger.warning(f"requirements.txt encoding error: {e}")
        # Return core scientific computing dependencies as fallback
        return [
            'numpy>=2.1.3',
            'scipy>=1.15.3',
            'opencv-python>=4.11.0', 
            'pandas>=2.2.0',
            'joblib>=1.6.0',
            'matplotlib>=3.9.0'
        ]
    
    except IOError as e:
        logger.warning(f"requirements.txt I/O error: {e}")
        # Return minimal dependencies for basic functionality
        return [
            'numpy>=2.1.3',
            'scipy>=1.15.3',
            'opencv-python>=4.11.0',
            'pandas>=2.2.0'
        ]
    
    except Exception as e:
        logger.warning(f"Unexpected error reading requirements.txt: {e}")
        # Return essential dependencies for package installation
        return ['numpy>=2.1.3', 'scipy>=1.15.3']


def get_version_from_init() -> str:
    """
    Extract version information from backend/__init__.py file using AST parsing to 
    avoid importing the package during setup for reliable version management without 
    dependency conflicts during installation.
    
    This function provides robust version extraction using abstract syntax tree parsing 
    to ensure version information is available during package installation without 
    requiring package imports that could cause dependency resolution issues.
    
    Returns:
        str: Package version string extracted from __init__.py file
        
    Raises:
        RuntimeError: When version cannot be found or is invalid format
    """
    try:
        # Read backend/__init__.py file with UTF-8 encoding
        if not VERSION_FILE.exists():
            logger.error(f"Version file not found: {VERSION_FILE}")
            raise RuntimeError(f"Version file not found: {VERSION_FILE}")
        
        with open(VERSION_FILE, 'r', encoding='utf-8') as version_file:
            file_content = version_file.read()
        
        # Parse file content using AST to extract __version__ variable
        try:
            tree = ast.parse(file_content)
        except SyntaxError as e:
            logger.error(f"Syntax error in version file: {e}")
            raise RuntimeError(f"Syntax error in version file: {e}")
        
        # Search for __version__ assignment in AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__version__':
                        # Extract version string value from assignment node
                        if isinstance(node.value, ast.Str):
                            version_string = node.value.s
                        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            version_string = node.value.value
                        else:
                            continue
                        
                        # Validate version format using semantic versioning standards
                        if re.match(r'^\d+\.\d+\.\d+([a-zA-Z0-9\-\.]*)?$', version_string):
                            logger.info(f"Successfully extracted version: {version_string}")
                            return version_string
                        else:
                            logger.error(f"Invalid version format: {version_string}")
                            raise RuntimeError(f"Invalid version format: {version_string}")
        
        # Version not found in AST
        logger.error("__version__ variable not found in __init__.py")
        raise RuntimeError("__version__ variable not found in __init__.py")
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        else:
            logger.error(f"Error extracting version from __init__.py: {e}")
            raise RuntimeError(f"Error extracting version from __init__.py: {e}")


def get_package_classifiers() -> List[str]:
    """
    Generate comprehensive package classifiers for PyPI categorization including 
    development status, intended audience, topic classification, and platform 
    compatibility for scientific computing discovery and package management.
    
    This function provides complete PyPI classifier configuration for optimal package 
    discovery, proper categorization in scientific computing context, and compatibility 
    declarations for research computing environments.
    
    Returns:
        List[str]: List of PyPI classifiers for package categorization and discovery
    """
    try:
        # Define comprehensive classifiers for scientific computing package
        classifiers = [
            # Development status classifier as Beta for research software
            'Development Status :: 4 - Beta',
            
            # Intended audience classifiers for Science/Research and Developers
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            
            # Topic classifiers for Scientific/Engineering, Physics, AI, and Mathematics
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Multimedia :: Video :: Analysis',
            
            # License classifier for MIT License compatibility
            'License :: OSI Approved :: MIT License',
            
            # Programming language classifiers for Python 3.9+ support
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10', 
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3 :: Only',
            
            # Operating system classifier for cross-platform compatibility
            'Operating System :: OS Independent',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
            
            # Environment and natural language classifiers
            'Environment :: Console',
            'Environment :: X11 Applications',
            'Natural Language :: English',
            
            # Typing classifier for type hint support
            'Typing :: Typed'
        ]
        
        logger.info(f"Generated {len(classifiers)} package classifiers")
        return classifiers
        
    except Exception as e:
        logger.warning(f"Error generating package classifiers: {e}")
        # Return minimal classifiers for basic functionality
        return [
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering'
        ]


def get_package_keywords() -> List[str]:
    """
    Generate package keywords for search optimization and discovery focusing on 
    scientific computing, plume navigation, algorithm evaluation, and research 
    applications for optimal PyPI discoverability.
    
    This function provides comprehensive keyword generation for scientific computing 
    package discovery, research domain categorization, and technical capability 
    description to enhance package visibility in academic and research contexts.
    
    Returns:
        List[str]: List of keywords for package search optimization and scientific discovery
    """
    try:
        # Generate comprehensive keywords for scientific computing and research discovery
        keywords = [
            # Core domain keywords: plume, navigation, simulation, olfactory
            'plume', 'navigation', 'simulation', 'olfactory',
            'biomimetics', 'bio-inspired', 'robotics',
            
            # Scientific computing keywords: scientific-computing, algorithm-evaluation
            'scientific-computing', 'algorithm-evaluation', 'numerical-analysis',
            'data-analysis', 'computational-biology', 'biophysics',
            
            # Technical keywords: batch-processing, cross-format-compatibility
            'batch-processing', 'cross-format-compatibility', 'video-processing',
            'data-normalization', 'calibration', 'validation',
            
            # Research domain keywords: fluid-dynamics, bio-inspired-robotics
            'fluid-dynamics', 'bio-inspired-robotics', 'sensory-navigation',
            'chemical-plumes', 'odor-tracking', 'environmental-sensing',
            
            # Methodology keywords: performance-analysis, statistical-validation
            'performance-analysis', 'statistical-validation', 'correlation-analysis',
            'comparative-study', 'benchmarking', 'metrics-evaluation',
            
            # Platform keywords: cross-platform, reproducible-research
            'cross-platform', 'reproducible-research', 'open-science',
            'research-tools', 'academic-software', 'scientific-workflow'
        ]
        
        logger.info(f"Generated {len(keywords)} package keywords")
        return keywords
        
    except Exception as e:
        logger.warning(f"Error generating package keywords: {e}")
        # Return minimal keywords for basic discoverability
        return [
            'plume', 'navigation', 'simulation', 'scientific-computing',
            'algorithm-evaluation', 'research-tools'
        ]


def get_extras_require() -> Dict[str, List[str]]:
    """
    Define optional dependency groups for different use cases including development 
    tools, documentation generation, performance optimization, visualization, and 
    scientific computing extensions for flexible installation options.
    
    This function provides comprehensive optional dependency management with specialized 
    groups for development workflows, documentation generation, performance optimization, 
    advanced visualization, and extended scientific computing capabilities.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping extra names to dependency lists for optional features
    """
    try:
        # Define comprehensive optional dependency groups for different use cases
        extras_require = {
            # Development extra with testing, linting, and development tools
            'dev': [
                'pytest>=8.3.5',
                'pytest-cov>=5.0.0',
                'black>=24.10.0',
                'isort>=5.13.0',
                'mypy>=1.13.0',
                'flake8>=7.1.0',
                'pre-commit>=4.0.0',
                'twine>=5.1.0',
                'build>=1.2.0'
            ],
            
            # Documentation extra with Sphinx and documentation generation tools
            'docs': [
                'sphinx>=8.1.0',
                'sphinx-rtd-theme>=3.0.0',
                'sphinx-autodoc-typehints>=2.4.0',
                'myst-parser>=4.0.0',
                'sphinxcontrib-napoleon>=0.7',
                'nbsphinx>=0.9.0'
            ],
            
            # Performance extra with Numba, Cython, and profiling tools
            'performance': [
                'numba>=0.60.0',
                'cython>=3.0.0',
                'memory-profiler>=0.61.0',
                'line-profiler>=4.1.0',
                'psutil>=6.1.0',
                'py-spy>=0.3.0'
            ],
            
            # Visualization extra with Plotly, Bokeh, and interactive plotting
            'visualization': [
                'plotly>=5.24.0',
                'bokeh>=3.6.0',
                'seaborn>=0.13.2',
                'ipywidgets>=8.1.0',
                'jupyter>=1.1.0',
                'notebook>=7.3.0'
            ],
            
            # Scientific extra with extended scientific data format support
            'scientific': [
                'zarr>=2.18.0',
                'xarray>=2024.11.0',
                'netcdf4>=1.7.0',
                'h5py>=3.12.0',
                'scikit-image>=0.24.0',
                'scikit-learn>=1.6.0'
            ],
            
            # All extra combining all optional dependency groups
            'all': []  # Will be populated below
        }
        
        # Populate 'all' extra with all dependencies from other groups
        all_deps = set()
        for group_name, deps in extras_require.items():
            if group_name != 'all':
                all_deps.update(deps)
        
        extras_require['all'] = sorted(list(all_deps))
        
        logger.info(f"Generated {len(extras_require)} optional dependency groups")
        return extras_require
        
    except Exception as e:
        logger.warning(f"Error generating optional dependencies: {e}")
        # Return minimal optional dependencies
        return {
            'dev': ['pytest>=8.3.5', 'black>=24.10.0'],
            'docs': ['sphinx>=8.1.0'],
            'all': ['pytest>=8.3.5', 'black>=24.10.0', 'sphinx>=8.1.0']
        }


def get_entry_points() -> Dict[str, List[str]]:
    """
    Define console script entry points for command-line interface tools including 
    main simulation interface, batch processing, validation, reporting, and cache 
    management utilities for comprehensive CLI workflow support.
    
    This function provides complete command-line interface configuration with specialized 
    entry points for different workflow stages, system management operations, and 
    administrative tasks for scientific computing environments.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping entry point groups to script definitions for CLI tools
    """
    try:
        # Define comprehensive console script entry points for CLI tools
        entry_points = {
            'console_scripts': [
                # Main simulation interface entry point for comprehensive workflow execution
                'plume-simulation=backend.cli:main',
                
                # Batch processing entry point for large-scale simulation operations
                'plume-batch=backend.scripts.run_batch_simulation:main',
                
                # Environment validation entry point for system requirements verification
                'plume-validate=backend.scripts.validate_environment:main',
                
                # Report generation entry point for comprehensive analysis reporting
                'plume-report=backend.scripts.generate_report:main',
                
                # Cache management entry point for system cleanup and optimization
                'plume-clean=backend.scripts.clean_cache:main'
            ]
        }
        
        logger.info(f"Generated {len(entry_points['console_scripts'])} console script entry points")
        return entry_points
        
    except Exception as e:
        logger.warning(f"Error generating entry points: {e}")
        # Return minimal entry points for basic functionality
        return {
            'console_scripts': [
                'plume-simulation=backend.cli:main'
            ]
        }


def get_package_data() -> Dict[str, List[str]]:
    """
    Define package data files to include in distribution including configuration 
    schemas, example data, shell scripts, and type information for complete 
    functionality and scientific computing support.
    
    This function provides comprehensive package data configuration with schema files, 
    example datasets, configuration templates, and type information to ensure complete 
    package functionality and scientific reproducibility.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping package names to data file patterns for inclusion
    """
    try:
        # Define comprehensive package data for complete functionality
        package_data = {
            'backend': [
                # Configuration JSON files from config/ directory
                'config/*.json',
                'config/*.yaml',
                'config/*.toml',
                
                # Schema validation files from config/schema/ directory
                'config/schema/*.json',
                'config/schema/*.yaml',
                
                # Example data files from examples/data/ directory
                'examples/data/*.avi',
                'examples/data/*.mp4',
                'examples/data/*.json',
                'examples/data/*.csv',
                
                # Shell scripts from scripts/ directory
                'scripts/*.sh',
                'scripts/*.bat',
                'scripts/*.ps1',
                
                # Type information file (py.typed) for type checking support
                'py.typed',
                
                # Documentation and template files
                'templates/*.json',
                'templates/*.yaml',
                'docs/*.md',
                'docs/*.rst'
            ]
        }
        
        logger.info(f"Generated package data configuration for {len(package_data)} packages")
        return package_data
        
    except Exception as e:
        logger.warning(f"Error generating package data: {e}")
        # Return minimal package data
        return {
            'backend': ['config/*.json', 'py.typed']
        }


def validate_python_version() -> bool:
    """
    Validate that the current Python version meets minimum requirements for scientific 
    computing libraries and package functionality with clear error messages for 
    unsupported versions and upgrade recommendations.
    
    This function provides comprehensive Python version validation with detailed error 
    reporting, upgrade guidance, and compatibility information for scientific computing 
    environments and research reproducibility.
    
    Returns:
        bool: True if Python version is supported, raises SystemExit if not
        
    Raises:
        SystemExit: When Python version is unsupported with detailed error information
    """
    try:
        # Get current Python version from sys.version_info
        current_version = sys.version_info
        required_major = 3
        required_minor = 9
        
        logger.info(f"Validating Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
        
        # Check if version is >= 3.9 for scientific computing compatibility
        if current_version.major < required_major or (current_version.major == required_major and current_version.minor < required_minor):
            
            # Print clear error message if version is unsupported
            error_message = f"""
ERROR: Python {current_version.major}.{current_version.minor}.{current_version.micro} is not supported.

This package requires Python {required_major}.{required_minor}+ for scientific computing compatibility.

Reasons for version requirement:
- NumPy 2.1.3+ requires Python 3.9+ for optimal performance
- SciPy 1.15.3+ requires Python 3.9+ for scientific computing functions  
- OpenCV 4.11.0+ requires Python 3.9+ for video processing capabilities
- Modern type hints and performance optimizations require Python 3.9+

Upgrade Instructions:
- Update Python to version {required_major}.{required_minor} or higher
- For conda users: conda install python>=3.9
- For system package managers: consult your distribution documentation
- For pyenv users: pyenv install 3.9.20 && pyenv global 3.9.20

For more information, visit: https://www.python.org/downloads/
"""
            
            print(error_message, file=sys.stderr)
            
            # Raise SystemExit with error code if version check fails
            raise SystemExit(1)
        
        # Validate version compatibility with NumPy, SciPy, and OpenCV requirements
        if current_version.major == 3 and current_version.minor == 9:
            logger.info("Python 3.9 detected - ensuring compatibility with scientific libraries")
        elif current_version.major == 3 and current_version.minor >= 10:
            logger.info(f"Python 3.{current_version.minor} detected - full compatibility with scientific libraries")
        
        logger.info("Python version validation passed successfully")
        return True
        
    except SystemExit:
        raise
    except Exception as e:
        error_message = f"Error during Python version validation: {e}"
        print(error_message, file=sys.stderr)
        raise SystemExit(1)


def check_platform_compatibility() -> Dict[str, Any]:
    """
    Check platform compatibility for scientific computing dependencies and provide 
    platform-specific installation guidance for optimal performance and functionality 
    with recommendations for library optimization.
    
    This function provides comprehensive platform assessment with scientific computing 
    library compatibility checks, performance optimization recommendations, and 
    platform-specific installation guidance for research environments.
    
    Returns:
        Dict[str, Any]: Platform compatibility information with recommendations and warnings
    """
    try:
        # Detect current platform using sys.platform and platform module
        platform_info = {
            'system': platform.system(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_implementation': platform.python_implementation(),
            'python_version': platform.python_version(),
            'compatibility_status': 'unknown',
            'recommendations': [],
            'warnings': [],
            'optimization_opportunities': []
        }
        
        logger.info(f"Checking platform compatibility: {platform_info['system']} {platform_info['machine']}")
        
        # Check compatibility with OpenCV and video processing libraries
        if platform_info['system'] == 'Linux':
            platform_info['compatibility_status'] = 'excellent'
            platform_info['recommendations'].extend([
                'Install system OpenCV development libraries for optimal performance',
                'Consider using conda for scientific library management',
                'Install BLAS/LAPACK libraries for NumPy/SciPy optimization'
            ])
            
            # Check for platform-specific optimization opportunities
            platform_info['optimization_opportunities'].extend([
                'Use Intel MKL for numerical computation acceleration',
                'Configure OpenMP for parallel processing optimization',
                'Install FFmpeg for enhanced video format support'
            ])
            
        elif platform_info['system'] == 'Darwin':  # macOS
            platform_info['compatibility_status'] = 'good'
            platform_info['recommendations'].extend([
                'Install Homebrew for scientific library management',
                'Use conda or pip for Python scientific packages',
                'Install Xcode command line tools for compilation support'
            ])
            
            # macOS specific optimization
            if platform_info['machine'] == 'arm64':  # Apple Silicon
                platform_info['optimization_opportunities'].extend([
                    'Use Apple Silicon optimized NumPy/SciPy builds',
                    'Consider native ARM64 scientific libraries for performance'
                ])
                platform_info['warnings'].append('Apple Silicon: Verify OpenCV ARM64 compatibility')
            else:
                platform_info['optimization_opportunities'].append('Use Intel MKL for x86_64 optimization')
            
        elif platform_info['system'] == 'Windows':
            platform_info['compatibility_status'] = 'good'
            platform_info['recommendations'].extend([
                'Use conda for simplified scientific library installation',
                'Install Microsoft Visual C++ Redistributable for compiled libraries',
                'Consider Windows Subsystem for Linux (WSL) for enhanced compatibility'
            ])
            
            platform_info['warnings'].extend([
                'Windows: Some scientific libraries may require additional configuration',
                'Video processing libraries may need manual codec installation'
            ])
            
            platform_info['optimization_opportunities'].extend([
                'Use Intel MKL for mathematical library acceleration',
                'Install Intel oneAPI for optimized scientific computing'
            ])
            
        else:
            platform_info['compatibility_status'] = 'limited'
            platform_info['warnings'].append(f'Platform {platform_info["system"]} has limited testing coverage')
            platform_info['recommendations'].append('Manual testing recommended for full functionality verification')
        
        # Validate NumPy and SciPy binary availability for platform
        if platform_info['machine'] in ['x86_64', 'AMD64', 'arm64', 'aarch64']:
            platform_info['recommendations'].append('Binary wheels available for fast installation')
        else:
            platform_info['warnings'].append(f'Architecture {platform_info["machine"]}: May require compilation from source')
        
        # Check for platform-specific limitations
        if platform_info['system'] == 'Windows' and platform_info['python_implementation'] != 'CPython':
            platform_info['warnings'].append('Non-CPython implementations may have scientific library limitations on Windows')
        
        logger.info(f"Platform compatibility check completed: {platform_info['compatibility_status']}")
        return platform_info
        
    except Exception as e:
        logger.warning(f"Platform compatibility check failed: {e}")
        return {
            'system': 'unknown',
            'compatibility_status': 'unknown',
            'error': str(e),
            'recommendations': ['Manual compatibility verification recommended'],
            'warnings': ['Platform compatibility check failed - proceed with caution']
        }


def setup_package() -> None:
    """
    Main setup function that configures and installs the plume simulation backend 
    package with comprehensive metadata, dependencies, and entry points for scientific 
    computing research with performance optimization and reproducibility standards.
    
    This function provides complete package configuration with scientific computing 
    dependencies, cross-platform compatibility, performance optimization, and 
    comprehensive CLI tool integration for research computing environments.
    
    Returns:
        None: Executes setuptools.setup() with complete package configuration
        
    Raises:
        SystemExit: When setup fails due to incompatible environment or configuration errors
    """
    try:
        logger.info(f"Starting package setup for {PACKAGE_NAME} version {PACKAGE_VERSION}")
        
        # Validate Python version compatibility for scientific computing
        validate_python_version()
        
        # Check platform compatibility and generate recommendations
        platform_info = check_platform_compatibility()
        
        if platform_info['compatibility_status'] == 'limited':
            logger.warning("Limited platform compatibility detected - some features may not work optimally")
        
        if platform_info.get('warnings'):
            for warning in platform_info['warnings']:
                logger.warning(f"Platform warning: {warning}")
        
        # Read package metadata from README and version files
        global LONG_DESCRIPTION, CLASSIFIERS, KEYWORDS, INSTALL_REQUIRES, EXTRAS_REQUIRE, ENTRY_POINTS, PACKAGE_DATA
        
        LONG_DESCRIPTION = read_readme()
        CLASSIFIERS = get_package_classifiers()
        KEYWORDS = get_package_keywords()
        INSTALL_REQUIRES = read_requirements()
        EXTRAS_REQUIRE = get_extras_require()
        ENTRY_POINTS = get_entry_points()
        PACKAGE_DATA = get_package_data()
        
        # Load dependencies from requirements.txt with validation
        logger.info(f"Core dependencies: {len(INSTALL_REQUIRES)} packages")
        logger.info(f"Optional dependency groups: {len(EXTRAS_REQUIRE)} groups")
        logger.info(f"Console script entry points: {len(ENTRY_POINTS.get('console_scripts', []))} commands")
        
        # Configure package classifiers and keywords for discovery
        logger.info(f"Package classifiers: {len(CLASSIFIERS)} classifiers for PyPI categorization")
        logger.info(f"Search keywords: {len(KEYWORDS)} keywords for package discovery")
        
        # Setup optional dependency groups for different use cases
        for group_name, group_deps in EXTRAS_REQUIRE.items():
            logger.info(f"Optional '{group_name}' group: {len(group_deps)} dependencies")
        
        # Configure console script entry points for CLI tools
        cli_commands = ENTRY_POINTS.get('console_scripts', [])
        for cmd in cli_commands:
            command_name = cmd.split('=')[0]
            logger.info(f"CLI command: {command_name}")
        
        # Setup package data inclusion for complete functionality
        total_data_patterns = sum(len(patterns) for patterns in PACKAGE_DATA.values())
        logger.info(f"Package data patterns: {total_data_patterns} file patterns included")
        
        # Execute setuptools.setup() with comprehensive configuration
        setuptools.setup(
            # Basic package metadata
            name=PACKAGE_NAME,
            version=PACKAGE_VERSION,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            long_description_content_type='text/markdown',
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            url=URL,
            license=LICENSE,
            
            # Python version requirements
            python_requires=PYTHON_REQUIRES,
            
            # Package discovery and structure
            packages=setuptools.find_packages(include=['backend', 'backend.*']),
            package_dir={'': '.'},
            include_package_data=INCLUDE_PACKAGE_DATA,
            package_data=PACKAGE_DATA,
            zip_safe=ZIP_SAFE,
            
            # Dependencies and requirements
            install_requires=INSTALL_REQUIRES,
            extras_require=EXTRAS_REQUIRE,
            
            # Entry points and console scripts
            entry_points=ENTRY_POINTS,
            
            # Package classification and discovery
            classifiers=CLASSIFIERS,
            keywords=KEYWORDS,
            
            # Additional metadata for scientific computing
            project_urls={
                'Bug Reports': f'{URL}/issues',
                'Source': URL,
                'Documentation': f'{URL}/docs',
                'Research': f'{URL}/research'
            }
        )
        
        # Log successful setup completion with installation summary
        setup_summary = {
            'package_name': PACKAGE_NAME,
            'version': PACKAGE_VERSION,
            'python_requires': PYTHON_REQUIRES,
            'core_dependencies': len(INSTALL_REQUIRES),
            'optional_groups': len(EXTRAS_REQUIRE),
            'cli_commands': len(ENTRY_POINTS.get('console_scripts', [])),
            'classifiers': len(CLASSIFIERS),
            'keywords': len(KEYWORDS),
            'platform_compatibility': platform_info['compatibility_status']
        }
        
        logger.info(f"Package setup completed successfully: {setup_summary}")
        
        # Display platform-specific recommendations
        if platform_info.get('recommendations'):
            logger.info("Platform-specific recommendations:")
            for recommendation in platform_info['recommendations']:
                logger.info(f"  - {recommendation}")
        
        if platform_info.get('optimization_opportunities'):
            logger.info("Performance optimization opportunities:")
            for optimization in platform_info['optimization_opportunities']:
                logger.info(f"  - {optimization}")
        
    except SystemExit:
        raise
    except Exception as e:
        error_message = f"Package setup failed: {e}"
        logger.error(error_message, exc_info=True)
        print(f"ERROR: {error_message}", file=sys.stderr)
        raise SystemExit(1)


# Execute package setup when script is run directly
if __name__ == '__main__':
    setup_package()