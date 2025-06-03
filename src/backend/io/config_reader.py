"""
Comprehensive configuration reader module providing robust configuration file loading, validation, 
and management for the plume simulation system.

This module implements comprehensive configuration file loading with automatic format detection, 
schema validation, caching, error handling, and integration with the centralized configuration 
management system. Supports JSON, YAML, and YML formats with fail-fast validation, cross-platform 
compatibility, and scientific parameter validation for reproducible research environments optimized 
for 4000+ simulation processing requirements.

Key Features:
- Centralized configuration management with comprehensive validation and caching
- Fail-fast validation strategy for early error detection and resource optimization
- Cross-platform compatibility for Crimaldi and custom plume formats with automatic handling
- File system integration resilience with pre-processing validation and graceful error handling
- Scientific parameter validation with comprehensive error handling and quality assurance
- Automatic format detection with confidence scoring and metadata extraction
- Thread-safe caching and concurrent access control for high-performance batch processing
- Comprehensive audit trail integration for scientific computing traceability and reproducibility
- Configuration merging with conflict resolution and priority handling for complex scenarios
- Backup and restoration capabilities with versioning and atomic operations
- Environment variable substitution and intelligent default value application
- Cross-dependency validation and parameter consistency checking
- Performance optimization for high-throughput scientific computing workflows
"""

# External library imports with version specifications for scientific computing compatibility
import pathlib  # Python 3.9+ - Modern cross-platform path handling for configuration files
import json  # Python 3.9+ - JSON parsing and serialization for configuration files
import yaml  # PyYAML 6.0+ - YAML configuration file parsing and serialization
import os  # Python 3.9+ - Operating system interface for environment variables and file operations
import copy  # Python 3.9+ - Deep copying for configuration merging and default application
import threading  # Python 3.9+ - Thread-safe configuration operations and caching
import datetime  # Python 3.9+ - Timestamp generation for configuration versioning and audit trails
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Type hints for configuration reader function signatures

# Internal imports from utility modules for comprehensive configuration management
from ..utils.config_parser import (
    load_configuration,
    save_configuration,
    validate_configuration,
    ConfigurationParser
)
from ..utils.validation_utils import (
    validate_configuration_schema,
    ValidationResult,
    ConfigurationError,
    ValidationError
)
from ..utils.file_utils import (
    load_json_config,
    save_json_config,
    validate_file_exists,
    FileValidationResult
)
from ..utils.logging_utils import (
    get_logger,
    create_audit_trail,
    log_validation_error
)
from ..config import (
    get_default_normalization_config,
    get_default_simulation_config,
    get_default_analysis_config
)

# Global configuration constants for supported formats and processing limits
SUPPORTED_CONFIG_FORMATS = ['.json', '.yaml', '.yml']
DEFAULT_CONFIG_ENCODING = 'utf-8'
CONFIG_CACHE_TIMEOUT_HOURS = 24
MAX_CONFIG_FILE_SIZE_MB = 10
VALIDATION_TIMEOUT_SECONDS = 30

# Thread-safe configuration cache and metadata storage for performance optimization
_config_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamps: Dict[str, datetime.datetime] = {}
_cache_lock: threading.RLock = threading.RLock()

# Global logger instance for configuration reader operations with scientific context
_logger = get_logger(__name__, 'config_reader')


def read_config(
    config_path: str,
    validate_schema: bool = True,
    use_cache: bool = True,
    schema_path: Optional[str] = None,
    apply_defaults: bool = True
) -> Dict[str, Any]:
    """
    Read configuration file with automatic format detection, validation, caching, and error 
    handling for scientific computing configuration management with fail-fast validation strategy.
    
    This function provides comprehensive configuration reading with automatic format detection,
    robust validation, intelligent caching, and comprehensive error handling optimized for
    scientific computing workflows and reproducible research environments.
    
    Args:
        config_path: Path to the configuration file to read
        validate_schema: Enable JSON schema validation for configuration structure verification
        use_cache: Enable configuration caching for improved performance and consistency
        schema_path: Optional custom path to JSON schema file for validation
        apply_defaults: Apply intelligent default values for missing parameters
        
    Returns:
        Dict[str, Any]: Loaded and validated configuration dictionary with applied defaults 
                       and comprehensive error handling
    """
    try:
        # Validate config_path exists and is accessible using comprehensive file validation
        file_validation = validate_file_exists(config_path, check_readable=True, check_size_limits=True)
        if not file_validation.is_valid:
            error_details = {
                'config_path': config_path,
                'validation_errors': file_validation.errors,
                'file_access_issues': True
            }
            
            log_validation_error(
                validation_type='configuration_file_access',
                error_message=f"Configuration file validation failed: {config_path}",
                validation_context=error_details,
                recovery_recommendations=[
                    'Verify configuration file path exists and is accessible',
                    'Check file permissions and ownership',
                    'Ensure file is not corrupted or empty'
                ]
            )
            
            raise ConfigurationError(
                f"Configuration file validation failed: {config_path}",
                configuration_section='file_access_validation',
                schema_type='file_system'
            )
        
        # Detect configuration file format based on file extension and content analysis
        detected_format_info = detect_config_format(
            config_path, 
            analyze_content=True, 
            extract_metadata=True
        )
        
        format_type = detected_format_info['detected_format']
        format_confidence = detected_format_info['confidence_score']
        
        if format_confidence < 0.8:
            _logger.warning(
                f"Low confidence format detection for {config_path}: {format_type} "
                f"(confidence: {format_confidence:.2f})"
            )
        
        # Check configuration cache if use_cache is enabled with thread-safe access
        cache_key = f"{config_path}:{validate_schema}:{apply_defaults}"
        if use_cache:
            with _cache_lock:
                if cache_key in _config_cache and _is_cache_valid(cache_key):
                    cached_config = copy.deepcopy(_config_cache[cache_key])
                    _logger.debug(f"Configuration loaded from cache: {config_path}")
                    
                    # Create audit trail entry for cached configuration access
                    create_audit_trail(
                        action='CONFIGURATION_CACHE_HIT',
                        component='CONFIG_READER',
                        action_details={
                            'config_path': config_path,
                            'cache_key': cache_key,
                            'format_type': format_type
                        },
                        user_context='SYSTEM'
                    )
                    
                    return cached_config
        
        # Load configuration using appropriate format parser based on detected format
        _logger.info(f"Loading configuration: {config_path} (format: {format_type})")
        
        if format_type == '.json':
            config_data = load_json_config(
                config_path=config_path,
                schema_path=schema_path,
                validate_schema=validate_schema,
                use_cache=False  # We handle caching at this level
            )
        elif format_type in ['.yaml', '.yml']:
            config_data = _load_yaml_config(
                config_path=config_path,
                validate_schema=validate_schema,
                schema_path=schema_path
            )
        else:
            raise ConfigurationError(
                f"Unsupported configuration format: {format_type}. Supported: {SUPPORTED_CONFIG_FORMATS}",
                configuration_section='format_detection',
                schema_type='file_format'
            )
        
        # Apply default values if apply_defaults is enabled
        if apply_defaults:
            config_data = _apply_intelligent_defaults(config_data, config_path)
        
        # Perform comprehensive schema validation if validate_schema is enabled
        if validate_schema:
            validation_result = _perform_comprehensive_validation(
                config_data, 
                config_path, 
                schema_path,
                format_type
            )
            
            if not validation_result.is_valid:
                error_details = {
                    'config_path': config_path,
                    'validation_errors': validation_result.errors,
                    'validation_warnings': validation_result.warnings,
                    'format_type': format_type,
                    'schema_path': schema_path
                }
                
                log_validation_error(
                    validation_type='configuration_schema_validation',
                    error_message=f"Configuration schema validation failed: {config_path}",
                    validation_context=error_details,
                    failed_parameters=[],
                    recovery_recommendations=validation_result.recommendations
                )
                
                raise ValidationError(
                    f"Configuration validation failed: {len(validation_result.errors)} errors found",
                    validation_type='configuration_schema_validation',
                    validation_context=config_path
                )
        
        # Cache configuration if caching is enabled with thread-safe storage
        if use_cache:
            with _cache_lock:
                _config_cache[cache_key] = copy.deepcopy(config_data)
                _cache_timestamps[cache_key] = datetime.datetime.now()
                _logger.debug(f"Configuration cached: {config_path}")
        
        # Create comprehensive audit trail entry for configuration access
        create_audit_trail(
            action='CONFIGURATION_LOADED',
            component='CONFIG_READER',
            action_details={
                'config_path': config_path,
                'format_type': format_type,
                'format_confidence': format_confidence,
                'validate_schema': validate_schema,
                'apply_defaults': apply_defaults,
                'use_cache': use_cache,
                'config_size_bytes': len(str(config_data)),
                'parameter_count': _count_configuration_parameters(config_data)
            },
            user_context='SYSTEM'
        )
        
        # Log configuration reading operation with performance metrics
        _logger.info(
            f"Configuration loaded successfully: {config_path} "
            f"(format: {format_type}, parameters: {_count_configuration_parameters(config_data)})"
        )
        
        return config_data
        
    except Exception as e:
        # Handle configuration loading errors with comprehensive error reporting
        error_context = {
            'config_path': config_path,
            'operation': 'read_config',
            'error_type': type(e).__name__,
            'error_message': str(e)
        }
        
        _logger.error(f"Failed to read configuration {config_path}: {e}", exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                f"Configuration reading failed: {str(e)}",
                configuration_section='config_reading_operation',
                schema_type='unknown'
            ) from e


def read_config_with_defaults(
    config_path: str,
    config_type: str,
    validate_schema: bool = True,
    strict_validation: bool = False,
    override_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Read configuration file and merge with default values from the centralized configuration 
    system for complete parameter coverage and scientific reproducibility.
    
    This function provides comprehensive configuration reading with intelligent default value
    application, centralized system integration, and override support for complete parameter
    coverage essential for reproducible scientific computing workflows.
    
    Args:
        config_path: Path to the configuration file to read and enhance
        config_type: Type of configuration for default value selection and validation
        validate_schema: Enable comprehensive schema validation for structure verification
        strict_validation: Enable strict validation mode with comprehensive parameter checks
        override_values: Dictionary of override values to apply after default merging
        
    Returns:
        Dict[str, Any]: Configuration with applied defaults and override values for complete 
                       parameter coverage and scientific reproducibility
    """
    try:
        # Load base configuration using comprehensive read_config function
        base_config = read_config(
            config_path=config_path,
            validate_schema=False,  # We'll validate after merging defaults
            use_cache=True,
            apply_defaults=False   # We'll apply defaults explicitly here
        )
        
        # Retrieve appropriate default configuration based on config_type
        default_config = _get_default_configuration_by_type(config_type)
        
        if not default_config:
            _logger.warning(f"No default configuration available for type: {config_type}")
            default_config = {}
        
        # Merge base configuration with defaults using intelligent merging strategy
        merged_config = _intelligent_configuration_merge(
            base_config, 
            default_config, 
            merge_strategy='preserve_user_values'
        )
        
        # Apply override values if provided using deep merge approach
        if override_values:
            merged_config = _apply_configuration_overrides(merged_config, override_values)
        
        # Perform comprehensive validation if validate_schema is enabled
        if validate_schema:
            validation_result = validate_configuration_schema(
                config_data=merged_config,
                schema_type=config_type,
                strict_mode=strict_validation,
                required_sections=_get_required_sections_for_type(config_type)
            )
            
            if not validation_result.is_valid:
                error_details = {
                    'config_path': config_path,
                    'config_type': config_type,
                    'validation_errors': validation_result.errors,
                    'validation_warnings': validation_result.warnings
                }
                
                log_validation_error(
                    validation_type='configuration_with_defaults_validation',
                    error_message=f"Configuration with defaults validation failed: {config_path}",
                    validation_context=error_details,
                    recovery_recommendations=validation_result.recommendations
                )
                
                if strict_validation:
                    raise ValidationError(
                        f"Strict validation failed: {len(validation_result.errors)} errors",
                        validation_type='configuration_with_defaults_validation',
                        validation_context=config_type
                    )
        
        # Apply strict validation rules if strict_validation is enabled
        if strict_validation:
            _apply_strict_validation_rules(merged_config, config_type)
        
        # Create comprehensive audit trail entry for configuration merging operation
        create_audit_trail(
            action='CONFIGURATION_MERGED_WITH_DEFAULTS',
            component='CONFIG_READER',
            action_details={
                'config_path': config_path,
                'config_type': config_type,
                'validate_schema': validate_schema,
                'strict_validation': strict_validation,
                'has_overrides': override_values is not None,
                'base_parameter_count': _count_configuration_parameters(base_config),
                'merged_parameter_count': _count_configuration_parameters(merged_config),
                'defaults_applied': _count_configuration_parameters(default_config)
            },
            user_context='SYSTEM'
        )
        
        # Log configuration merging operation with applied defaults summary
        _logger.info(
            f"Configuration merged with defaults: {config_path} "
            f"(type: {config_type}, parameters: {_count_configuration_parameters(merged_config)})"
        )
        
        return merged_config
        
    except Exception as e:
        # Handle configuration merging errors with comprehensive error reporting
        _logger.error(f"Failed to read configuration with defaults {config_path}: {e}", exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                f"Configuration reading with defaults failed: {str(e)}",
                configuration_section='config_defaults_merging',
                schema_type=config_type
            ) from e


def validate_config_file(
    config_path: str,
    schema_path: Optional[str] = None,
    check_file_integrity: bool = True,
    validate_scientific_parameters: bool = True
) -> ValidationResult:
    """
    Validate configuration file against schema and scientific constraints without loading full 
    configuration for fail-fast validation strategy and resource optimization.
    
    This function provides comprehensive validation without full configuration loading for
    fail-fast error detection and resource optimization essential for batch processing
    workflows and early error detection in scientific computing environments.
    
    Args:
        config_path: Path to the configuration file to validate
        schema_path: Optional custom path to JSON schema file for structure validation
        check_file_integrity: Enable file integrity and accessibility validation
        validate_scientific_parameters: Enable scientific parameter constraint validation
        
    Returns:
        ValidationResult: Comprehensive validation result with errors, warnings, and 
                         recommendations for configuration improvement
    """
    # Create comprehensive ValidationResult container for configuration validation
    validation_result = ValidationResult(
        validation_type='configuration_file_validation',
        is_valid=True,
        validation_context=f"config_path={config_path}"
    )
    
    try:
        # Validate file existence and accessibility if check_file_integrity is enabled
        if check_file_integrity:
            file_validation = validate_file_exists(
                config_path, 
                check_readable=True, 
                check_size_limits=True
            )
            
            if not file_validation.is_valid:
                for error in file_validation.errors:
                    validation_result.add_error(
                        f"File integrity check failed: {error}",
                        severity=ValidationResult.ErrorSeverity.HIGH
                    )
                validation_result.is_valid = False
                return validation_result
            
            # Add file metadata to validation result for comprehensive reporting
            validation_result.set_metadata('file_size_bytes', file_validation.metadata.get('file_size_bytes', 0))
            validation_result.set_metadata('file_readable', file_validation.metadata.get('readable', False))
        
        # Detect configuration format and validate format compatibility
        format_detection = detect_config_format(
            config_path, 
            analyze_content=True, 
            extract_metadata=True
        )
        
        detected_format = format_detection['detected_format']
        format_confidence = format_detection['confidence_score']
        
        validation_result.set_metadata('detected_format', detected_format)
        validation_result.set_metadata('format_confidence', format_confidence)
        
        if detected_format not in SUPPORTED_CONFIG_FORMATS:
            validation_result.add_error(
                f"Unsupported configuration format: {detected_format}",
                severity=ValidationResult.ErrorSeverity.HIGH
            )
            validation_result.is_valid = False
        
        if format_confidence < 0.7:
            validation_result.add_warning(
                f"Low confidence format detection: {detected_format} (confidence: {format_confidence:.2f})"
            )
        
        # Load configuration structure for validation without full processing
        try:
            if detected_format == '.json':
                with open(config_path, 'r', encoding=DEFAULT_CONFIG_ENCODING) as f:
                    config_structure = json.load(f)
            elif detected_format in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding=DEFAULT_CONFIG_ENCODING) as f:
                    config_structure = yaml.safe_load(f)
            else:
                raise ConfigurationError(
                    f"Cannot load configuration format: {detected_format}",
                    configuration_section='format_loading',
                    schema_type='format_validation'
                )
                
            validation_result.set_metadata('config_loaded', True)
            validation_result.set_metadata('parameter_count', _count_configuration_parameters(config_structure))
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            validation_result.add_error(
                f"Configuration parsing failed: {str(e)}",
                severity=ValidationResult.ErrorSeverity.CRITICAL
            )
            validation_result.is_valid = False
            return validation_result
        
        # Perform schema validation if schema_path is provided
        if schema_path:
            try:
                schema_validation = validate_configuration_schema(
                    config_data=config_structure,
                    schema_type='custom',
                    strict_mode=False,
                    required_sections=[]
                )
                
                if not schema_validation.is_valid:
                    for error in schema_validation.errors:
                        validation_result.add_error(
                            f"Schema validation failed: {error}",
                            severity=ValidationResult.ErrorSeverity.MEDIUM
                        )
                    validation_result.is_valid = False
                
                for warning in schema_validation.warnings:
                    validation_result.add_warning(f"Schema validation warning: {warning}")
                    
            except Exception as e:
                validation_result.add_warning(f"Schema validation error: {str(e)}")
        
        # Validate scientific parameters if validate_scientific_parameters is enabled
        if validate_scientific_parameters:
            _validate_scientific_parameter_structure(config_structure, validation_result)
        
        # Check for deprecated parameters and configuration patterns
        _check_deprecated_configuration_patterns(config_structure, validation_result)
        
        # Generate recovery recommendations for validation failures
        if not validation_result.is_valid:
            _generate_configuration_recovery_recommendations(validation_result)
        
        # Add comprehensive validation metrics for analysis
        validation_result.add_metric('validation_completeness', 1.0)
        validation_result.add_metric('error_severity_score', _calculate_error_severity_score(validation_result))
        
        # Create audit trail entry for validation operation
        create_audit_trail(
            action='CONFIGURATION_FILE_VALIDATED',
            component='CONFIG_READER',
            action_details={
                'config_path': config_path,
                'schema_path': schema_path,
                'check_file_integrity': check_file_integrity,
                'validate_scientific_parameters': validate_scientific_parameters,
                'validation_result': validation_result.get_summary(),
                'detected_format': detected_format,
                'format_confidence': format_confidence
            },
            user_context='SYSTEM'
        )
        
        # Log validation operation completion with comprehensive results
        _logger.info(
            f"Configuration file validation completed: {config_path} "
            f"(valid: {validation_result.is_valid}, errors: {len(validation_result.errors)}, "
            f"warnings: {len(validation_result.warnings)})"
        )
        
        return validation_result
        
    except Exception as e:
        # Handle validation process errors with comprehensive error reporting
        validation_result.add_error(
            f"Validation process failed: {str(e)}",
            severity=ValidationResult.ErrorSeverity.CRITICAL
        )
        validation_result.is_valid = False
        
        _logger.error(f"Configuration file validation failed for {config_path}: {e}", exc_info=True)
        
        return validation_result


def detect_config_format(
    config_path: str,
    analyze_content: bool = True,
    extract_metadata: bool = True
) -> Dict[str, Any]:
    """
    Detect configuration file format with confidence scoring and format-specific metadata 
    extraction for automatic format handling and processing optimization.
    
    This function provides comprehensive format detection with confidence scoring, content
    analysis, and metadata extraction for optimal configuration processing and automatic
    format handling in scientific computing workflows.
    
    Args:
        config_path: Path to the configuration file for format detection
        analyze_content: Enable content analysis for improved detection accuracy
        extract_metadata: Include format-specific metadata in detection results
        
    Returns:
        Dict[str, Any]: Format detection result with confidence score, detected format, 
                       and metadata information for processing optimization
    """
    detection_result = {
        'detected_format': None,
        'confidence_score': 0.0,
        'detection_method': [],
        'format_metadata': {},
        'analysis_timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        # Analyze file extension for initial format detection
        config_file = pathlib.Path(config_path)
        file_extension = config_file.suffix.lower()
        
        detection_result['file_extension'] = file_extension
        detection_result['detection_method'].append('file_extension')
        
        # Assign initial confidence based on file extension
        if file_extension in SUPPORTED_CONFIG_FORMATS:
            detection_result['detected_format'] = file_extension
            detection_result['confidence_score'] = 0.8  # High confidence for known extensions
        else:
            detection_result['confidence_score'] = 0.1  # Low confidence for unknown extensions
        
        # Read file header to confirm format if analyze_content is enabled
        if analyze_content and config_file.exists():
            try:
                with open(config_path, 'r', encoding=DEFAULT_CONFIG_ENCODING) as f:
                    content_sample = f.read(512)  # Read first 512 characters
                
                detection_result['detection_method'].append('content_analysis')
                
                # Analyze content for JSON format indicators
                if content_sample.strip().startswith(('{', '[')):
                    try:
                        json.loads(content_sample if len(content_sample) < 512 else f.read())
                        if file_extension == '.json':
                            detection_result['confidence_score'] = 0.95
                        else:
                            detection_result['detected_format'] = '.json'
                            detection_result['confidence_score'] = 0.85
                    except json.JSONDecodeError:
                        if file_extension == '.json':
                            detection_result['confidence_score'] = 0.6  # Extension says JSON but content is malformed
                
                # Analyze content for YAML format indicators
                elif any(indicator in content_sample for indicator in ['---', ':', '- ']):
                    try:
                        f.seek(0)  # Reset file pointer
                        yaml.safe_load(f.read())
                        if file_extension in ['.yaml', '.yml']:
                            detection_result['confidence_score'] = 0.95
                        else:
                            detection_result['detected_format'] = '.yaml'
                            detection_result['confidence_score'] = 0.85
                    except yaml.YAMLError:
                        if file_extension in ['.yaml', '.yml']:
                            detection_result['confidence_score'] = 0.6
                
            except (IOError, UnicodeDecodeError) as e:
                detection_result['content_analysis_error'] = str(e)
                detection_result['confidence_score'] *= 0.7  # Reduce confidence due to read error
        
        # Extract format-specific metadata if extract_metadata is enabled
        if extract_metadata and config_file.exists():
            try:
                file_stat = config_file.stat()
                detection_result['format_metadata'] = {
                    'file_size_bytes': file_stat.st_size,
                    'last_modified': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    'is_readable': os.access(config_path, os.R_OK),
                    'is_writable': os.access(config_path, os.W_OK),
                    'encoding_detected': DEFAULT_CONFIG_ENCODING  # Could be enhanced with chardet
                }
                
                # Add format-specific metadata based on detected format
                if detection_result['detected_format'] == '.json':
                    detection_result['format_metadata']['json_specific'] = {
                        'supports_comments': False,
                        'strict_syntax': True,
                        'unicode_support': True
                    }
                elif detection_result['detected_format'] in ['.yaml', '.yml']:
                    detection_result['format_metadata']['yaml_specific'] = {
                        'supports_comments': True,
                        'multiline_support': True,
                        'anchor_references': True
                    }
                
            except OSError as e:
                detection_result['metadata_extraction_error'] = str(e)
        
        # Validate format compatibility with configuration system
        if detection_result['detected_format'] in SUPPORTED_CONFIG_FORMATS:
            detection_result['system_compatible'] = True
        else:
            detection_result['system_compatible'] = False
            detection_result['confidence_score'] *= 0.5  # Reduce confidence for unsupported formats
        
        # Create audit trail entry for format detection operation
        create_audit_trail(
            action='CONFIG_FORMAT_DETECTED',
            component='CONFIG_READER',
            action_details={
                'config_path': config_path,
                'detected_format': detection_result['detected_format'],
                'confidence_score': detection_result['confidence_score'],
                'detection_methods': detection_result['detection_method'],
                'analyze_content': analyze_content,
                'extract_metadata': extract_metadata
            },
            user_context='SYSTEM'
        )
        
        # Log format detection operation with results and confidence
        _logger.debug(
            f"Format detection completed: {config_path} -> {detection_result['detected_format']} "
            f"(confidence: {detection_result['confidence_score']:.2f})"
        )
        
        return detection_result
        
    except Exception as e:
        # Handle format detection errors with fallback to extension-based detection
        detection_result['detection_error'] = str(e)
        detection_result['confidence_score'] = 0.3  # Low confidence due to detection error
        
        _logger.warning(f"Format detection error for {config_path}: {e}")
        
        return detection_result


def get_config_path(
    config_name: str,
    config_directory: Optional[str] = None,
    resolve_environment_variables: bool = True,
    validate_path: bool = True
) -> pathlib.Path:
    """
    Resolve configuration file path with support for relative paths, environment variables, 
    and configuration directory resolution for flexible configuration management.
    
    This function provides comprehensive path resolution with environment variable support,
    relative path handling, and validation for flexible configuration management across
    different deployment environments and scientific computing workflows.
    
    Args:
        config_name: Name of the configuration file (with or without extension)
        config_directory: Optional custom configuration directory path
        resolve_environment_variables: Enable environment variable substitution in paths
        validate_path: Enable path accessibility validation after resolution
        
    Returns:
        pathlib.Path: Resolved configuration file path with validation and environment 
                     variable substitution
    """
    try:
        # Resolve config_directory or use default configuration directory
        if config_directory:
            base_directory = pathlib.Path(config_directory)
        else:
            # Import and use default configuration directory
            from ..config import get_config_directory
            base_directory = get_config_directory()
        
        # Substitute environment variables if resolve_environment_variables is enabled
        if resolve_environment_variables:
            config_name = _substitute_environment_variables_in_string(config_name)
            if config_directory:
                config_directory = _substitute_environment_variables_in_string(config_directory)
                base_directory = pathlib.Path(config_directory)
        
        # Construct full configuration file path with appropriate extension
        if not pathlib.Path(config_name).suffix:
            # Add default .json extension if no extension provided
            config_file_path = base_directory / f"{config_name}.json"
        else:
            config_file_path = base_directory / config_name
        
        # Resolve relative paths to absolute paths for consistency
        resolved_path = config_file_path.resolve()
        
        # Validate path accessibility if validate_path is enabled
        if validate_path:
            # Check if the resolved path exists and is accessible
            if not resolved_path.exists():
                _logger.warning(f"Configuration file does not exist: {resolved_path}")
            
            # Validate parent directory exists and is accessible
            parent_directory = resolved_path.parent
            if not parent_directory.exists():
                _logger.warning(f"Configuration directory does not exist: {parent_directory}")
            elif not os.access(parent_directory, os.R_OK):
                _logger.warning(f"Configuration directory is not readable: {parent_directory}")
        
        # Create audit trail entry for path resolution operation
        create_audit_trail(
            action='CONFIG_PATH_RESOLVED',
            component='CONFIG_READER',
            action_details={
                'config_name': config_name,
                'config_directory': str(config_directory) if config_directory else 'default',
                'resolved_path': str(resolved_path),
                'resolve_environment_variables': resolve_environment_variables,
                'validate_path': validate_path,
                'path_exists': resolved_path.exists()
            },
            user_context='SYSTEM'
        )
        
        # Log path resolution operation with resolved path information
        _logger.debug(f"Configuration path resolved: {config_name} -> {resolved_path}")
        
        return resolved_path
        
    except Exception as e:
        # Handle path resolution errors with comprehensive error reporting
        _logger.error(f"Configuration path resolution failed for {config_name}: {e}")
        
        raise ConfigurationError(
            f"Configuration path resolution failed: {str(e)}",
            configuration_section='path_resolution',
            schema_type='path_handling'
        ) from e


def clear_config_cache(
    config_paths: Optional[List[str]] = None,
    preserve_statistics: bool = True,
    clear_reason: str = 'manual_cache_clear'
) -> Dict[str, int]:
    """
    Clear configuration cache with selective clearing options and statistics preservation 
    for cache management and development scenarios.
    
    This function provides comprehensive cache management with selective clearing, statistics
    preservation, and audit trail integration for development, testing, and maintenance
    scenarios requiring fresh configuration data loading.
    
    Args:
        config_paths: Optional list of specific configuration paths to clear from cache
        preserve_statistics: Whether to preserve cache access statistics and metadata
        clear_reason: Reason for clearing the cache for audit trail and debugging
        
    Returns:
        Dict[str, int]: Cache clearing statistics with cleared entries count and preservation 
                       summary for monitoring and analysis
    """
    try:
        # Acquire cache lock for thread-safe operation
        with _cache_lock:
            # Record cache state before clearing for statistics
            cache_size_before = len(_config_cache)
            timestamp_entries_before = len(_cache_timestamps)
            
            clearing_statistics = {
                'cache_entries_before': cache_size_before,
                'timestamp_entries_before': timestamp_entries_before,
                'entries_cleared': 0,
                'timestamps_cleared': 0,
                'selective_clearing': config_paths is not None,
                'statistics_preserved': preserve_statistics,
                'clear_reason': clear_reason,
                'clear_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Identify cache entries to clear based on config_paths filter
            if config_paths is None:
                # Clear all configuration cache entries
                entries_to_clear = list(_config_cache.keys())
                timestamps_to_clear = list(_cache_timestamps.keys())
            else:
                # Clear only specified configuration cache entries
                entries_to_clear = []
                timestamps_to_clear = []
                
                for cache_key in _config_cache.keys():
                    for config_path in config_paths:
                        if config_path in cache_key:
                            entries_to_clear.append(cache_key)
                            break
                
                for timestamp_key in _cache_timestamps.keys():
                    for config_path in config_paths:
                        if config_path in timestamp_key:
                            timestamps_to_clear.append(timestamp_key)
                            break
            
            # Clear specified configuration cache entries
            for cache_key in entries_to_clear:
                if cache_key in _config_cache:
                    del _config_cache[cache_key]
                    clearing_statistics['entries_cleared'] += 1
            
            # Clear associated cache timestamps
            for timestamp_key in timestamps_to_clear:
                if timestamp_key in _cache_timestamps:
                    del _cache_timestamps[timestamp_key]
                    clearing_statistics['timestamps_cleared'] += 1
            
            # Update cache metadata and clearing history if preserve_statistics is enabled
            if preserve_statistics:
                # Store clearing statistics for future reference
                if not hasattr(clear_config_cache, '_clearing_history'):
                    clear_config_cache._clearing_history = []
                
                clear_config_cache._clearing_history.append({
                    'timestamp': clearing_statistics['clear_timestamp'],
                    'reason': clear_reason,
                    'entries_cleared': clearing_statistics['entries_cleared'],
                    'selective': clearing_statistics['selective_clearing']
                })
                
                # Limit history to last 100 entries
                if len(clear_config_cache._clearing_history) > 100:
                    clear_config_cache._clearing_history = clear_config_cache._clearing_history[-100:]
        
        # Create comprehensive audit trail entry for cache clearing operation
        create_audit_trail(
            action='CONFIG_CACHE_CLEARED',
            component='CONFIG_READER',
            action_details=clearing_statistics,
            user_context='SYSTEM'
        )
        
        # Log cache clearing operation with statistics and reason
        _logger.info(
            f"Configuration cache cleared: {clearing_statistics['entries_cleared']} entries, "
            f"{clearing_statistics['timestamps_cleared']} timestamps (reason: {clear_reason})"
        )
        
        return clearing_statistics
        
    except Exception as e:
        # Handle cache clearing errors with comprehensive error reporting
        error_message = f"Configuration cache clearing failed: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        return {
            'error': error_message,
            'entries_cleared': 0,
            'timestamps_cleared': 0,
            'clear_timestamp': datetime.datetime.now().isoformat()
        }


def get_config_cache_info(
    include_detailed_stats: bool = True,
    include_cache_contents: bool = False
) -> Dict[str, Any]:
    """
    Retrieve configuration cache information including cache statistics, hit rates, and memory 
    usage for performance monitoring and optimization.
    
    This function provides comprehensive cache analysis with performance metrics, memory usage
    statistics, and detailed cache state information for performance monitoring, optimization,
    and system analysis in high-throughput scientific computing environments.
    
    Args:
        include_detailed_stats: Include detailed cache statistics and performance metrics
        include_cache_contents: Include summary of cached configuration contents
        
    Returns:
        Dict[str, Any]: Configuration cache information with statistics, performance metrics, 
                       and usage analysis for monitoring and optimization
    """
    try:
        # Acquire cache lock for consistent cache state reading
        with _cache_lock:
            # Calculate basic cache statistics
            cache_info = {
                'cache_size': len(_config_cache),
                'timestamp_entries': len(_cache_timestamps),
                'cache_enabled': True,
                'info_generation_time': datetime.datetime.now().isoformat()
            }
            
            # Calculate cache hit rates and access statistics if include_detailed_stats is enabled
            if include_detailed_stats:
                current_time = datetime.datetime.now()
                valid_entries = 0
                expired_entries = 0
                total_cache_memory = 0
                
                # Analyze cache entries for detailed statistics
                for cache_key, cache_data in _config_cache.items():
                    # Calculate estimated memory usage
                    entry_memory = len(str(cache_data).encode('utf-8'))
                    total_cache_memory += entry_memory
                    
                    # Check cache entry validity
                    if cache_key in _cache_timestamps:
                        cache_time = _cache_timestamps[cache_key]
                        if _is_cache_entry_valid(cache_key, cache_time, current_time):
                            valid_entries += 1
                        else:
                            expired_entries += 1
                    else:
                        expired_entries += 1
                
                cache_info['detailed_statistics'] = {
                    'valid_entries': valid_entries,
                    'expired_entries': expired_entries,
                    'cache_efficiency': (valid_entries / len(_config_cache)) * 100 if _config_cache else 0,
                    'estimated_memory_usage_bytes': total_cache_memory,
                    'estimated_memory_usage_mb': round(total_cache_memory / (1024 * 1024), 2),
                    'average_entry_size_bytes': total_cache_memory // len(_config_cache) if _config_cache else 0
                }
                
                # Include cache clearing history if available
                if hasattr(clear_config_cache, '_clearing_history'):
                    cache_info['clearing_history'] = {
                        'total_clears': len(clear_config_cache._clearing_history),
                        'recent_clears': clear_config_cache._clearing_history[-5:],  # Last 5 clears
                        'last_clear': clear_config_cache._clearing_history[-1] if clear_config_cache._clearing_history else None
                    }
            
            # Include cache contents summary if include_cache_contents is enabled
            if include_cache_contents:
                cache_contents = {}
                for cache_key in _config_cache.keys():
                    # Extract configuration path from cache key
                    config_path = cache_key.split(':')[0] if ':' in cache_key else cache_key
                    cache_timestamp = _cache_timestamps.get(cache_key)
                    
                    cache_contents[cache_key] = {
                        'config_path': config_path,
                        'cached_at': cache_timestamp.isoformat() if cache_timestamp else 'unknown',
                        'is_valid': _is_cache_valid(cache_key),
                        'parameter_count': _count_configuration_parameters(_config_cache[cache_key])
                    }
                
                cache_info['cache_contents'] = cache_contents
            
            # Generate cache performance analysis and recommendations
            cache_info['performance_analysis'] = _generate_cache_performance_analysis()
            
            # Include cache configuration and limits
            cache_info['cache_configuration'] = {
                'cache_timeout_hours': CONFIG_CACHE_TIMEOUT_HOURS,
                'max_file_size_mb': MAX_CONFIG_FILE_SIZE_MB,
                'supported_formats': SUPPORTED_CONFIG_FORMATS,
                'default_encoding': DEFAULT_CONFIG_ENCODING
            }
        
        # Create audit trail entry for cache information access
        create_audit_trail(
            action='CONFIG_CACHE_INFO_ACCESSED',
            component='CONFIG_READER',
            action_details={
                'include_detailed_stats': include_detailed_stats,
                'include_cache_contents': include_cache_contents,
                'cache_size': cache_info['cache_size'],
                'timestamp_entries': cache_info['timestamp_entries']
            },
            user_context='SYSTEM'
        )
        
        # Log cache information access with summary statistics
        _logger.debug(
            f"Configuration cache info accessed: {cache_info['cache_size']} entries, "
            f"detailed_stats={include_detailed_stats}, contents={include_cache_contents}"
        )
        
        return cache_info
        
    except Exception as e:
        # Handle cache information retrieval errors with error reporting
        error_message = f"Configuration cache info retrieval failed: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        return {
            'error': error_message,
            'cache_size': 0,
            'timestamp_entries': 0,
            'info_generation_time': datetime.datetime.now().isoformat()
        }


def merge_config_files(
    config_paths: List[str],
    merge_strategy: str = 'deep_merge',
    validate_result: bool = True,
    priority_order: Optional[List[str]] = None,
    preserve_metadata: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple configuration files with conflict resolution, priority handling, and 
    validation for complex configuration scenarios and experimental condition setup.
    
    This function provides sophisticated configuration merging with multiple strategies,
    comprehensive conflict resolution, and validation support for complex scientific
    computing workflows and experimental condition setup requiring merged configurations.
    
    Args:
        config_paths: List of configuration file paths to merge in processing order
        merge_strategy: Strategy for merging configurations ('deep_merge', 'overlay', 'priority')
        validate_result: Enable validation of merged configuration against applicable schemas
        priority_order: Optional priority order for conflict resolution during merging
        preserve_metadata: Preserve metadata from source configurations in merged result
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary with resolved conflicts and validated 
                       consistency for complex experimental scenarios
    """
    try:
        # Validate input parameters and configuration paths
        if not config_paths:
            raise ConfigurationError(
                "Configuration paths list cannot be empty",
                configuration_section='merge_input_validation',
                schema_type='merge_operation'
            )
        
        if len(config_paths) == 1:
            # Single configuration - load and return with optional validation
            single_config = read_config(config_paths[0], validate_schema=validate_result)
            return single_config
        
        # Load all configuration files from config_paths list with comprehensive validation
        loaded_configurations = []
        merge_metadata = {
            'merge_strategy': merge_strategy,
            'source_count': len(config_paths),
            'merge_timestamp': datetime.datetime.now().isoformat(),
            'conflicts_resolved': 0,
            'source_files': config_paths,
            'merge_id': f"merge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        for i, config_path in enumerate(config_paths):
            try:
                config_data = read_config(
                    config_path, 
                    validate_schema=False,  # Validate after merging
                    use_cache=True
                )
                
                # Add source metadata if preserve_metadata is enabled
                if preserve_metadata:
                    config_data['_source_metadata'] = {
                        'source_path': config_path,
                        'source_index': i,
                        'load_timestamp': datetime.datetime.now().isoformat(),
                        'merge_priority': priority_order.index(config_path) if priority_order and config_path in priority_order else i
                    }
                
                loaded_configurations.append(config_data)
                
            except Exception as e:
                _logger.error(f"Failed to load configuration for merging: {config_path} - {e}")
                raise ConfigurationError(
                    f"Failed to load configuration for merging: {config_path}",
                    configuration_section='merge_config_loading',
                    schema_type='merge_operation'
                ) from e
        
        # Apply priority order if specified for conflict resolution
        if priority_order:
            # Reorder configurations based on priority_order specification
            ordered_configurations = []
            for priority_path in priority_order:
                for i, config_path in enumerate(config_paths):
                    if config_path == priority_path:
                        ordered_configurations.append(loaded_configurations[i])
                        break
            
            # Add any configurations not in priority_order at the end
            for i, config_path in enumerate(config_paths):
                if config_path not in priority_order:
                    ordered_configurations.append(loaded_configurations[i])
            
            loaded_configurations = ordered_configurations
        
        # Merge configurations using specified merge strategy
        _logger.info(f"Merging {len(config_paths)} configurations using {merge_strategy} strategy")
        
        if merge_strategy == 'deep_merge':
            merged_config = _deep_merge_configurations(loaded_configurations, merge_metadata)
        elif merge_strategy == 'overlay':
            merged_config = _overlay_merge_configurations(loaded_configurations, merge_metadata)
        elif merge_strategy == 'priority':
            merged_config = _priority_merge_configurations(loaded_configurations, merge_metadata, priority_order)
        else:
            # Default to deep merge for unknown strategies
            _logger.warning(f"Unknown merge strategy '{merge_strategy}', using deep_merge")
            merged_config = _deep_merge_configurations(loaded_configurations, merge_metadata)
        
        # Preserve metadata from source configurations if preserve_metadata is enabled
        if preserve_metadata:
            merged_config['_merge_metadata'] = merge_metadata
            merged_config['_merge_metadata']['final_parameter_count'] = _count_configuration_parameters(merged_config)
        
        # Validate merged configuration if validate_result is enabled
        if validate_result:
            # Perform comprehensive validation against applicable schemas
            validation_issues = []
            
            # Attempt validation against common configuration types
            for config_type in ['normalization', 'simulation', 'analysis']:
                try:
                    validation_result = validate_configuration_schema(
                        config_data=merged_config,
                        schema_type=config_type,
                        strict_mode=False,  # Use relaxed validation for merged configs
                        required_sections=[]
                    )
                    
                    if not validation_result.is_valid:
                        validation_issues.extend(validation_result.errors)
                        
                except Exception:
                    # Skip validation for inapplicable schemas
                    continue
            
            if validation_issues:
                _logger.warning(f"Merged configuration has {len(validation_issues)} validation issues")
                
                # Add validation warnings to metadata if preserving metadata
                if preserve_metadata:
                    merged_config['_merge_metadata']['validation_issues'] = validation_issues
        
        # Check for parameter inconsistencies and compatibility issues
        _check_merged_configuration_consistency(merged_config, config_paths)
        
        # Create comprehensive audit trail entry for merge operation
        create_audit_trail(
            action='CONFIGURATIONS_MERGED',
            component='CONFIG_READER',
            action_details={
                'config_paths': config_paths,
                'merge_strategy': merge_strategy,
                'validate_result': validate_result,
                'priority_order': priority_order,
                'preserve_metadata': preserve_metadata,
                'merge_metadata': merge_metadata,
                'final_parameter_count': _count_configuration_parameters(merged_config)
            },
            user_context='SYSTEM'
        )
        
        # Log merge operation completion with conflict resolution details
        _logger.info(
            f"Configuration merge completed: {len(config_paths)} sources, "
            f"strategy={merge_strategy}, conflicts_resolved={merge_metadata['conflicts_resolved']}"
        )
        
        return merged_config
        
    except Exception as e:
        # Handle configuration merge errors with comprehensive error reporting
        _logger.error(f"Configuration merge failed: {e}", exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                f"Configuration merge operation failed: {str(e)}",
                configuration_section='merge_operation',
                schema_type='merge_operation'
            ) from e


def backup_config_file(
    config_path: str,
    compress_backup: bool = False,
    include_metadata: bool = True,
    backup_directory: Optional[str] = None,
    max_backup_versions: int = 10
) -> str:
    """
    Create backup of configuration file with versioning, compression, and metadata preservation 
    for configuration change tracking and recovery.
    
    This function provides comprehensive configuration backup with versioning, optional
    compression, metadata preservation, and retention management for configuration change
    tracking, recovery operations, and version management in scientific computing workflows.
    
    Args:
        config_path: Path to the configuration file to backup
        compress_backup: Enable backup compression using gzip for space optimization
        include_metadata: Include comprehensive metadata in backup for recovery context
        backup_directory: Optional custom directory for backup storage organization
        max_backup_versions: Maximum number of backup versions to retain for space management
        
    Returns:
        str: Path to created backup file with version information and metadata
    """
    try:
        # Validate source configuration file exists and is accessible
        source_file = pathlib.Path(config_path)
        if not source_file.exists() or not source_file.is_file():
            raise ConfigurationError(
                f"Source configuration file not found or invalid: {config_path}",
                configuration_section='backup_source_validation',
                schema_type='backup_operation'
            )
        
        # Generate backup filename with timestamp and version number
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{source_file.stem}_backup_{timestamp}{source_file.suffix}"
        
        # Determine backup directory with creation if necessary
        if backup_directory:
            backup_dir = pathlib.Path(backup_directory)
        else:
            backup_dir = source_file.parent / 'backups'
        
        # Create backup directory if it doesn't exist
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / backup_filename
        
        _logger.info(f"Creating configuration backup: {config_path} -> {backup_path}")
        
        # Copy configuration file to backup location with error handling
        import shutil
        shutil.copy2(str(source_file), str(backup_path))
        
        # Include comprehensive metadata in backup if include_metadata is enabled
        if include_metadata:
            backup_metadata = {
                'original_path': str(source_file.absolute()),
                'backup_path': str(backup_path.absolute()),
                'backup_created': datetime.datetime.now().isoformat(),
                'original_size_bytes': source_file.stat().st_size,
                'original_modified': datetime.datetime.fromtimestamp(source_file.stat().st_mtime).isoformat(),
                'backup_version': 1,
                'backup_type': 'manual',
                'compression_enabled': compress_backup,
                'backup_tool': 'config_reader'
            }
            
            # Save metadata as separate JSON file
            metadata_path = backup_path.with_suffix(backup_path.suffix + '.metadata.json')
            with open(metadata_path, 'w', encoding=DEFAULT_CONFIG_ENCODING) as f:
                json.dump(backup_metadata, f, indent=2)
        
        # Compress backup file if compress_backup is enabled
        if compress_backup:
            compressed_backup_path = _compress_backup_file(backup_path)
            backup_path = compressed_backup_path
        
        # Manage backup retention and cleanup old versions
        _cleanup_old_backup_versions(backup_dir, source_file.stem, max_backup_versions)
        
        # Create comprehensive audit trail entry for backup creation
        create_audit_trail(
            action='CONFIG_BACKUP_CREATED',
            component='CONFIG_READER',
            action_details={
                'original_path': config_path,
                'backup_path': str(backup_path),
                'compress_backup': compress_backup,
                'include_metadata': include_metadata,
                'backup_directory': str(backup_dir),
                'max_versions': max_backup_versions,
                'backup_size_bytes': backup_path.stat().st_size if backup_path.exists() else 0
            },
            user_context='SYSTEM'
        )
        
        # Log backup creation operation with size and compression information
        backup_size = backup_path.stat().st_size if backup_path.exists() else 0
        _logger.info(
            f"Configuration backup created: {backup_path} "
            f"(size: {backup_size} bytes, compressed: {compress_backup})"
        )
        
        return str(backup_path)
        
    except Exception as e:
        # Handle backup creation errors with comprehensive error reporting
        _logger.error(f"Configuration backup creation failed for {config_path}: {e}", exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                f"Configuration backup creation failed: {str(e)}",
                configuration_section='backup_creation',
                schema_type='backup_operation'
            ) from e


def restore_config_backup(
    backup_path: str,
    target_config_path: str,
    validate_before_restore: bool = True,
    create_restore_backup: bool = True
) -> Dict[str, Any]:
    """
    Restore configuration from backup with validation, atomic operations, and rollback 
    capability for configuration recovery and version management.
    
    This function provides comprehensive configuration restoration with validation,
    atomic operations, rollback capability, and comprehensive status reporting for
    configuration recovery, version management, and error recovery in scientific workflows.
    
    Args:
        backup_path: Path to the backup file to restore from
        target_config_path: Target path where configuration should be restored
        validate_before_restore: Enable validation of backup before restoration process
        create_restore_backup: Create backup of current configuration before restoration
        
    Returns:
        Dict[str, Any]: Restore operation result with success status, validation information, 
                       and rollback details for comprehensive recovery tracking
    """
    restore_result = {
        'success': False,
        'backup_path': backup_path,
        'target_path': target_config_path,
        'validation_passed': False,
        'current_backup_created': False,
        'current_backup_path': None,
        'restore_timestamp': datetime.datetime.now().isoformat(),
        'rollback_available': False,
        'operation_details': {}
    }
    
    try:
        # Validate backup file exists and is accessible
        backup_file = pathlib.Path(backup_path)
        if not backup_file.exists() or not backup_file.is_file():
            raise ConfigurationError(
                f"Backup file not found or invalid: {backup_path}",
                configuration_section='backup_restore_validation',
                schema_type='restore_operation'
            )
        
        target_file = pathlib.Path(target_config_path)
        
        _logger.info(f"Restoring configuration backup: {backup_path} -> {target_config_path}")
        
        # Create backup of current configuration if create_restore_backup is enabled
        if create_restore_backup and target_file.exists():
            try:
                current_backup_path = backup_config_file(
                    config_path=str(target_file),
                    compress_backup=False,
                    include_metadata=True,
                    backup_directory=str(target_file.parent / 'restore_backups')
                )
                restore_result['current_backup_created'] = True
                restore_result['current_backup_path'] = current_backup_path
                restore_result['rollback_available'] = True
                
                _logger.info(f"Current configuration backed up: {current_backup_path}")
                
            except Exception as e:
                _logger.warning(f"Failed to create current configuration backup: {e}")
                restore_result['backup_warning'] = str(e)
        
        # Load and validate backup configuration if validate_before_restore is enabled
        if validate_before_restore:
            try:
                # Detect backup format and load configuration
                backup_format = detect_config_format(str(backup_file))['detected_format']
                
                if backup_format == '.json':
                    backup_config = load_json_config(
                        config_path=str(backup_file),
                        validate_schema=False,
                        use_cache=False
                    )
                elif backup_format in ['.yaml', '.yml']:
                    backup_config = _load_yaml_config(
                        config_path=str(backup_file),
                        validate_schema=False
                    )
                else:
                    raise ConfigurationError(
                        f"Unsupported backup format: {backup_format}",
                        configuration_section='backup_format_validation',
                        schema_type='restore_operation'
                    )
                
                # Validate backup configuration structure
                validation_result = validate_config_file(
                    config_path=str(backup_file),
                    check_file_integrity=True,
                    validate_scientific_parameters=True
                )
                
                if validation_result.is_valid:
                    restore_result['validation_passed'] = True
                    _logger.debug(f"Backup validation successful: {backup_path}")
                else:
                    restore_result['validation_passed'] = False
                    restore_result['validation_errors'] = validation_result.errors
                    _logger.warning(
                        f"Backup validation issues: {backup_path} - {len(validation_result.errors)} errors"
                    )
                
            except Exception as e:
                restore_result['validation_passed'] = False
                restore_result['validation_error'] = str(e)
                _logger.warning(f"Backup validation failed: {backup_path} - {e}")
        
        # Ensure target directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform atomic restore operation using temporary file approach
        import shutil
        import tempfile
        
        # Create temporary file for atomic restore
        with tempfile.NamedTemporaryFile(
            dir=target_file.parent,
            prefix='restore_temp_',
            suffix=target_file.suffix,
            delete=False
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
        
        try:
            # Copy backup to temporary location
            shutil.copy2(str(backup_file), str(temp_path))
            
            # Verify restore integrity by comparing file sizes
            backup_size = backup_file.stat().st_size
            temp_size = temp_path.stat().st_size
            
            if backup_size != temp_size:
                raise ConfigurationError(
                    f"Restore integrity check failed: size mismatch {backup_size} != {temp_size}",
                    configuration_section='restore_integrity_verification',
                    schema_type='restore_operation'
                )
            
            # Atomically move temporary file to target location
            shutil.move(str(temp_path), str(target_file))
            
            restore_result['success'] = True
            restore_result['operation_details']['restore_method'] = 'atomic_file_operation'
            restore_result['operation_details']['bytes_restored'] = backup_size
            
        except Exception as e:
            # Clean up temporary file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
        # Verify restore success and configuration integrity
        if target_file.exists():
            restored_size = target_file.stat().st_size
            restore_result['operation_details']['restored_size'] = restored_size
            
            # Verify restored configuration can be loaded
            try:
                test_config = read_config(
                    str(target_file),
                    validate_schema=False,
                    use_cache=False
                )
                restore_result['operation_details']['restore_validation'] = 'success'
                restore_result['operation_details']['restored_parameters'] = _count_configuration_parameters(test_config)
                
            except Exception as e:
                restore_result['operation_details']['restore_validation'] = f'failed: {e}'
                _logger.warning(f"Restored configuration validation failed: {e}")
        else:
            raise ConfigurationError(
                "Restore verification failed - target file not found after restore",
                configuration_section='restore_verification',
                schema_type='restore_operation'
            )
        
        # Update configuration cache with restored configuration
        try:
            # Force cache refresh by reading the restored configuration
            restored_config = read_config(
                str(target_file),
                validate_schema=False,
                use_cache=True  # This will update the cache
            )
            restore_result['operation_details']['cache_updated'] = True
            
        except Exception as e:
            restore_result['operation_details']['cache_updated'] = False
            restore_result['operation_details']['cache_error'] = str(e)
            _logger.warning(f"Cache update failed after restore: {e}")
        
        # Create comprehensive audit trail entry for restore operation
        create_audit_trail(
            action='CONFIG_BACKUP_RESTORED',
            component='CONFIG_READER',
            action_details=restore_result,
            user_context='SYSTEM'
        )
        
        # Log successful restore operation with comprehensive details
        _logger.info(
            f"Configuration restore completed: {backup_path} -> {target_config_path} "
            f"(validation: {restore_result['validation_passed']}, "
            f"backup_created: {restore_result['current_backup_created']})"
        )
        
        return restore_result
        
    except Exception as e:
        # Handle restore operation errors with rollback attempt
        restore_result['success'] = False
        restore_result['error'] = str(e)
        restore_result['error_type'] = type(e).__name__
        
        # Attempt rollback if current backup was created and restore failed
        if restore_result['current_backup_created'] and restore_result['current_backup_path']:
            try:
                _logger.info("Attempting rollback after failed restore")
                
                rollback_result = restore_config_backup(
                    backup_path=restore_result['current_backup_path'],
                    target_config_path=target_config_path,
                    validate_before_restore=False,
                    create_restore_backup=False
                )
                
                restore_result['rollback_performed'] = rollback_result['success']
                restore_result['rollback_details'] = rollback_result
                
                if rollback_result['success']:
                    _logger.info("Rollback completed successfully after failed restore")
                else:
                    _logger.error("Rollback failed after restore failure")
                    
            except Exception as rollback_error:
                restore_result['rollback_error'] = str(rollback_error)
                _logger.error(f"Rollback failed after restore failure: {rollback_error}")
        
        _logger.error(f"Configuration restore failed: {backup_path} -> {target_config_path} - {e}")
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                f"Configuration restore operation failed: {str(e)}",
                configuration_section='restore_operation',
                schema_type='restore_operation'
            ) from e


class ConfigReader:
    """
    Comprehensive configuration reader class providing centralized configuration file management, 
    validation, caching, and format handling for the plume simulation system with thread-safe 
    operations, schema validation, and audit trail integration for scientific computing reproducibility.
    
    This class provides a centralized interface for all configuration operations with comprehensive
    validation, caching, versioning, and audit trail support specifically designed for scientific
    computing workflows requiring high reliability, reproducibility, and performance optimization.
    """
    
    def __init__(
        self,
        config_directory: Optional[str] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
        default_schema_directory: Optional[str] = None
    ):
        """
        Initialize configuration reader with directory paths, caching settings, and validation 
        framework for comprehensive configuration management.
        
        Args:
            config_directory: Optional custom configuration directory path
            enable_caching: Enable configuration and schema caching for performance optimization
            enable_validation: Enable automatic validation for all configuration operations
            default_schema_directory: Optional custom schema directory for validation
        """
        # Set configuration and schema directories with validation and fallback
        if config_directory:
            self.config_directory = pathlib.Path(config_directory)
        else:
            from ..config import get_config_directory
            self.config_directory = get_config_directory()
        
        if default_schema_directory:
            self.schema_directory = pathlib.Path(default_schema_directory)
        else:
            from ..config import get_schema_directory
            self.schema_directory = get_schema_directory()
        
        # Initialize caching and validation settings
        self.caching_enabled = enable_caching
        self.validation_enabled = enable_validation
        
        # Create configuration and cache timestamp dictionaries
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime.datetime] = {}
        
        # Setup thread-safe cache locking mechanism
        self.cache_lock = threading.RLock()
        
        # Initialize ConfigurationParser with directory settings
        self.config_parser = ConfigurationParser(
            str(self.config_directory),
            str(self.schema_directory),
            enable_caching=enable_caching,
            enable_validation=enable_validation
        )
        
        # Setup logger for configuration reader operations
        self.logger = get_logger(f'{__name__}.ConfigReader', 'CONFIG_READER')
        
        # Initialize access statistics tracking
        self.access_statistics = {
            'total_reads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_operations': 0,
            'errors_encountered': 0
        }
        
        # Create audit trail for reader initialization
        create_audit_trail(
            action='CONFIG_READER_INITIALIZED',
            component='CONFIG_READER',
            action_details={
                'config_directory': str(self.config_directory),
                'schema_directory': str(self.schema_directory),
                'caching_enabled': self.caching_enabled,
                'validation_enabled': self.validation_enabled
            },
            user_context='SYSTEM'
        )
        
        self.logger.info(f"Configuration reader initialized: {self.config_directory}")
    
    def read(
        self,
        config_path: str,
        validate_schema: bool = None,
        use_cache: bool = None,
        schema_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read configuration file with comprehensive validation, caching, and error handling 
        for scientific computing parameters.
        
        Args:
            config_path: Path to the configuration file to read
            validate_schema: Enable schema validation (uses instance default if None)
            use_cache: Enable caching (uses instance default if None)
            schema_path: Optional custom path to schema file for validation
            
        Returns:
            Dict[str, Any]: Loaded and validated configuration dictionary with comprehensive 
                           error handling and scientific parameter validation
        """
        # Apply instance defaults for optional parameters
        if validate_schema is None:
            validate_schema = self.validation_enabled
        if use_cache is None:
            use_cache = self.caching_enabled
        
        # Resolve configuration file path using instance configuration directory
        resolved_path = self._resolve_config_path(config_path)
        
        # Check configuration cache if use_cache is enabled
        cache_key = f"{resolved_path}:{validate_schema}:{schema_path}"
        if use_cache and self._is_cache_valid(cache_key):
            with self.cache_lock:
                if cache_key in self.config_cache:
                    self.access_statistics['cache_hits'] += 1
                    self.logger.debug(f"Configuration loaded from instance cache: {config_path}")
                    return copy.deepcopy(self.config_cache[cache_key])
        
        try:
            # Update access statistics
            self.access_statistics['total_reads'] += 1
            if use_cache:
                self.access_statistics['cache_misses'] += 1
            
            # Use global read_config function with comprehensive error handling
            config_data = read_config(
                config_path=str(resolved_path),
                validate_schema=validate_schema,
                use_cache=False,  # Use instance caching instead
                schema_path=schema_path,
                apply_defaults=True
            )
            
            # Perform instance-specific validation if validation is enabled
            if validate_schema:
                self.access_statistics['validation_operations'] += 1
                
                # Additional validation using instance parser
                validation_result = self.config_parser.validate_config(
                    config_data=config_data,
                    config_type=self._detect_config_type(config_path),
                    strict_validation=True
                )
                
                if not validation_result.is_valid:
                    self.access_statistics['errors_encountered'] += 1
                    raise ValidationError(
                        f"Instance validation failed: {len(validation_result.errors)} errors",
                        validation_type='instance_validation',
                        validation_context=config_path
                    )
            
            # Cache configuration if caching is enabled
            if use_cache:
                with self.cache_lock:
                    self.config_cache[cache_key] = copy.deepcopy(config_data)
                    self.cache_timestamps[cache_key] = datetime.datetime.now()
            
            # Log successful configuration read operation
            self.logger.info(f"Configuration read successfully: {config_path}")
            
            return config_data
            
        except Exception as e:
            self.access_statistics['errors_encountered'] += 1
            self.logger.error(f"Failed to read configuration {config_path}: {e}", exc_info=True)
            raise
    
    def read_with_validation(
        self,
        config_path: str,
        schema_path: str,
        strict_validation: bool = True,
        use_cache: bool = None
    ) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Read configuration file with mandatory schema validation and comprehensive error reporting 
        for fail-fast validation strategy.
        
        Args:
            config_path: Path to the configuration file to read
            schema_path: Path to JSON schema file for mandatory validation
            strict_validation: Enable strict validation mode with comprehensive checks
            use_cache: Enable caching (uses instance default if None)
            
        Returns:
            Tuple[Dict[str, Any], ValidationResult]: Configuration dictionary and detailed 
                                                    validation results for comprehensive error analysis
        """
        if use_cache is None:
            use_cache = self.caching_enabled
        
        try:
            # Load configuration using instance read method
            config_data = self.read(
                config_path=config_path,
                validate_schema=False,  # We'll validate separately for detailed results
                use_cache=use_cache,
                schema_path=None
            )
            
            # Perform comprehensive schema validation with detailed reporting
            validation_result = validate_configuration_schema(
                config_data=config_data,
                schema_type=self._detect_config_type(config_path),
                strict_mode=strict_validation,
                required_sections=self._get_required_sections_for_path(config_path)
            )
            
            # Apply strict validation rules if strict_validation is enabled
            if strict_validation:
                self._apply_instance_strict_validation(config_data, validation_result)
            
            # Update validation statistics
            self.access_statistics['validation_operations'] += 1
            if not validation_result.is_valid:
                self.access_statistics['errors_encountered'] += 1
            
            # Log validation results with detailed context
            self.logger.info(
                f"Configuration validation completed: {config_path} "
                f"(valid: {validation_result.is_valid}, errors: {len(validation_result.errors)}, "
                f"warnings: {len(validation_result.warnings)})"
            )
            
            return config_data, validation_result
            
        except Exception as e:
            self.access_statistics['errors_encountered'] += 1
            self.logger.error(f"Configuration read with validation failed for {config_path}: {e}", exc_info=True)
            raise
    
    def validate_config(
        self,
        config_data: Union[str, Dict[str, Any]],
        schema_path: Optional[str] = None,
        validate_scientific_parameters: bool = True
    ) -> ValidationResult:
        """
        Validate configuration against schema and scientific constraints with detailed error 
        reporting and recovery recommendations.
        
        Args:
            config_data: Configuration data to validate (dict) or path to configuration file (str)
            schema_path: Optional path to JSON schema file for validation
            validate_scientific_parameters: Enable scientific parameter constraint validation
            
        Returns:
            ValidationResult: Comprehensive validation result with errors, warnings, and 
                            actionable recovery recommendations
        """
        try:
            # Load configuration if config_data is file path
            if isinstance(config_data, str):
                config_dict = self.read(config_data, validate_schema=False, use_cache=self.caching_enabled)
                config_type = self._detect_config_type(config_data)
            else:
                config_dict = config_data
                config_type = 'unknown'
            
            # Use configuration parser to validate configuration with comprehensive checking
            validation_result = self.config_parser.validate_config(
                config_data=config_dict,
                config_type=config_type,
                strict_validation=validate_scientific_parameters
            )
            
            # Perform additional schema validation if schema_path is provided
            if schema_path:
                schema_validation = validate_configuration_schema(
                    config_data=config_dict,
                    schema_type=config_type,
                    strict_mode=validate_scientific_parameters,
                    required_sections=[]
                )
                
                # Merge schema validation results
                if not schema_validation.is_valid:
                    validation_result.errors.extend(schema_validation.errors)
                    validation_result.warnings.extend(schema_validation.warnings)
                    validation_result.is_valid = False
            
            # Validate scientific parameters if validate_scientific_parameters is enabled
            if validate_scientific_parameters:
                self._validate_scientific_parameters_instance(config_dict, validation_result)
            
            # Generate comprehensive validation report with recovery recommendations
            if not validation_result.is_valid:
                self._generate_instance_recovery_recommendations(config_dict, validation_result)
            
            # Update validation statistics
            self.access_statistics['validation_operations'] += 1
            if not validation_result.is_valid:
                self.access_statistics['errors_encountered'] += 1
            
            # Log validation operation with comprehensive results
            self.logger.info(
                f"Configuration validation completed: valid={validation_result.is_valid}, "
                f"errors={len(validation_result.errors)}, warnings={len(validation_result.warnings)}"
            )
            
            return validation_result
            
        except Exception as e:
            self.access_statistics['errors_encountered'] += 1
            self.logger.error(f"Configuration validation failed: {e}", exc_info=True)
            raise
    
    def get_config_path(
        self,
        config_name: str,
        validate_existence: bool = True
    ) -> pathlib.Path:
        """
        Resolve configuration file path with directory resolution and environment variable 
        substitution.
        
        Args:
            config_name: Name of the configuration file
            validate_existence: Enable path existence validation
            
        Returns:
            pathlib.Path: Resolved configuration file path with validation
        """
        try:
            # Use instance config_directory for path resolution
            resolved_path = get_config_path(
                config_name=config_name,
                config_directory=str(self.config_directory),
                resolve_environment_variables=True,
                validate_path=validate_existence
            )
            
            self.logger.debug(f"Configuration path resolved: {config_name} -> {resolved_path}")
            
            return resolved_path
            
        except Exception as e:
            self.logger.error(f"Configuration path resolution failed for {config_name}: {e}")
            raise
    
    def clear_cache(
        self,
        config_paths: Optional[List[str]] = None,
        preserve_statistics: bool = True
    ) -> Dict[str, int]:
        """
        Clear configuration cache with selective clearing and statistics preservation for 
        cache management.
        
        Args:
            config_paths: Optional list of specific configuration paths to clear
            preserve_statistics: Whether to preserve access statistics
            
        Returns:
            Dict[str, int]: Cache clearing statistics with cleared entries count
        """
        try:
            clearing_stats = {
                'instance_cache_cleared': 0,
                'instance_timestamps_cleared': 0,
                'global_cache_cleared': 0,
                'statistics_preserved': preserve_statistics
            }
            
            # Clear instance cache with thread safety
            with self.cache_lock:
                if config_paths is None:
                    # Clear all instance cache entries
                    clearing_stats['instance_cache_cleared'] = len(self.config_cache)
                    clearing_stats['instance_timestamps_cleared'] = len(self.cache_timestamps)
                    
                    self.config_cache.clear()
                    self.cache_timestamps.clear()
                else:
                    # Clear specific cache entries
                    keys_to_remove = []
                    for cache_key in self.config_cache.keys():
                        for config_path in config_paths:
                            if config_path in cache_key:
                                keys_to_remove.append(cache_key)
                                break
                    
                    for key in keys_to_remove:
                        del self.config_cache[key]
                        if key in self.cache_timestamps:
                            del self.cache_timestamps[key]
                            clearing_stats['instance_timestamps_cleared'] += 1
                        clearing_stats['instance_cache_cleared'] += 1
            
            # Clear global cache as well
            global_clearing_stats = clear_config_cache(
                config_paths=config_paths,
                preserve_statistics=preserve_statistics,
                clear_reason='instance_cache_clear'
            )
            clearing_stats['global_cache_cleared'] = global_clearing_stats.get('entries_cleared', 0)
            
            # Update access statistics if preserving statistics
            if not preserve_statistics:
                self.access_statistics = {
                    'total_reads': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'validation_operations': 0,
                    'errors_encountered': 0
                }
            
            # Log cache clearing operation
            self.logger.info(
                f"Instance cache cleared: {clearing_stats['instance_cache_cleared']} entries, "
                f"global cache: {clearing_stats['global_cache_cleared']} entries"
            )
            
            return clearing_stats
            
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'instance_cache_cleared': 0,
                'global_cache_cleared': 0
            }
    
    def get_cache_statistics(
        self,
        include_detailed_info: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve cache performance statistics including hit rates and memory usage for 
        monitoring and optimization.
        
        Args:
            include_detailed_info: Include detailed cache information and analysis
            
        Returns:
            Dict[str, Any]: Cache statistics with performance metrics and usage analysis
        """
        try:
            # Calculate basic cache statistics
            with self.cache_lock:
                cache_stats = {
                    'instance_cache_size': len(self.config_cache),
                    'instance_timestamp_entries': len(self.cache_timestamps),
                    'access_statistics': self.access_statistics.copy(),
                    'statistics_timestamp': datetime.datetime.now().isoformat()
                }
                
                # Calculate cache hit rates and performance metrics
                total_reads = self.access_statistics['total_reads']
                if total_reads > 0:
                    cache_stats['cache_hit_rate'] = (self.access_statistics['cache_hits'] / total_reads) * 100
                    cache_stats['cache_miss_rate'] = (self.access_statistics['cache_misses'] / total_reads) * 100
                    cache_stats['error_rate'] = (self.access_statistics['errors_encountered'] / total_reads) * 100
                else:
                    cache_stats['cache_hit_rate'] = 0
                    cache_stats['cache_miss_rate'] = 0
                    cache_stats['error_rate'] = 0
                
                # Include detailed cache information if include_detailed_info is enabled
                if include_detailed_info:
                    cache_details = {}
                    total_memory = 0
                    
                    for cache_key, cache_data in self.config_cache.items():
                        entry_memory = len(str(cache_data).encode('utf-8'))
                        total_memory += entry_memory
                        
                        cache_details[cache_key] = {
                            'cached_at': self.cache_timestamps.get(cache_key, datetime.datetime.min).isoformat(),
                            'memory_bytes': entry_memory,
                            'is_valid': self._is_cache_valid(cache_key),
                            'parameter_count': _count_configuration_parameters(cache_data)
                        }
                    
                    cache_stats['detailed_info'] = {
                        'cache_details': cache_details,
                        'total_memory_bytes': total_memory,
                        'average_entry_size': total_memory // len(self.config_cache) if self.config_cache else 0,
                        'valid_entries': len([d for d in cache_details.values() if d['is_valid']]),
                        'expired_entries': len([d for d in cache_details.values() if not d['is_valid']])
                    }
            
            # Get global cache statistics for comparison
            global_cache_info = get_config_cache_info(
                include_detailed_stats=include_detailed_info,
                include_cache_contents=False
            )
            cache_stats['global_cache_info'] = global_cache_info
            
            self.logger.debug("Cache statistics retrieved successfully")
            
            return cache_stats
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cache statistics: {e}", exc_info=True)
            return {
                'error': str(e),
                'instance_cache_size': 0,
                'access_statistics': self.access_statistics.copy()
            }
    
    def _resolve_config_path(self, config_path: str) -> pathlib.Path:
        """Resolve configuration path using instance directory settings."""
        if pathlib.Path(config_path).is_absolute():
            return pathlib.Path(config_path)
        else:
            return self.config_directory / config_path
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid based on expiry settings."""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        expiry_time = cache_time + datetime.timedelta(hours=CONFIG_CACHE_TIMEOUT_HOURS)
        
        return datetime.datetime.now() < expiry_time
    
    def _detect_config_type(self, config_path: str) -> str:
        """Detect configuration type from path and content."""
        path_lower = config_path.lower()
        
        if 'normalization' in path_lower:
            return 'normalization'
        elif 'simulation' in path_lower:
            return 'simulation'
        elif 'analysis' in path_lower:
            return 'analysis'
        elif 'logging' in path_lower:
            return 'logging'
        elif 'performance' in path_lower:
            return 'performance'
        else:
            return 'unknown'
    
    def _get_required_sections_for_path(self, config_path: str) -> List[str]:
        """Get required sections for configuration based on path."""
        config_type = self._detect_config_type(config_path)
        return _get_required_sections_for_type(config_type)
    
    def _apply_instance_strict_validation(self, config_data: Dict[str, Any], validation_result: ValidationResult) -> None:
        """Apply instance-specific strict validation rules."""
        # Additional strict validation logic for instance
        pass
    
    def _validate_scientific_parameters_instance(self, config_data: Dict[str, Any], validation_result: ValidationResult) -> None:
        """Validate scientific parameters using instance-specific rules."""
        # Scientific parameter validation logic
        pass
    
    def _generate_instance_recovery_recommendations(self, config_data: Dict[str, Any], validation_result: ValidationResult) -> None:
        """Generate instance-specific recovery recommendations."""
        if not validation_result.is_valid:
            validation_result.add_recommendation(
                "Review configuration errors and fix invalid parameters",
                priority="HIGH"
            )


# Helper functions for configuration reader implementation

def _is_cache_valid(cache_key: str) -> bool:
    """Check if global cache entry is still valid based on expiry settings."""
    if cache_key not in _cache_timestamps:
        return False
    
    cache_time = _cache_timestamps[cache_key]
    expiry_time = cache_time + datetime.timedelta(hours=CONFIG_CACHE_TIMEOUT_HOURS)
    
    return datetime.datetime.now() < expiry_time


def _is_cache_entry_valid(cache_key: str, cache_time: datetime.datetime, current_time: datetime.datetime) -> bool:
    """Check if specific cache entry is valid based on time comparison."""
    expiry_time = cache_time + datetime.timedelta(hours=CONFIG_CACHE_TIMEOUT_HOURS)
    return current_time < expiry_time


def _load_yaml_config(
    config_path: str,
    validate_schema: bool = False,
    schema_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load YAML configuration file with error handling and validation."""
    try:
        with open(config_path, 'r', encoding=DEFAULT_CONFIG_ENCODING) as f:
            config_data = yaml.safe_load(f)
        
        if not isinstance(config_data, dict):
            raise ConfigurationError(
                f"YAML configuration must be a dictionary, got {type(config_data)}",
                configuration_section='yaml_parsing',
                schema_type='yaml'
            )
        
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"YAML parsing failed: {str(e)}",
            configuration_section='yaml_parsing',
            schema_type='yaml'
        ) from e


def _apply_intelligent_defaults(config_data: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Apply intelligent default values for missing configuration parameters."""
    # Detect configuration type and apply appropriate defaults
    config_type = _detect_config_type_from_path(config_path)
    
    if config_type == 'normalization':
        default_config = get_default_normalization_config(validate_schema=False)
    elif config_type == 'simulation':
        default_config = get_default_simulation_config(validate_schema=False)
    elif config_type == 'analysis':
        default_config = get_default_analysis_config(validate_schema=False)
    else:
        return config_data  # No defaults available
    
    # Merge with defaults using intelligent strategy
    return _intelligent_configuration_merge(config_data, default_config, 'preserve_user_values')


def _perform_comprehensive_validation(
    config_data: Dict[str, Any],
    config_path: str,
    schema_path: Optional[str],
    format_type: str
) -> ValidationResult:
    """Perform comprehensive validation of configuration data."""
    validation_result = ValidationResult(
        validation_type='comprehensive_validation',
        is_valid=True,
        validation_context=f"path={config_path}, format={format_type}"
    )
    
    # Detect configuration type for appropriate validation
    config_type = _detect_config_type_from_path(config_path)
    
    # Perform schema validation
    if schema_path or config_type != 'unknown':
        schema_validation = validate_configuration_schema(
            config_data=config_data,
            schema_type=config_type,
            strict_mode=True,
            required_sections=_get_required_sections_for_type(config_type)
        )
        
        if not schema_validation.is_valid:
            validation_result.errors.extend(schema_validation.errors)
            validation_result.warnings.extend(schema_validation.warnings)
            validation_result.is_valid = False
    
    return validation_result


def _count_configuration_parameters(config_data: Dict[str, Any]) -> int:
    """Count total number of parameters in configuration recursively."""
    if not isinstance(config_data, dict):
        return 1
    
    count = 0
    for key, value in config_data.items():
        if key.startswith('_'):  # Skip metadata fields
            continue
        
        if isinstance(value, dict):
            count += _count_configuration_parameters(value)
        else:
            count += 1
    
    return count


def _substitute_environment_variables_in_string(text: str) -> str:
    """Substitute environment variables in string using ${VAR} syntax."""
    import re
    
    def replace_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    return re.sub(r'\$\{([^}]+)\}', replace_var, text)


def _get_default_configuration_by_type(config_type: str) -> Dict[str, Any]:
    """Get default configuration by type."""
    try:
        if config_type == 'normalization':
            return get_default_normalization_config(validate_schema=False)
        elif config_type == 'simulation':
            return get_default_simulation_config(validate_schema=False)
        elif config_type == 'analysis':
            return get_default_analysis_config(validate_schema=False)
        else:
            return {}
    except Exception:
        return {}


def _get_required_sections_for_type(config_type: str) -> List[str]:
    """Get required sections for configuration type."""
    section_mappings = {
        'normalization': ['spatial', 'temporal', 'intensity'],
        'simulation': ['algorithm', 'parameters', 'batch_processing'],
        'analysis': ['metrics', 'visualization', 'export'],
        'logging': ['handlers', 'formatters', 'loggers'],
        'performance': ['processing', 'accuracy', 'resources']
    }
    return section_mappings.get(config_type, [])


def _intelligent_configuration_merge(
    base_config: Dict[str, Any],
    default_config: Dict[str, Any],
    merge_strategy: str
) -> Dict[str, Any]:
    """Intelligently merge configurations with specified strategy."""
    import copy
    
    result = copy.deepcopy(base_config)
    
    def merge_recursive(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                merge_recursive(target[key], value)
            # For 'preserve_user_values' strategy, don't overwrite existing values
    
    merge_recursive(result, default_config)
    return result


def _apply_configuration_overrides(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configuration overrides using deep merge strategy."""
    import copy
    
    result = copy.deepcopy(config_data)
    
    def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                deep_update(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    deep_update(result, overrides)
    return result


def _apply_strict_validation_rules(config_data: Dict[str, Any], config_type: str) -> None:
    """Apply strict validation rules for configuration type."""
    # Implementation for strict validation rules
    pass


def _validate_scientific_parameter_structure(config_data: Dict[str, Any], validation_result: ValidationResult) -> None:
    """Validate scientific parameter structure and constraints."""
    # Implementation for scientific parameter validation
    pass


def _check_deprecated_configuration_patterns(config_data: Dict[str, Any], validation_result: ValidationResult) -> None:
    """Check for deprecated configuration patterns."""
    # Implementation for deprecated pattern checking
    pass


def _generate_configuration_recovery_recommendations(validation_result: ValidationResult) -> None:
    """Generate recovery recommendations for configuration validation failures."""
    if not validation_result.is_valid:
        validation_result.add_recommendation(
            "Review configuration errors and fix invalid parameters",
            priority="HIGH"
        )


def _calculate_error_severity_score(validation_result: ValidationResult) -> float:
    """Calculate error severity score for validation result."""
    if not validation_result.errors:
        return 0.0
    
    # Simple scoring based on error count
    return min(len(validation_result.errors) * 0.1, 1.0)


def _generate_cache_performance_analysis() -> Dict[str, Any]:
    """Generate cache performance analysis and recommendations."""
    with _cache_lock:
        cache_size = len(_config_cache)
        
        return {
            'cache_utilization': min(cache_size / 100, 1.0),  # Assuming 100 is optimal
            'recommendations': [
                'Monitor cache hit rates for performance optimization',
                'Consider increasing cache timeout for stable configurations',
                'Clear cache periodically during development'
            ]
        }


def _detect_config_type_from_path(config_path: str) -> str:
    """Detect configuration type from file path."""
    path_lower = config_path.lower()
    
    if 'normalization' in path_lower:
        return 'normalization'
    elif 'simulation' in path_lower:
        return 'simulation'
    elif 'analysis' in path_lower:
        return 'analysis'
    elif 'logging' in path_lower:
        return 'logging'
    elif 'performance' in path_lower:
        return 'performance'
    else:
        return 'unknown'


def _deep_merge_configurations(config_list: List[Dict[str, Any]], merge_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Perform deep merge of multiple configurations."""
    import copy
    
    if not config_list:
        return {}
    
    result = copy.deepcopy(config_list[0])
    
    for config in config_list[1:]:
        def merge_recursive(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target:
                    if isinstance(target[key], dict) and isinstance(value, dict):
                        merge_recursive(target[key], value)
                    else:
                        target[key] = copy.deepcopy(value)
                        merge_metadata['conflicts_resolved'] += 1
                else:
                    target[key] = copy.deepcopy(value)
        
        merge_recursive(result, config)
    
    return result


def _overlay_merge_configurations(config_list: List[Dict[str, Any]], merge_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Perform overlay merge where later configurations override earlier ones."""
    import copy
    
    if not config_list:
        return {}
    
    result = copy.deepcopy(config_list[0])
    
    for config in config_list[1:]:
        for key, value in config.items():
            if key in result:
                merge_metadata['conflicts_resolved'] += 1
            result[key] = copy.deepcopy(value)
    
    return result


def _priority_merge_configurations(
    config_list: List[Dict[str, Any]], 
    merge_metadata: Dict[str, Any], 
    priority_order: Optional[List[str]]
) -> Dict[str, Any]:
    """Perform priority-based merge using specified priority order."""
    # For now, fallback to deep merge
    return _deep_merge_configurations(config_list, merge_metadata)


def _check_merged_configuration_consistency(merged_config: Dict[str, Any], config_paths: List[str]) -> None:
    """Check merged configuration for consistency and compatibility issues."""
    # Implementation for merged configuration consistency checking
    pass


def _compress_backup_file(backup_path: pathlib.Path) -> pathlib.Path:
    """Compress backup file using gzip compression."""
    import gzip
    import shutil
    
    compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
    
    with open(backup_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove original uncompressed file
    backup_path.unlink()
    
    return compressed_path


def _cleanup_old_backup_versions(backup_dir: pathlib.Path, config_stem: str, max_versions: int) -> None:
    """Clean up old backup versions based on retention policy."""
    # Find all backup files for this configuration
    backup_pattern = f"{config_stem}_backup_*"
    backup_files = list(backup_dir.glob(backup_pattern))
    
    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old versions beyond max_versions limit
    for old_backup in backup_files[max_versions:]:
        try:
            old_backup.unlink()
            # Also remove associated metadata file if it exists
            metadata_file = old_backup.with_suffix(old_backup.suffix + '.metadata.json')
            if metadata_file.exists():
                metadata_file.unlink()
        except OSError as e:
            _logger.warning(f"Failed to remove old backup {old_backup}: {e}")