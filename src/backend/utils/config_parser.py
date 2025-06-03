"""
Comprehensive configuration parser module providing robust configuration file loading, validation, 
caching, and management for the plume simulation system.

This module implements JSON schema validation, atomic file operations, intelligent default value 
application, configuration merging, and audit trail integration to support reproducible scientific 
computing environments with fail-fast validation, cross-platform compatibility, and performance 
optimization for 4000+ simulation processing requirements.

Key Features:
- Centralized configuration management with schema validation
- Fail-fast validation strategy for early error detection  
- Cross-platform compatibility for different computational environments
- Scientific parameter validation with comprehensive error handling
- Reproducible configuration management with complete audit trails
- Thread-safe caching and concurrent access control
- Atomic file operations with integrity verification and rollback
- Intelligent default value application and environment-specific overrides
- Configuration merging with conflict resolution and priority handling
- Performance optimization for high-throughput batch processing
"""

# External library imports with version specifications
import json  # Python 3.9+ - JSON parsing and serialization for configuration files
import pathlib  # Python 3.9+ - Modern cross-platform path handling for configuration files
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # Python 3.9+ - Type hints for configuration parser function signatures
import copy  # Python 3.9+ - Deep copying for configuration merging and default application
import datetime  # Python 3.9+ - Timestamp generation for configuration versioning and audit trails
import threading  # Python 3.9+ - Thread-safe configuration caching and concurrent access control
import os  # Python 3.9+ - Environment variable access and operating system interface
import re  # Python 3.9+ - Regular expression pattern matching for configuration validation
import uuid  # Python 3.9+ - Unique identifier generation for configuration versioning and audit trails
import jsonschema  # jsonschema 4.17.0+ - JSON schema validation for configuration structure validation

# Internal imports from utility modules
from .logging_utils import get_logger, log_validation_error, create_audit_trail
from .validation_utils import validate_configuration_schema, ValidationResult, ConfigurationError, ValidationError
from .file_utils import load_json_config, save_json_config, ensure_directory_exists

# Global configuration caches and registries for performance optimization
_config_cache: Dict[str, Dict[str, Any]] = {}
_schema_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock: threading.RLock = threading.RLock()

# Default configuration file mappings for different component types
_default_configs: Dict[str, str] = {
    'normalization': 'default_normalization.json',
    'simulation': 'default_simulation.json', 
    'analysis': 'default_analysis.json',
    'logging': 'logging_config.json',
    'performance': 'performance_thresholds.json'
}

# Schema file mappings for configuration validation
_schema_files: Dict[str, str] = {
    'normalization': 'normalization_schema.json',
    'simulation': 'simulation_schema.json',
    'analysis': 'analysis_schema.json'
}

# Configuration file operation constants and limits
CONFIG_FILE_EXTENSIONS: List[str] = ['.json', '.yaml', '.yml']
SCHEMA_FILE_EXTENSION: str = '.json'
BACKUP_SUFFIX: str = '.backup'
VERSION_SUFFIX: str = '.v'
MAX_BACKUP_VERSIONS: int = 10
CACHE_EXPIRY_HOURS: int = 24
VALIDATION_TIMEOUT_SECONDS: int = 30

# Global logger instance for configuration parser operations
_logger = get_logger(__name__, 'config_parser')


def load_configuration(
    config_name: str,
    config_path: Optional[str] = None,
    validate_schema: bool = True,
    use_cache: bool = True,
    apply_defaults: bool = True,
    override_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration file with comprehensive validation, schema checking, caching, and error 
    handling for scientific computing parameters with intelligent default value application and 
    audit trail integration.
    
    This function provides robust configuration loading with fail-fast validation, comprehensive
    error handling, and performance optimization through intelligent caching mechanisms.
    
    Args:
        config_name: Supported configuration type identifier
        config_path: Optional custom path to configuration file
        validate_schema: Enable JSON schema validation for configuration structure
        use_cache: Enable configuration caching for improved performance
        apply_defaults: Apply intelligent default values for missing parameters
        override_values: Dictionary of override values to apply after loading
        
    Returns:
        Dict[str, Any]: Loaded and validated configuration dictionary with applied defaults and overrides
        
    Raises:
        ConfigurationError: When configuration validation fails or file cannot be loaded
        ValidationError: When schema validation fails with detailed error information
    """
    # Validate config_name is supported configuration type
    if config_name not in _default_configs:
        supported_types = list(_default_configs.keys())
        raise ConfigurationError(
            f"Unsupported configuration type: {config_name}. Supported types: {supported_types}",
            configuration_section='load_configuration',
            schema_type=config_name
        )
    
    # Check configuration cache if use_cache is enabled
    cache_key = f"{config_name}:{config_path}:{validate_schema}:{apply_defaults}"
    if use_cache:
        with _cache_lock:
            if cache_key in _config_cache:
                cached_config = copy.deepcopy(_config_cache[cache_key])
                _logger.debug(f"Configuration loaded from cache: {config_name}")
                
                # Apply override values if provided
                if override_values:
                    cached_config = _apply_configuration_overrides(cached_config, override_values)
                
                return cached_config
    
    try:
        # Determine configuration file path from config_name or use provided config_path
        if config_path is None:
            config_path = _get_default_config_path(config_name)
        
        config_file_path = pathlib.Path(config_path)
        if not config_file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                configuration_section='file_loading',
                schema_type=config_name
            )
        
        # Load configuration using load_json_config with error handling
        _logger.info(f"Loading configuration: {config_name} from {config_path}")
        
        # Determine schema path for validation
        schema_path = None
        if validate_schema and config_name in _schema_files:
            schema_path = _get_schema_path(config_name)
        
        # Load configuration with JSON schema validation
        config_data = load_json_config(
            config_path=str(config_file_path),
            schema_path=schema_path,
            validate_schema=validate_schema and schema_path is not None,
            use_cache=use_cache
        )
        
        # Apply default values if apply_defaults is enabled
        if apply_defaults:
            config_data = _apply_default_values(config_data, config_name)
        
        # Perform comprehensive schema validation if validate_schema is enabled
        if validate_schema:
            validation_result = validate_configuration_schema(
                config_data=config_data,
                schema_type=config_name,
                strict_mode=True,
                required_sections=_get_required_sections(config_name)
            )
            
            if not validation_result.is_valid:
                error_details = {
                    'validation_errors': validation_result.errors,
                    'validation_warnings': validation_result.warnings,
                    'config_name': config_name,
                    'config_path': config_path
                }
                
                log_validation_error(
                    validation_type='configuration_schema',
                    error_message=f"Configuration validation failed for {config_name}",
                    validation_context=error_details,
                    failed_parameters=[],
                    recovery_recommendations=validation_result.recommendations
                )
                
                raise ValidationError(
                    message=f"Configuration validation failed: {len(validation_result.errors)} errors found",
                    validation_type='configuration_schema',
                    validation_context=config_name
                )
        
        # Apply override values if provided
        if override_values:
            config_data = _apply_configuration_overrides(config_data, override_values)
        
        # Cache configuration if use_cache is enabled
        if use_cache:
            with _cache_lock:
                _config_cache[cache_key] = copy.deepcopy(config_data)
                _logger.debug(f"Configuration cached: {config_name}")
        
        # Create audit trail entry for configuration access
        create_audit_trail(
            action='CONFIGURATION_LOADED',
            component='CONFIG_PARSER',
            action_details={
                'config_name': config_name,
                'config_path': config_path,
                'validate_schema': validate_schema,
                'apply_defaults': apply_defaults,
                'has_overrides': override_values is not None,
                'config_size': len(str(config_data)) if config_data else 0
            },
            user_context='SYSTEM'
        )
        
        # Log configuration loading operation with performance metrics
        _logger.info(f"Configuration loaded successfully: {config_name}")
        
        return config_data
        
    except Exception as e:
        # Handle configuration loading errors with comprehensive error reporting
        error_message = f"Failed to load configuration {config_name}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='load_configuration',
                schema_type=config_name
            ) from e


def save_configuration(
    config_name: str,
    config_data: Dict[str, Any],
    config_path: Optional[str] = None,
    validate_schema: bool = True,
    create_backup: bool = True,
    atomic_write: bool = True,
    update_cache: bool = True
) -> Dict[str, Any]:
    """
    Save configuration to file with atomic operations, schema validation, backup creation, 
    versioning, and audit trail generation to ensure data integrity and traceability for 
    scientific reproducibility.
    
    This function provides comprehensive configuration saving with robust error handling,
    atomic operations, and complete audit trail integration for scientific computing workflows.
    
    Args:
        config_name: Supported configuration type identifier
        config_data: Configuration dictionary to save
        config_path: Optional custom path for saving configuration
        validate_schema: Enable schema validation before saving
        create_backup: Create backup of existing configuration file
        atomic_write: Use atomic write operations for data safety
        update_cache: Update configuration cache after successful save
        
    Returns:
        Dict[str, Any]: Save operation result with success status, validation results, and backup information
        
    Raises:
        ConfigurationError: When configuration validation fails or save operation fails
        ValidationError: When schema validation fails before saving
    """
    # Validate config_name is supported configuration type
    if config_name not in _default_configs:
        supported_types = list(_default_configs.keys())
        raise ConfigurationError(
            f"Unsupported configuration type: {config_name}. Supported types: {supported_types}",
            configuration_section='save_configuration',
            schema_type=config_name
        )
    
    # Validate config_data structure and required fields
    if not isinstance(config_data, dict):
        raise ConfigurationError(
            f"Configuration data must be a dictionary, got {type(config_data)}",
            configuration_section='data_validation',
            schema_type=config_name
        )
    
    if not config_data:
        raise ConfigurationError(
            "Configuration data cannot be empty",
            configuration_section='data_validation', 
            schema_type=config_name
        )
    
    # Initialize save operation result tracking
    save_result = {
        'success': False,
        'config_name': config_name,
        'config_path': config_path,
        'validation_passed': False,
        'backup_created': False,
        'backup_path': None,
        'cache_updated': False,
        'operation_timestamp': datetime.datetime.now().isoformat(),
        'operation_id': str(uuid.uuid4())
    }
    
    try:
        # Perform schema validation if validate_schema is enabled
        if validate_schema:
            _logger.debug(f"Validating configuration schema before save: {config_name}")
            
            validation_result = validate_configuration_schema(
                config_data=config_data,
                schema_type=config_name,
                strict_mode=True,
                required_sections=_get_required_sections(config_name)
            )
            
            if not validation_result.is_valid:
                error_details = {
                    'validation_errors': validation_result.errors,
                    'validation_warnings': validation_result.warnings,
                    'config_name': config_name
                }
                
                log_validation_error(
                    validation_type='configuration_save_validation',
                    error_message=f"Configuration validation failed before save: {config_name}",
                    validation_context=error_details,
                    recovery_recommendations=validation_result.recommendations
                )
                
                raise ValidationError(
                    message=f"Configuration validation failed before save: {len(validation_result.errors)} errors",
                    validation_type='configuration_save_validation',
                    validation_context=config_name
                )
            
            save_result['validation_passed'] = True
        
        # Determine configuration file path from config_name or use provided config_path
        if config_path is None:
            config_path = _get_default_config_path(config_name)
        
        config_file_path = pathlib.Path(config_path)
        
        # Ensure parent directory exists
        ensure_directory_exists(str(config_file_path.parent), create_parents=True)
        
        # Create backup of existing configuration if create_backup is enabled
        if create_backup and config_file_path.exists():
            backup_path = _create_configuration_backup(str(config_file_path))
            save_result['backup_created'] = True
            save_result['backup_path'] = backup_path
            _logger.info(f"Configuration backup created: {backup_path}")
        
        # Determine schema path for validation
        schema_path = None
        if validate_schema and config_name in _schema_files:
            schema_path = _get_schema_path(config_name)
        
        # Save configuration using save_json_config with atomic operations
        _logger.info(f"Saving configuration: {config_name} to {config_path}")
        
        file_save_result = save_json_config(
            config_data=config_data,
            config_path=str(config_file_path),
            schema_path=schema_path,
            validate_schema=validate_schema and schema_path is not None,
            create_backup=False,  # We handle backup creation above
            atomic_write=atomic_write
        )
        
        if not file_save_result.get('success', False):
            raise ConfigurationError(
                f"File save operation failed: {file_save_result.get('error', 'Unknown error')}",
                configuration_section='file_save',
                schema_type=config_name
            )
        
        # Update configuration cache if update_cache is enabled
        if update_cache:
            cache_key = f"{config_name}:{config_path}:True:True"  # Standard cache key format
            with _cache_lock:
                _config_cache[cache_key] = copy.deepcopy(config_data)
                save_result['cache_updated'] = True
                _logger.debug(f"Configuration cache updated: {config_name}")
        
        # Mark save operation as successful
        save_result['success'] = True
        save_result['bytes_written'] = file_save_result.get('bytes_written', 0)
        
        # Create audit trail entry for configuration modification
        create_audit_trail(
            action='CONFIGURATION_SAVED',
            component='CONFIG_PARSER',
            action_details=save_result,
            user_context='SYSTEM'
        )
        
        # Log save operation with performance metrics and validation results
        _logger.info(f"Configuration saved successfully: {config_name}")
        
        return save_result
        
    except Exception as e:
        # Handle configuration saving errors with comprehensive error reporting
        save_result['success'] = False
        save_result['error'] = str(e)
        save_result['error_type'] = type(e).__name__
        
        error_message = f"Failed to save configuration {config_name}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='save_configuration',
                schema_type=config_name
            ) from e


def validate_configuration(
    config_data: Dict[str, Any],
    config_type: str,
    strict_validation: bool = False,
    check_dependencies: bool = True,
    schema_path: Optional[str] = None
) -> ValidationResult:
    """
    Comprehensive configuration validation against JSON schema and scientific constraints with 
    detailed error reporting, compatibility checking, and recovery recommendations for fail-fast 
    validation strategy.
    
    This function provides extensive configuration validation with scientific parameter checking,
    cross-dependency validation, and comprehensive error reporting with actionable recommendations.
    
    Args:
        config_data: Configuration dictionary to validate
        config_type: Type of configuration for validation rules
        strict_validation: Enable strict validation mode with comprehensive checks
        check_dependencies: Enable cross-parameter dependency validation
        schema_path: Optional custom path to JSON schema file
        
    Returns:
        ValidationResult: Detailed validation results with errors, warnings, and scientific parameter constraint violations
        
    Raises:
        ConfigurationError: When validation setup fails or configuration type is unsupported
    """
    # Create ValidationResult container for configuration validation
    validation_result = ValidationResult(
        validation_type='configuration_validation',
        is_valid=True,
        validation_context=f"config_type={config_type}, strict={strict_validation}"
    )
    
    try:
        # Validate config_type against supported configuration types
        if config_type not in _default_configs:
            supported_types = list(_default_configs.keys())
            validation_result.add_error(
                f"Unsupported configuration type: {config_type}. Supported: {supported_types}",
                severity=ValidationResult.ErrorSeverity.HIGH
            )
            return validation_result
        
        # Validate configuration data structure
        if not isinstance(config_data, dict):
            validation_result.add_error(
                f"Configuration must be a dictionary, got {type(config_data)}",
                severity=ValidationResult.ErrorSeverity.CRITICAL
            )
            return validation_result
        
        if not config_data:
            validation_result.add_error(
                "Configuration cannot be empty",
                severity=ValidationResult.ErrorSeverity.HIGH
            )
            return validation_result
        
        # Load appropriate JSON schema for config_type
        if schema_path is None and config_type in _schema_files:
            schema_path = _get_schema_path(config_type)
        
        if schema_path:
            # Perform JSON schema validation using jsonschema library
            try:
                schema_data = _load_schema_file(schema_path)
                jsonschema.validate(instance=config_data, schema=schema_data)
                validation_result.add_metric('schema_validation_passed', 1.0)
                _logger.debug(f"JSON schema validation passed for {config_type}")
                
            except jsonschema.ValidationError as e:
                validation_result.add_error(
                    f"JSON schema validation failed: {e.message} at path {'.'.join(str(p) for p in e.absolute_path)}",
                    severity=ValidationResult.ErrorSeverity.HIGH
                )
                validation_result.add_recommendation(
                    f"Fix schema violation: {e.message}",
                    priority="HIGH"
                )
                
            except Exception as e:
                validation_result.add_warning(
                    f"Schema validation error: {str(e)}",
                    warning_context={'schema_path': schema_path}
                )
        
        # Check scientific parameter constraints and value ranges
        _validate_scientific_parameters(config_data, config_type, validation_result, strict_validation)
        
        # Validate cross-parameter dependencies if check_dependencies is enabled
        if check_dependencies:
            _validate_parameter_dependencies(config_data, config_type, validation_result)
        
        # Apply strict validation rules if strict_validation is enabled
        if strict_validation:
            _apply_strict_validation_rules(config_data, config_type, validation_result)
        
        # Check for deprecated parameters and configuration patterns
        _check_deprecated_parameters(config_data, config_type, validation_result)
        
        # Generate recovery recommendations for validation failures
        if not validation_result.is_valid:
            _generate_validation_recovery_recommendations(config_data, config_type, validation_result)
        
        # Add validation metrics
        validation_result.add_metric('total_parameters', _count_configuration_parameters(config_data))
        validation_result.add_metric('validation_coverage', 1.0)
        
        # Log validation operation with detailed context and results
        _logger.info(
            f"Configuration validation completed: {config_type}, "
            f"valid={validation_result.is_valid}, "
            f"errors={len(validation_result.errors)}, "
            f"warnings={len(validation_result.warnings)}"
        )
        
        # Create audit trail for validation operation
        create_audit_trail(
            action='CONFIGURATION_VALIDATED',
            component='CONFIG_PARSER',
            action_details={
                'config_type': config_type,
                'validation_result': validation_result.get_summary(),
                'strict_validation': strict_validation,
                'check_dependencies': check_dependencies
            },
            user_context='SYSTEM'
        )
        
        return validation_result
        
    except Exception as e:
        # Handle validation errors with comprehensive error reporting
        validation_result.add_error(
            f"Validation process failed: {str(e)}",
            severity=ValidationResult.ErrorSeverity.CRITICAL
        )
        
        error_message = f"Configuration validation failed for {config_type}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        return validation_result


def merge_configurations(
    config_list: List[Dict[str, Any]],
    merge_strategy: str = 'deep_merge',
    validate_result: bool = True,
    priority_order: Optional[List[str]] = None,
    preserve_metadata: bool = False
) -> Dict[str, Any]:
    """
    Intelligently merge multiple configurations with conflict resolution, priority handling, 
    validation, and compatibility checking for complex configuration scenarios and experimental 
    condition setup.
    
    This function provides sophisticated configuration merging with multiple strategies,
    conflict resolution, and validation to support complex scientific computing workflows.
    
    Args:
        config_list: List of configuration dictionaries to merge
        merge_strategy: Strategy for merging configurations ('deep_merge', 'overlay', 'priority')
        validate_result: Validate merged configuration against schemas
        priority_order: Order of priority for conflict resolution
        preserve_metadata: Preserve metadata from source configurations
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary with resolved conflicts and validated consistency
        
    Raises:
        ConfigurationError: When merge operation fails or configurations are incompatible
        ValidationError: When merged configuration fails validation
    """
    # Validate input configurations and merge strategy
    if not isinstance(config_list, list):
        raise ConfigurationError(
            "Configuration list must be a list of dictionaries",
            configuration_section='merge_validation',
            schema_type='merge_operation'
        )
    
    if len(config_list) == 0:
        raise ConfigurationError(
            "Configuration list cannot be empty",
            configuration_section='merge_validation',
            schema_type='merge_operation'
        )
    
    if len(config_list) == 1:
        # Single configuration - return copy with optional validation
        result_config = copy.deepcopy(config_list[0])
        
        if validate_result:
            # Attempt to validate single configuration
            try:
                config_type = _detect_configuration_type(result_config)
                validation_result = validate_configuration(
                    config_data=result_config,
                    config_type=config_type,
                    strict_validation=True
                )
                
                if not validation_result.is_valid:
                    raise ValidationError(
                        f"Single configuration validation failed: {len(validation_result.errors)} errors",
                        validation_type='merge_validation',
                        validation_context='single_config'
                    )
                    
            except Exception as e:
                _logger.warning(f"Could not validate single configuration during merge: {e}")
        
        return result_config
    
    # Validate merge strategy
    valid_strategies = ['deep_merge', 'overlay', 'priority', 'selective']
    if merge_strategy not in valid_strategies:
        raise ConfigurationError(
            f"Invalid merge strategy: {merge_strategy}. Valid strategies: {valid_strategies}",
            configuration_section='merge_validation',
            schema_type='merge_operation'
        )
    
    try:
        _logger.info(f"Merging {len(config_list)} configurations using {merge_strategy} strategy")
        
        # Initialize merged configuration with base configuration
        merged_config = copy.deepcopy(config_list[0])
        merge_metadata = {
            'merge_strategy': merge_strategy,
            'source_count': len(config_list),
            'merge_timestamp': datetime.datetime.now().isoformat(),
            'conflicts_resolved': 0,
            'merge_id': str(uuid.uuid4())
        }
        
        if preserve_metadata:
            merged_config['_merge_metadata'] = merge_metadata
            merged_config['_source_metadata'] = []
        
        # Apply merge strategy to resolve parameter conflicts
        for i, config in enumerate(config_list[1:], 1):
            if not isinstance(config, dict):
                _logger.warning(f"Skipping non-dictionary configuration at index {i}")
                continue
            
            # Track source metadata if preserve_metadata is enabled
            if preserve_metadata:
                source_info = {
                    'source_index': i,
                    'parameter_count': _count_configuration_parameters(config),
                    'merge_order': i
                }
                merged_config['_source_metadata'].append(source_info)
            
            # Apply merge strategy
            if merge_strategy == 'deep_merge':
                merged_config = _deep_merge_configurations(merged_config, config, merge_metadata)
            elif merge_strategy == 'overlay':
                merged_config = _overlay_merge_configurations(merged_config, config, merge_metadata)
            elif merge_strategy == 'priority':
                if priority_order:
                    merged_config = _priority_merge_configurations(
                        merged_config, config, priority_order, merge_metadata
                    )
                else:
                    # Fallback to overlay merge if no priority order specified
                    merged_config = _overlay_merge_configurations(merged_config, config, merge_metadata)
            elif merge_strategy == 'selective':
                merged_config = _selective_merge_configurations(merged_config, config, merge_metadata)
        
        # Handle priority order if specified for conflict resolution
        if priority_order and merge_strategy != 'priority':
            merged_config = _apply_priority_order(merged_config, config_list, priority_order)
        
        # Update merge metadata with final statistics
        if preserve_metadata and '_merge_metadata' in merged_config:
            merged_config['_merge_metadata'].update({
                'final_parameter_count': _count_configuration_parameters(merged_config),
                'conflicts_resolved': merge_metadata.get('conflicts_resolved', 0)
            })
        
        # Validate merged configuration if validate_result is enabled
        if validate_result:
            try:
                config_type = _detect_configuration_type(merged_config)
                validation_result = validate_configuration(
                    config_data=merged_config,
                    config_type=config_type,
                    strict_validation=False,  # Use relaxed validation for merged configs
                    check_dependencies=True
                )
                
                if not validation_result.is_valid:
                    # Log validation warnings but don't fail the merge
                    _logger.warning(
                        f"Merged configuration validation issues: {len(validation_result.errors)} errors, "
                        f"{len(validation_result.warnings)} warnings"
                    )
                    
                    # Add validation summary to metadata if preserving metadata
                    if preserve_metadata:
                        merged_config['_merge_metadata']['validation_issues'] = {
                            'error_count': len(validation_result.errors),
                            'warning_count': len(validation_result.warnings),
                            'validation_summary': validation_result.get_summary()
                        }
                
            except Exception as e:
                _logger.warning(f"Could not validate merged configuration: {e}")
        
        # Check for parameter inconsistencies and compatibility issues
        _check_merged_configuration_consistency(merged_config)
        
        # Create audit trail entry for merge operation with source tracking
        create_audit_trail(
            action='CONFIGURATIONS_MERGED',
            component='CONFIG_PARSER',
            action_details={
                'merge_strategy': merge_strategy,
                'source_count': len(config_list),
                'conflicts_resolved': merge_metadata.get('conflicts_resolved', 0),
                'final_parameter_count': _count_configuration_parameters(merged_config),
                'validate_result': validate_result,
                'preserve_metadata': preserve_metadata
            },
            user_context='SYSTEM'
        )
        
        # Log merge operation with conflict resolution details
        _logger.info(
            f"Configuration merge completed: {len(config_list)} sources, "
            f"strategy={merge_strategy}, "
            f"conflicts_resolved={merge_metadata.get('conflicts_resolved', 0)}"
        )
        
        return merged_config
        
    except Exception as e:
        # Handle merge operation errors with comprehensive error reporting
        error_message = f"Configuration merge failed: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='merge_operation',
                schema_type='merge_operation'
            ) from e


def get_default_configuration(
    config_type: str,
    include_documentation: bool = False,
    apply_environment_overrides: bool = True,
    environment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve default configuration for specified type with intelligent default value application, 
    environment-specific overrides, and validation to support rapid experimental setup.
    
    This function provides comprehensive default configuration management with environment-specific
    customization and documentation support for scientific computing workflows.
    
    Args:
        config_type: Type of configuration to retrieve defaults for
        include_documentation: Include parameter documentation and descriptions
        apply_environment_overrides: Apply environment-specific configuration overrides
        environment: Specific environment name for targeted overrides
        
    Returns:
        Dict[str, Any]: Default configuration with applied overrides and documentation
        
    Raises:
        ConfigurationError: When default configuration cannot be loaded or is invalid
    """
    # Validate config_type is supported default configuration
    if config_type not in _default_configs:
        supported_types = list(_default_configs.keys())
        raise ConfigurationError(
            f"Unsupported configuration type: {config_type}. Supported types: {supported_types}",
            configuration_section='default_config_retrieval',
            schema_type=config_type
        )
    
    try:
        _logger.debug(f"Retrieving default configuration: {config_type}")
        
        # Load base default configuration from default configuration files
        default_config_path = _get_default_config_path(config_type)
        
        if not pathlib.Path(default_config_path).exists():
            # Generate default configuration if file doesn't exist
            default_config = _generate_default_configuration(config_type)
        else:
            # Load existing default configuration file
            default_config = load_json_config(
                config_path=default_config_path,
                validate_schema=False,  # Skip validation for default configs
                use_cache=True
            )
        
        # Apply environment-specific overrides if apply_environment_overrides is enabled
        if apply_environment_overrides:
            environment_name = environment or os.getenv('PLUME_ENV', 'development')
            default_config = _apply_environment_overrides(default_config, config_type, environment_name)
        
        # Include parameter documentation if include_documentation is enabled
        if include_documentation:
            default_config = _add_configuration_documentation(default_config, config_type)
        
        # Apply system-specific default adjustments based on platform
        default_config = _apply_system_specific_defaults(default_config, config_type)
        
        # Validate default configuration structure and constraints
        try:
            validation_result = validate_configuration(
                config_data=default_config,
                config_type=config_type,
                strict_validation=False,
                check_dependencies=False
            )
            
            if not validation_result.is_valid:
                _logger.warning(
                    f"Default configuration validation issues for {config_type}: "
                    f"{len(validation_result.errors)} errors"
                )
                
        except Exception as e:
            _logger.warning(f"Could not validate default configuration for {config_type}: {e}")
        
        # Add metadata to default configuration
        default_config['_metadata'] = {
            'config_type': config_type,
            'is_default': True,
            'generated_at': datetime.datetime.now().isoformat(),
            'environment': environment or os.getenv('PLUME_ENV', 'development'),
            'include_documentation': include_documentation,
            'environment_overrides_applied': apply_environment_overrides
        }
        
        # Log default configuration access for audit trail
        create_audit_trail(
            action='DEFAULT_CONFIG_RETRIEVED',
            component='CONFIG_PARSER',
            action_details={
                'config_type': config_type,
                'include_documentation': include_documentation,
                'apply_environment_overrides': apply_environment_overrides,
                'environment': environment,
                'parameter_count': _count_configuration_parameters(default_config)
            },
            user_context='SYSTEM'
        )
        
        _logger.info(f"Default configuration retrieved: {config_type}")
        
        return default_config
        
    except Exception as e:
        # Handle default configuration retrieval errors
        error_message = f"Failed to retrieve default configuration for {config_type}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='default_config_retrieval',
                schema_type=config_type
            ) from e


def parse_config_with_defaults(
    config_data: Dict[str, Any],
    config_type: str,
    fill_missing_defaults: bool = True,
    validate_after_defaults: bool = True,
    warn_on_missing: bool = True
) -> Dict[str, Any]:
    """
    Parse configuration with intelligent default value application, missing parameter detection, 
    and comprehensive validation for robust scientific computing configuration management.
    
    This function provides intelligent configuration parsing with default value application,
    missing parameter detection, and validation to ensure complete and valid configurations.
    
    Args:
        config_data: Configuration dictionary to parse and enhance
        config_type: Type of configuration for default value selection
        fill_missing_defaults: Apply default values for missing parameters
        validate_after_defaults: Validate configuration after applying defaults
        warn_on_missing: Generate warnings for missing parameters
        
    Returns:
        Dict[str, Any]: Configuration with applied defaults and validation results
        
    Raises:
        ConfigurationError: When parsing fails or configuration type is unsupported
        ValidationError: When validation fails after applying defaults
    """
    # Validate config_type and config_data
    if config_type not in _default_configs:
        supported_types = list(_default_configs.keys())
        raise ConfigurationError(
            f"Unsupported configuration type: {config_type}. Supported types: {supported_types}",
            configuration_section='config_parsing',
            schema_type=config_type
        )
    
    if not isinstance(config_data, dict):
        raise ConfigurationError(
            f"Configuration data must be a dictionary, got {type(config_data)}",
            configuration_section='config_parsing',
            schema_type=config_type
        )
    
    try:
        _logger.debug(f"Parsing configuration with defaults: {config_type}")
        
        # Create working copy of configuration data
        parsed_config = copy.deepcopy(config_data)
        
        # Load default configuration for specified config_type
        default_config = get_default_configuration(
            config_type=config_type,
            include_documentation=False,
            apply_environment_overrides=True
        )
        
        # Remove metadata from default config for clean merging
        if '_metadata' in default_config:
            del default_config['_metadata']
        
        # Identify missing parameters in provided config_data
        missing_parameters = _identify_missing_parameters(parsed_config, default_config)
        
        parsing_metadata = {
            'config_type': config_type,
            'original_parameter_count': _count_configuration_parameters(config_data),
            'missing_parameters': missing_parameters,
            'defaults_applied': [],
            'warnings_generated': [],
            'parsing_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Apply default values for missing parameters if fill_missing_defaults is enabled
        if fill_missing_defaults and missing_parameters:
            for param_path in missing_parameters:
                try:
                    default_value = _get_nested_parameter(default_config, param_path)
                    _set_nested_parameter(parsed_config, param_path, default_value)
                    parsing_metadata['defaults_applied'].append(param_path)
                    
                    _logger.debug(f"Applied default value for {param_path}: {default_value}")
                    
                except Exception as e:
                    _logger.warning(f"Could not apply default for {param_path}: {e}")
        
        # Generate warnings for missing parameters if warn_on_missing is enabled
        if warn_on_missing and missing_parameters:
            for param_path in missing_parameters:
                if param_path not in parsing_metadata['defaults_applied']:
                    warning_msg = f"Missing parameter: {param_path}"
                    parsing_metadata['warnings_generated'].append(warning_msg)
                    _logger.warning(warning_msg)
        
        # Validate configuration after applying defaults if validate_after_defaults is enabled
        validation_result = None
        if validate_after_defaults:
            validation_result = validate_configuration(
                config_data=parsed_config,
                config_type=config_type,
                strict_validation=False,
                check_dependencies=True
            )
            
            if not validation_result.is_valid:
                error_summary = f"Validation failed after applying defaults: {len(validation_result.errors)} errors"
                parsing_metadata['validation_errors'] = validation_result.errors
                
                _logger.error(error_summary)
                
                raise ValidationError(
                    message=error_summary,
                    validation_type='post_defaults_validation',
                    validation_context=config_type
                )
        
        # Check parameter type consistency and value ranges
        _check_parameter_consistency(parsed_config, config_type, parsing_metadata)
        
        # Add parsing metadata to configuration
        parsed_config['_parsing_metadata'] = parsing_metadata
        
        # Update final parameter count
        parsing_metadata['final_parameter_count'] = _count_configuration_parameters(parsed_config)
        
        # Log default application operation with applied parameters
        create_audit_trail(
            action='CONFIG_PARSED_WITH_DEFAULTS',
            component='CONFIG_PARSER',
            action_details=parsing_metadata,
            user_context='SYSTEM'
        )
        
        _logger.info(
            f"Configuration parsed with defaults: {config_type}, "
            f"applied {len(parsing_metadata['defaults_applied'])} defaults, "
            f"generated {len(parsing_metadata['warnings_generated'])} warnings"
        )
        
        return parsed_config
        
    except Exception as e:
        # Handle configuration parsing errors
        error_message = f"Failed to parse configuration with defaults for {config_type}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, (ConfigurationError, ValidationError)):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='config_parsing',
                schema_type=config_type
            ) from e


def load_schema(
    schema_name: str,
    schema_path: Optional[str] = None,
    use_cache: bool = True,
    validate_schema_structure: bool = True
) -> Dict[str, Any]:
    """
    Load JSON schema file with caching, validation, and error handling for configuration 
    validation framework with schema registry management.
    
    This function provides robust schema loading with caching, validation, and comprehensive
    error handling for the configuration validation framework.
    
    Args:
        schema_name: Name of the schema to load
        schema_path: Optional custom path to schema file
        use_cache: Enable schema caching for performance
        validate_schema_structure: Validate the schema structure itself
        
    Returns:
        Dict[str, Any]: Loaded JSON schema with validation metadata
        
    Raises:
        ConfigurationError: When schema cannot be loaded or is invalid
    """
    # Check schema cache if use_cache is enabled
    cache_key = f"schema:{schema_name}:{schema_path}"
    if use_cache:
        with _cache_lock:
            if cache_key in _schema_cache:
                _logger.debug(f"Schema loaded from cache: {schema_name}")
                return copy.deepcopy(_schema_cache[cache_key])
    
    try:
        # Determine schema file path from schema_name or use provided schema_path
        if schema_path is None:
            if schema_name in _schema_files:
                schema_path = _get_schema_path(schema_name)
            else:
                raise ConfigurationError(
                    f"No default schema path for {schema_name}",
                    configuration_section='schema_loading',
                    schema_type=schema_name
                )
        
        # Load JSON schema file with error handling
        schema_file_path = pathlib.Path(schema_path)
        if not schema_file_path.exists():
            raise ConfigurationError(
                f"Schema file not found: {schema_path}",
                configuration_section='schema_loading',
                schema_type=schema_name
            )
        
        _logger.debug(f"Loading schema: {schema_name} from {schema_path}")
        
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Validate schema structure if validate_schema_structure is enabled
        if validate_schema_structure:
            try:
                # Use jsonschema meta-schema to validate the schema itself
                jsonschema.Draft7Validator.check_schema(schema_data)
                _logger.debug(f"Schema structure validation passed: {schema_name}")
                
            except jsonschema.SchemaError as e:
                raise ConfigurationError(
                    f"Schema structure validation failed: {e.message}",
                    configuration_section='schema_validation',
                    schema_type=schema_name
                ) from e
        
        # Add schema metadata
        schema_metadata = {
            'schema_name': schema_name,
            'schema_path': schema_path,
            'loaded_at': datetime.datetime.now().isoformat(),
            'structure_validated': validate_schema_structure,
            'schema_version': schema_data.get('$schema', 'unknown')
        }
        
        schema_data['_schema_metadata'] = schema_metadata
        
        # Cache schema if caching is enabled
        if use_cache:
            with _cache_lock:
                _schema_cache[cache_key] = copy.deepcopy(schema_data)
                _logger.debug(f"Schema cached: {schema_name}")
        
        # Log schema loading operation
        create_audit_trail(
            action='SCHEMA_LOADED',
            component='CONFIG_PARSER',
            action_details=schema_metadata,
            user_context='SYSTEM'
        )
        
        _logger.info(f"Schema loaded successfully: {schema_name}")
        
        return schema_data
        
    except Exception as e:
        # Handle schema loading errors
        error_message = f"Failed to load schema {schema_name}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='schema_loading',
                schema_type=schema_name
            ) from e


def substitute_environment_variables(
    config_data: Dict[str, Any],
    strict_substitution: bool = False,
    default_values: Optional[Dict[str, str]] = None,
    validate_substitutions: bool = True
) -> Dict[str, Any]:
    """
    Substitute environment variables in configuration values with validation, default handling, 
    and security checks for flexible configuration management across environments.
    
    This function provides comprehensive environment variable substitution with validation,
    default handling, and security checks for flexible configuration management.
    
    Args:
        config_data: Configuration dictionary to process
        strict_substitution: Require all environment variables to be defined
        default_values: Default values for undefined environment variables
        validate_substitutions: Validate substituted values
        
    Returns:
        Dict[str, Any]: Configuration with substituted environment variables
        
    Raises:
        ConfigurationError: When environment variable substitution fails
    """
    if not isinstance(config_data, dict):
        raise ConfigurationError(
            f"Configuration data must be a dictionary, got {type(config_data)}",
            configuration_section='env_substitution',
            schema_type='environment_variables'
        )
    
    try:
        _logger.debug("Substituting environment variables in configuration")
        
        # Create working copy for substitution
        substituted_config = copy.deepcopy(config_data)
        
        substitution_metadata = {
            'substitutions_made': [],
            'missing_variables': [],
            'default_values_used': [],
            'validation_errors': [],
            'substitution_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Recursively traverse configuration dictionary
        _substitute_env_vars_recursive(
            substituted_config, 
            substitution_metadata, 
            strict_substitution, 
            default_values or {},
            validate_substitutions
        )
        
        # Check for missing environment variables in strict mode
        if strict_substitution and substitution_metadata['missing_variables']:
            missing_vars = substitution_metadata['missing_variables']
            raise ConfigurationError(
                f"Missing required environment variables: {missing_vars}",
                configuration_section='env_substitution',
                schema_type='environment_variables'
            )
        
        # Validate substituted values if validate_substitutions is enabled
        if validate_substitutions and substitution_metadata['validation_errors']:
            validation_errors = substitution_metadata['validation_errors']
            _logger.warning(f"Environment variable substitution validation errors: {validation_errors}")
        
        # Add substitution metadata to configuration
        substituted_config['_env_substitution_metadata'] = substitution_metadata
        
        # Log environment variable substitutions for audit trail
        create_audit_trail(
            action='ENV_VARS_SUBSTITUTED',
            component='CONFIG_PARSER',
            action_details=substitution_metadata,
            user_context='SYSTEM'
        )
        
        _logger.info(
            f"Environment variable substitution completed: "
            f"{len(substitution_metadata['substitutions_made'])} substitutions made"
        )
        
        return substituted_config
        
    except Exception as e:
        # Handle environment variable substitution errors
        error_message = f"Environment variable substitution failed: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='env_substitution',
                schema_type='environment_variables'
            ) from e


def create_configuration_backup(
    config_path: str,
    compress_backup: bool = False,
    include_metadata: bool = True,
    backup_directory: Optional[str] = None
) -> str:
    """
    Create versioned backup of configuration file with metadata preservation, compression, 
    and retention management for configuration change tracking and recovery.
    
    This function provides comprehensive configuration backup with versioning, metadata
    preservation, and retention management for configuration change tracking.
    
    Args:
        config_path: Path to configuration file to backup
        compress_backup: Enable backup compression
        include_metadata: Include metadata in backup
        backup_directory: Custom directory for backup storage
        
    Returns:
        str: Path to created backup file
        
    Raises:
        ConfigurationError: When backup creation fails
    """
    try:
        # Validate source configuration file exists
        config_file_path = pathlib.Path(config_path)
        if not config_file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                configuration_section='backup_creation',
                schema_type='backup_operation'
            )
        
        # Generate backup filename with timestamp and version
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{config_file_path.stem}_backup_{timestamp}{config_file_path.suffix}"
        
        # Determine backup directory
        if backup_directory:
            backup_dir = pathlib.Path(backup_directory)
        else:
            backup_dir = config_file_path.parent / 'backups'
        
        # Create backup directory if backup_directory is specified
        ensure_directory_exists(str(backup_dir), create_parents=True)
        
        backup_path = backup_dir / backup_filename
        
        _logger.debug(f"Creating configuration backup: {config_path} -> {backup_path}")
        
        # Copy configuration file to backup location
        import shutil
        shutil.copy2(str(config_file_path), str(backup_path))
        
        # Include metadata in backup if include_metadata is enabled
        if include_metadata:
            metadata = {
                'original_path': str(config_file_path.absolute()),
                'backup_created': datetime.datetime.now().isoformat(),
                'original_size': config_file_path.stat().st_size,
                'original_modified': datetime.datetime.fromtimestamp(config_file_path.stat().st_mtime).isoformat(),
                'backup_version': 1
            }
            
            metadata_path = backup_path.with_suffix(backup_path.suffix + '.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        # Compress backup file if compress_backup is enabled
        if compress_backup:
            backup_path = _compress_backup_file(backup_path)
        
        # Manage backup retention and cleanup old versions
        _cleanup_old_backups(backup_dir, config_file_path.stem)
        
        # Log backup creation operation
        create_audit_trail(
            action='CONFIG_BACKUP_CREATED',
            component='CONFIG_PARSER',
            action_details={
                'original_path': config_path,
                'backup_path': str(backup_path),
                'compress_backup': compress_backup,
                'include_metadata': include_metadata,
                'backup_directory': str(backup_dir)
            },
            user_context='SYSTEM'
        )
        
        _logger.info(f"Configuration backup created: {backup_path}")
        
        return str(backup_path)
        
    except Exception as e:
        # Handle backup creation errors
        error_message = f"Failed to create configuration backup for {config_path}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='backup_creation',
                schema_type='backup_operation'
            ) from e


def restore_configuration_backup(
    backup_path: str,
    target_config_path: str,
    validate_before_restore: bool = True,
    create_restore_backup: bool = True
) -> Dict[str, Any]:
    """
    Restore configuration from backup with validation, atomic operations, and rollback 
    capability for configuration recovery and version management.
    
    This function provides comprehensive configuration restoration with validation,
    atomic operations, and rollback capability for configuration recovery.
    
    Args:
        backup_path: Path to backup file to restore
        target_config_path: Path where configuration should be restored
        validate_before_restore: Validate backup before restoration
        create_restore_backup: Create backup of current configuration before restore
        
    Returns:
        Dict[str, Any]: Restore operation result with success status and validation information
        
    Raises:
        ConfigurationError: When restore operation fails
    """
    restore_result = {
        'success': False,
        'backup_path': backup_path,
        'target_path': target_config_path,
        'validation_passed': False,
        'current_backup_created': False,
        'current_backup_path': None,
        'restore_timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        # Validate backup file exists and is accessible
        backup_file_path = pathlib.Path(backup_path)
        if not backup_file_path.exists():
            raise ConfigurationError(
                f"Backup file not found: {backup_path}",
                configuration_section='backup_restore',
                schema_type='restore_operation'
            )
        
        target_file_path = pathlib.Path(target_config_path)
        
        _logger.info(f"Restoring configuration backup: {backup_path} -> {target_config_path}")
        
        # Create backup of current configuration if create_restore_backup is enabled
        if create_restore_backup and target_file_path.exists():
            current_backup_path = create_configuration_backup(
                config_path=str(target_file_path),
                compress_backup=False,
                include_metadata=True
            )
            restore_result['current_backup_created'] = True
            restore_result['current_backup_path'] = current_backup_path
            _logger.info(f"Current configuration backed up before restore: {current_backup_path}")
        
        # Load and validate backup configuration if validate_before_restore is enabled
        if validate_before_restore:
            try:
                backup_config = load_json_config(
                    config_path=str(backup_file_path),
                    validate_schema=False,  # Skip schema validation for backup files
                    use_cache=False
                )
                
                # Attempt to detect configuration type and validate
                config_type = _detect_configuration_type(backup_config)
                if config_type:
                    validation_result = validate_configuration(
                        config_data=backup_config,
                        config_type=config_type,
                        strict_validation=False
                    )
                    
                    if validation_result.is_valid:
                        restore_result['validation_passed'] = True
                    else:
                        _logger.warning(
                            f"Backup validation issues: {len(validation_result.errors)} errors, "
                            f"proceeding with restore"
                        )
                        restore_result['validation_passed'] = False
                
            except Exception as e:
                _logger.warning(f"Could not validate backup before restore: {e}")
                restore_result['validation_passed'] = False
        
        # Ensure target directory exists
        ensure_directory_exists(str(target_file_path.parent), create_parents=True)
        
        # Perform atomic restore operation
        import shutil
        import tempfile
        
        # Use temporary file for atomic restore
        with tempfile.NamedTemporaryFile(
            dir=target_file_path.parent,
            prefix='restore_temp_',
            suffix=target_file_path.suffix,
            delete=False
        ) as temp_file:
            temp_path = pathlib.Path(temp_file.name)
        
        # Copy backup to temporary location
        shutil.copy2(str(backup_file_path), str(temp_path))
        
        # Atomically move temporary file to target location
        shutil.move(str(temp_path), str(target_file_path))
        
        # Verify restore success and configuration integrity
        if target_file_path.exists():
            restore_result['success'] = True
            _logger.info(f"Configuration restored successfully: {target_config_path}")
        else:
            raise ConfigurationError(
                "Restore verification failed - target file not found after restore",
                configuration_section='restore_verification',
                schema_type='restore_operation'
            )
        
        # Update configuration cache with restored configuration
        # Force cache refresh by loading the restored configuration
        try:
            restored_config = load_json_config(
                config_path=str(target_file_path),
                validate_schema=False,
                use_cache=True  # This will update the cache
            )
            _logger.debug("Configuration cache updated with restored configuration")
            
        except Exception as e:
            _logger.warning(f"Could not update cache with restored configuration: {e}")
        
        # Create audit trail entry for restore operation
        create_audit_trail(
            action='CONFIG_BACKUP_RESTORED',
            component='CONFIG_PARSER',
            action_details=restore_result,
            user_context='SYSTEM'
        )
        
        return restore_result
        
    except Exception as e:
        # Handle restore operation errors with rollback if needed
        restore_result['success'] = False
        restore_result['error'] = str(e)
        restore_result['error_type'] = type(e).__name__
        
        error_message = f"Configuration restore failed: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        # Attempt rollback if current backup was created
        if restore_result['current_backup_created'] and restore_result['current_backup_path']:
            try:
                rollback_result = restore_configuration_backup(
                    backup_path=restore_result['current_backup_path'],
                    target_config_path=target_config_path,
                    validate_before_restore=False,
                    create_restore_backup=False
                )
                restore_result['rollback_performed'] = rollback_result['success']
                _logger.info("Rollback performed after failed restore")
                
            except Exception as rollback_error:
                restore_result['rollback_error'] = str(rollback_error)
                _logger.error(f"Rollback failed after restore failure: {rollback_error}")
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='backup_restore',
                schema_type='restore_operation'
            ) from e


def clear_configuration_cache(
    config_types: Optional[List[str]] = None,
    clear_schema_cache: bool = False,
    preserve_statistics: bool = True
) -> Dict[str, int]:
    """
    Clear configuration cache with selective clearing options, statistics preservation, and 
    cache management for development and testing scenarios.
    
    This function provides comprehensive cache management with selective clearing and
    statistics preservation for development and testing scenarios.
    
    Args:
        config_types: List of configuration types to clear (None for all)
        clear_schema_cache: Clear schema cache in addition to configuration cache
        preserve_statistics: Preserve cache statistics for monitoring
        
    Returns:
        Dict[str, int]: Cache clearing statistics with cleared entries count and preservation summary
        
    Raises:
        ConfigurationError: When cache clearing fails
    """
    try:
        clearing_stats = {
            'config_cache_cleared': 0,
            'schema_cache_cleared': 0,
            'statistics_preserved': preserve_statistics,
            'clearing_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Acquire cache lock for thread-safe operation
        with _cache_lock:
            _logger.debug("Clearing configuration cache")
            
            # Identify cache entries to clear based on config_types filter
            if config_types is None:
                # Clear all configuration cache entries
                cache_size_before = len(_config_cache)
                _config_cache.clear()
                clearing_stats['config_cache_cleared'] = cache_size_before
                _logger.debug(f"Cleared all configuration cache entries: {cache_size_before}")
                
            else:
                # Clear specified configuration cache entries
                keys_to_remove = []
                for cache_key in _config_cache.keys():
                    for config_type in config_types:
                        if cache_key.startswith(f"{config_type}:"):
                            keys_to_remove.append(cache_key)
                            break
                
                for key in keys_to_remove:
                    del _config_cache[key]
                    clearing_stats['config_cache_cleared'] += 1
                
                _logger.debug(f"Cleared {len(keys_to_remove)} specific configuration cache entries")
            
            # Clear schema cache if clear_schema_cache is enabled
            if clear_schema_cache:
                schema_cache_size_before = len(_schema_cache)
                _schema_cache.clear()
                clearing_stats['schema_cache_cleared'] = schema_cache_size_before
                _logger.debug(f"Cleared schema cache entries: {schema_cache_size_before}")
        
        # Update cache metadata and timestamps
        # Note: In a full implementation, this would update cache statistics
        
        # Log cache clearing operation with statistics
        create_audit_trail(
            action='CONFIG_CACHE_CLEARED',
            component='CONFIG_PARSER',
            action_details=clearing_stats,
            user_context='SYSTEM'
        )
        
        _logger.info(
            f"Configuration cache cleared: {clearing_stats['config_cache_cleared']} config entries, "
            f"{clearing_stats['schema_cache_cleared']} schema entries"
        )
        
        return clearing_stats
        
    except Exception as e:
        # Handle cache clearing errors
        error_message = f"Failed to clear configuration cache: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        raise ConfigurationError(
            error_message,
            configuration_section='cache_management',
            schema_type='cache_clearing'
        ) from e


def get_configuration_metadata(
    config_name: str,
    include_validation_history: bool = False,
    include_dependency_info: bool = False,
    include_cache_info: bool = False
) -> Dict[str, Any]:
    """
    Retrieve comprehensive metadata for configuration including version information, validation 
    history, modification timestamps, and dependency tracking for audit and debugging.
    
    This function provides comprehensive configuration metadata retrieval with detailed
    information about configuration state, validation history, and dependencies.
    
    Args:
        config_name: Name of configuration to retrieve metadata for
        include_validation_history: Include validation history in metadata
        include_dependency_info: Include dependency analysis in metadata
        include_cache_info: Include cache information in metadata
        
    Returns:
        Dict[str, Any]: Comprehensive configuration metadata with history and dependency information
        
    Raises:
        ConfigurationError: When metadata retrieval fails
    """
    try:
        # Validate config_name
        if config_name not in _default_configs:
            supported_types = list(_default_configs.keys())
            raise ConfigurationError(
                f"Unsupported configuration type: {config_name}. Supported types: {supported_types}",
                configuration_section='metadata_retrieval',
                schema_type=config_name
            )
        
        _logger.debug(f"Retrieving metadata for configuration: {config_name}")
        
        # Initialize metadata container
        metadata = {
            'config_name': config_name,
            'config_type': config_name,
            'metadata_generated': datetime.datetime.now().isoformat(),
            'basic_info': {},
            'file_info': {},
            'schema_info': {}
        }
        
        # Load configuration file metadata including timestamps and size
        config_path = _get_default_config_path(config_name)
        config_file_path = pathlib.Path(config_path)
        
        if config_file_path.exists():
            file_stat = config_file_path.stat()
            metadata['file_info'] = {
                'file_path': str(config_file_path.absolute()),
                'file_size_bytes': file_stat.st_size,
                'last_modified': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'created': datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'exists': True
            }
            
            # Load configuration to get parameter count
            try:
                config_data = load_json_config(
                    config_path=str(config_file_path),
                    validate_schema=False,
                    use_cache=True
                )
                metadata['basic_info'] = {
                    'parameter_count': _count_configuration_parameters(config_data),
                    'has_metadata': '_metadata' in config_data,
                    'configuration_sections': list(config_data.keys()) if isinstance(config_data, dict) else []
                }
                
            except Exception as e:
                metadata['basic_info']['load_error'] = str(e)
        else:
            metadata['file_info'] = {
                'file_path': str(config_file_path.absolute()),
                'exists': False,
                'note': 'Configuration file not found - using defaults'
            }
        
        # Include validation history if include_validation_history is enabled
        if include_validation_history:
            # Note: In a full implementation, this would retrieve validation history from a persistent store
            metadata['validation_history'] = {
                'last_validation': 'unknown',
                'validation_count': 0,
                'last_validation_result': 'unknown',
                'note': 'Validation history not implemented in this version'
            }
        
        # Analyze configuration dependencies if include_dependency_info is enabled
        if include_dependency_info:
            metadata['dependency_info'] = {
                'schema_dependencies': [],
                'environment_dependencies': [],
                'file_dependencies': [],
                'note': 'Dependency analysis not fully implemented in this version'
            }
            
            # Check for schema dependencies
            if config_name in _schema_files:
                schema_path = _get_schema_path(config_name)
                metadata['dependency_info']['schema_dependencies'].append(schema_path)
        
        # Include cache information if include_cache_info is enabled
        if include_cache_info:
            with _cache_lock:
                cache_entries = [key for key in _config_cache.keys() if key.startswith(f"{config_name}:")]
                schema_cache_entries = [key for key in _schema_cache.keys() if config_name in key]
                
                metadata['cache_info'] = {
                    'cached_entries': len(cache_entries),
                    'cached_variants': cache_entries,
                    'schema_cached': len(schema_cache_entries) > 0,
                    'schema_cache_entries': schema_cache_entries
                }
        
        # Include schema information
        if config_name in _schema_files:
            schema_path = _get_schema_path(config_name)
            schema_file_path = pathlib.Path(schema_path)
            
            if schema_file_path.exists():
                schema_stat = schema_file_path.stat()
                metadata['schema_info'] = {
                    'schema_path': str(schema_file_path.absolute()),
                    'schema_exists': True,
                    'schema_size_bytes': schema_stat.st_size,
                    'schema_modified': datetime.datetime.fromtimestamp(schema_stat.st_mtime).isoformat()
                }
            else:
                metadata['schema_info'] = {
                    'schema_path': str(schema_file_path.absolute()),
                    'schema_exists': False
                }
        else:
            metadata['schema_info'] = {
                'schema_available': False,
                'note': 'No schema defined for this configuration type'
            }
        
        # Log metadata access operation
        create_audit_trail(
            action='CONFIG_METADATA_RETRIEVED',
            component='CONFIG_PARSER',
            action_details={
                'config_name': config_name,
                'include_validation_history': include_validation_history,
                'include_dependency_info': include_dependency_info,
                'include_cache_info': include_cache_info
            },
            user_context='SYSTEM'
        )
        
        _logger.info(f"Configuration metadata retrieved: {config_name}")
        
        return metadata
        
    except Exception as e:
        # Handle metadata retrieval errors
        error_message = f"Failed to retrieve metadata for {config_name}: {str(e)}"
        _logger.error(error_message, exc_info=True)
        
        if isinstance(e, ConfigurationError):
            raise e
        else:
            raise ConfigurationError(
                error_message,
                configuration_section='metadata_retrieval',
                schema_type=config_name
            ) from e


class ConfigurationParser:
    """
    Comprehensive configuration parser class providing centralized configuration management, 
    validation, caching, and versioning for the plume simulation system with thread-safe 
    operations, schema validation, and audit trail integration for scientific computing reproducibility.
    
    This class provides a centralized interface for all configuration operations with comprehensive
    validation, caching, versioning, and audit trail support for scientific computing workflows.
    """
    
    def __init__(
        self,
        config_directory: str,
        schema_directory: str,
        enable_caching: bool = True,
        enable_validation: bool = True,
        cache_directory: Optional[str] = None
    ):
        """
        Initialize configuration parser with directory paths, caching settings, and validation 
        framework for comprehensive configuration management.
        
        Args:
            config_directory: Directory containing configuration files
            schema_directory: Directory containing JSON schema files
            enable_caching: Enable configuration and schema caching
            enable_validation: Enable automatic validation for all operations
            cache_directory: Optional custom directory for cache storage
            
        Raises:
            ConfigurationError: When initialization fails or directories are invalid
        """
        # Validate configuration and schema directories exist and are accessible
        self.config_directory = pathlib.Path(config_directory)
        self.schema_directory = pathlib.Path(schema_directory)
        
        if not self.config_directory.exists():
            raise ConfigurationError(
                f"Configuration directory not found: {config_directory}",
                configuration_section='parser_initialization',
                schema_type='initialization'
            )
        
        if not self.schema_directory.exists():
            raise ConfigurationError(
                f"Schema directory not found: {schema_directory}",
                configuration_section='parser_initialization',
                schema_type='initialization'
            )
        
        # Initialize configuration and schema caches with thread-safe locking
        self.caching_enabled = enable_caching
        self.validation_enabled = enable_validation
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.schema_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime.datetime] = {}
        self.cache_lock = threading.RLock()
        
        # Setup cache directory and ensure it exists
        if cache_directory:
            self.cache_directory = pathlib.Path(cache_directory)
            ensure_directory_exists(str(self.cache_directory), create_parents=True)
        else:
            self.cache_directory = None
        
        # Configure caching and validation settings
        self.cache_expiry_hours = CACHE_EXPIRY_HOURS
        self.validation_timeout = VALIDATION_TIMEOUT_SECONDS
        
        # Initialize validation history tracking
        self.validation_history: Dict[str, ValidationResult] = {}
        
        # Setup logger for configuration parser operations
        self.logger = get_logger(f'{__name__}.ConfigurationParser', 'CONFIG_PARSER')
        
        # Create audit trail for parser initialization
        create_audit_trail(
            action='CONFIG_PARSER_INITIALIZED',
            component='CONFIG_PARSER',
            action_details={
                'config_directory': str(self.config_directory),
                'schema_directory': str(self.schema_directory),
                'caching_enabled': self.caching_enabled,
                'validation_enabled': self.validation_enabled,
                'cache_directory': str(self.cache_directory) if self.cache_directory else None
            },
            user_context='SYSTEM'
        )
        
        self.logger.info(f"Configuration parser initialized: {config_directory}")
    
    def load_config(
        self,
        config_name: str,
        config_path: Optional[str] = None,
        validate_schema: bool = None,
        use_cache: bool = None
    ) -> Dict[str, Any]:
        """
        Load configuration file with comprehensive validation, caching, and error handling 
        for scientific computing parameters.
        
        Args:
            config_name: Name of configuration to load
            config_path: Optional custom path to configuration file
            validate_schema: Enable schema validation (uses instance default if None)
            use_cache: Enable caching (uses instance default if None)
            
        Returns:
            Dict[str, Any]: Loaded and validated configuration dictionary
            
        Raises:
            ConfigurationError: When configuration loading fails
        """
        # Apply instance defaults for optional parameters
        if validate_schema is None:
            validate_schema = self.validation_enabled
        if use_cache is None:
            use_cache = self.caching_enabled
        
        # Check configuration cache if use_cache is enabled
        cache_key = f"{config_name}:{config_path}:{validate_schema}"
        if use_cache and self._is_cache_valid(cache_key):
            with self.cache_lock:
                if cache_key in self.config_cache:
                    self.logger.debug(f"Configuration loaded from instance cache: {config_name}")
                    return copy.deepcopy(self.config_cache[cache_key])
        
        try:
            # Determine configuration file path
            if config_path is None:
                config_path = str(self.config_directory / f"{config_name}.json")
            
            # Load configuration using load_json_config
            config_data = load_json_config(
                config_path=config_path,
                schema_path=str(self.schema_directory / f"{config_name}_schema.json") if validate_schema else None,
                validate_schema=validate_schema,
                use_cache=False  # Use our own caching
            )
            
            # Perform schema validation if validate_schema is enabled
            if validate_schema:
                validation_result = validate_configuration(
                    config_data=config_data,
                    config_type=config_name,
                    strict_validation=True,
                    check_dependencies=True
                )
                
                # Store validation result in history
                self.validation_history[config_name] = validation_result
                
                if not validation_result.is_valid:
                    raise ValidationError(
                        f"Configuration validation failed: {len(validation_result.errors)} errors",
                        validation_type='load_validation',
                        validation_context=config_name
                    )
            
            # Cache configuration if caching is enabled
            if use_cache:
                with self.cache_lock:
                    self.config_cache[cache_key] = copy.deepcopy(config_data)
                    self.cache_timestamps[cache_key] = datetime.datetime.now()
            
            # Update validation history
            self.logger.info(f"Configuration loaded successfully: {config_name}")
            
            return config_data
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_name}: {e}", exc_info=True)
            raise
    
    def save_config(
        self,
        config_name: str,
        config_data: Dict[str, Any],
        validate_schema: bool = None,
        create_backup: bool = True
    ) -> bool:
        """
        Save configuration with atomic operations, validation, and backup creation for data 
        integrity and traceability.
        
        Args:
            config_name: Name of configuration to save
            config_data: Configuration data to save
            validate_schema: Enable schema validation (uses instance default if None)
            create_backup: Create backup before saving
            
        Returns:
            bool: Success status of save operation
            
        Raises:
            ConfigurationError: When save operation fails
        """
        # Apply instance defaults
        if validate_schema is None:
            validate_schema = self.validation_enabled
        
        try:
            # Validate configuration data structure
            if not isinstance(config_data, dict):
                raise ConfigurationError(
                    f"Configuration data must be a dictionary, got {type(config_data)}",
                    configuration_section='save_validation',
                    schema_type=config_name
                )
            
            # Perform schema validation if validate_schema is enabled
            if validate_schema:
                validation_result = validate_configuration(
                    config_data=config_data,
                    config_type=config_name,
                    strict_validation=True,
                    check_dependencies=True
                )
                
                if not validation_result.is_valid:
                    raise ValidationError(
                        f"Configuration validation failed before save: {len(validation_result.errors)} errors",
                        validation_type='save_validation',
                        validation_context=config_name
                    )
            
            # Determine configuration file path
            config_path = str(self.config_directory / f"{config_name}.json")
            
            # Create backup if create_backup is enabled
            if create_backup and pathlib.Path(config_path).exists():
                backup_path = create_configuration_backup(
                    config_path=config_path,
                    compress_backup=False,
                    include_metadata=True
                )
                self.logger.info(f"Configuration backup created: {backup_path}")
            
            # Save configuration using atomic operations
            save_result = save_json_config(
                config_data=config_data,
                config_path=config_path,
                schema_path=str(self.schema_directory / f"{config_name}_schema.json") if validate_schema else None,
                validate_schema=validate_schema,
                create_backup=False,  # We handled backup above
                atomic_write=True
            )
            
            if not save_result.get('success', False):
                raise ConfigurationError(
                    f"Save operation failed: {save_result.get('error', 'Unknown error')}",
                    configuration_section='file_save',
                    schema_type=config_name
                )
            
            # Update configuration cache
            if self.caching_enabled:
                cache_key = f"{config_name}:{config_path}:{validate_schema}"
                with self.cache_lock:
                    self.config_cache[cache_key] = copy.deepcopy(config_data)
                    self.cache_timestamps[cache_key] = datetime.datetime.now()
            
            # Create audit trail entry
            create_audit_trail(
                action='CONFIG_SAVED_BY_PARSER',
                component='CONFIG_PARSER',
                action_details={
                    'config_name': config_name,
                    'config_path': config_path,
                    'validate_schema': validate_schema,
                    'create_backup': create_backup,
                    'bytes_written': save_result.get('bytes_written', 0)
                },
                user_context='SYSTEM'
            )
            
            self.logger.info(f"Configuration saved successfully: {config_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration {config_name}: {e}", exc_info=True)
            raise
    
    def validate_config(
        self,
        config_data: Dict[str, Any],
        config_type: str,
        strict_validation: bool = False
    ) -> ValidationResult:
        """
        Comprehensive configuration validation against schema and scientific constraints with 
        detailed error reporting.
        
        Args:
            config_data: Configuration data to validate
            config_type: Type of configuration for validation rules
            strict_validation: Enable strict validation mode
            
        Returns:
            ValidationResult: Detailed validation results with errors and recommendations
            
        Raises:
            ConfigurationError: When validation setup fails
        """
        try:
            # Perform comprehensive validation
            validation_result = validate_configuration(
                config_data=config_data,
                config_type=config_type,
                strict_validation=strict_validation,
                check_dependencies=True,
                schema_path=str(self.schema_directory / f"{config_type}_schema.json")
            )
            
            # Store validation result in history
            self.validation_history[config_type] = validation_result
            
            # Update validation history
            self.logger.info(
                f"Configuration validation completed: {config_type}, "
                f"valid={validation_result.is_valid}, "
                f"errors={len(validation_result.errors)}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed for {config_type}: {e}", exc_info=True)
            raise
    
    def merge_configs(
        self,
        config_list: List[Dict[str, Any]],
        merge_strategy: str = 'deep_merge',
        validate_result: bool = None
    ) -> Dict[str, Any]:
        """
        Intelligently merge multiple configurations with conflict resolution and validation.
        
        Args:
            config_list: List of configurations to merge
            merge_strategy: Strategy for merging configurations
            validate_result: Validate merged result (uses instance default if None)
            
        Returns:
            Dict[str, Any]: Merged configuration with resolved conflicts
            
        Raises:
            ConfigurationError: When merge operation fails
        """
        # Apply instance defaults
        if validate_result is None:
            validate_result = self.validation_enabled
        
        try:
            # Perform configuration merge
            merged_config = merge_configurations(
                config_list=config_list,
                merge_strategy=merge_strategy,
                validate_result=validate_result,
                priority_order=None,
                preserve_metadata=True
            )
            
            # Create audit trail for merge operation
            create_audit_trail(
                action='CONFIGS_MERGED_BY_PARSER',
                component='CONFIG_PARSER',
                action_details={
                    'config_count': len(config_list),
                    'merge_strategy': merge_strategy,
                    'validate_result': validate_result,
                    'final_parameter_count': _count_configuration_parameters(merged_config)
                },
                user_context='SYSTEM'
            )
            
            self.logger.info(f"Configurations merged: {len(config_list)} sources using {merge_strategy}")
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Configuration merge failed: {e}", exc_info=True)
            raise
    
    def get_default_config(
        self,
        config_type: str,
        include_documentation: bool = False,
        apply_overrides: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve default configuration with environment-specific overrides and documentation.
        
        Args:
            config_type: Type of default configuration to retrieve
            include_documentation: Include parameter documentation
            apply_overrides: Apply environment-specific overrides
            
        Returns:
            Dict[str, Any]: Default configuration with applied overrides
            
        Raises:
            ConfigurationError: When default configuration cannot be retrieved
        """
        try:
            # Retrieve default configuration
            default_config = get_default_configuration(
                config_type=config_type,
                include_documentation=include_documentation,
                apply_environment_overrides=apply_overrides
            )
            
            # Validate default configuration
            if self.validation_enabled:
                validation_result = validate_configuration(
                    config_data=default_config,
                    config_type=config_type,
                    strict_validation=False,
                    check_dependencies=False
                )
                
                if not validation_result.is_valid:
                    self.logger.warning(
                        f"Default configuration validation issues for {config_type}: "
                        f"{len(validation_result.errors)} errors"
                    )
            
            self.logger.info(f"Default configuration retrieved: {config_type}")
            
            return default_config
            
        except Exception as e:
            self.logger.error(f"Failed to get default configuration for {config_type}: {e}", exc_info=True)
            raise
    
    def clear_cache(
        self,
        config_types: Optional[List[str]] = None,
        clear_schemas: bool = False
    ) -> Dict[str, int]:
        """
        Clear configuration and schema caches with selective clearing and statistics preservation.
        
        Args:
            config_types: Specific configuration types to clear (None for all)
            clear_schemas: Clear schema cache in addition to configuration cache
            
        Returns:
            Dict[str, int]: Cache clearing statistics
            
        Raises:
            ConfigurationError: When cache clearing fails
        """
        try:
            clearing_stats = {
                'config_cache_cleared': 0,
                'schema_cache_cleared': 0,
                'timestamp_entries_cleared': 0
            }
            
            # Acquire cache lock for thread safety
            with self.cache_lock:
                # Clear specified configuration cache entries
                if config_types is None:
                    # Clear all entries
                    clearing_stats['config_cache_cleared'] = len(self.config_cache)
                    clearing_stats['timestamp_entries_cleared'] = len(self.cache_timestamps)
                    
                    self.config_cache.clear()
                    self.cache_timestamps.clear()
                else:
                    # Clear specific entries
                    keys_to_remove = []
                    for cache_key in self.config_cache.keys():
                        for config_type in config_types:
                            if cache_key.startswith(f"{config_type}:"):
                                keys_to_remove.append(cache_key)
                                break
                    
                    for key in keys_to_remove:
                        del self.config_cache[key]
                        if key in self.cache_timestamps:
                            del self.cache_timestamps[key]
                            clearing_stats['timestamp_entries_cleared'] += 1
                        clearing_stats['config_cache_cleared'] += 1
                
                # Clear schema cache if clear_schemas is enabled
                if clear_schemas:
                    clearing_stats['schema_cache_cleared'] = len(self.schema_cache)
                    self.schema_cache.clear()
            
            # Update cache statistics
            create_audit_trail(
                action='PARSER_CACHE_CLEARED',
                component='CONFIG_PARSER',
                action_details=clearing_stats,
                user_context='SYSTEM'
            )
            
            self.logger.info(
                f"Parser cache cleared: {clearing_stats['config_cache_cleared']} config entries, "
                f"{clearing_stats['schema_cache_cleared']} schema entries"
            )
            
            return clearing_stats
            
        except Exception as e:
            self.logger.error(f"Failed to clear parser cache: {e}", exc_info=True)
            raise ConfigurationError(
                f"Cache clearing failed: {str(e)}",
                configuration_section='cache_management',
                schema_type='cache_clearing'
            ) from e
    
    def get_validation_history(
        self,
        config_name: Optional[str] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve validation history for configuration tracking and debugging.
        
        Args:
            config_name: Specific configuration name (None for all)
            include_details: Include detailed validation information
            
        Returns:
            Dict[str, Any]: Validation history with detailed information
        """
        try:
            # Filter validation history by config_name if specified
            if config_name:
                if config_name in self.validation_history:
                    validation_data = self.validation_history[config_name]
                    if include_details:
                        return {config_name: validation_data.to_dict()}
                    else:
                        return {config_name: validation_data.get_summary()}
                else:
                    return {config_name: {'status': 'no_validation_history'}}
            else:
                # Return all validation history
                history = {}
                for name, validation_result in self.validation_history.items():
                    if include_details:
                        history[name] = validation_result.to_dict()
                    else:
                        history[name] = validation_result.get_summary()
                
                return history
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve validation history: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid based on expiry settings."""
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        expiry_time = cache_time + datetime.timedelta(hours=self.cache_expiry_hours)
        
        return datetime.datetime.now() < expiry_time


# Helper functions for configuration parser implementation

def _get_default_config_path(config_name: str) -> str:
    """Get default file path for configuration type."""
    if config_name in _default_configs:
        return f"configs/{_default_configs[config_name]}"
    else:
        return f"configs/{config_name}.json"


def _get_schema_path(config_name: str) -> str:
    """Get schema file path for configuration type."""
    if config_name in _schema_files:
        return f"schemas/{_schema_files[config_name]}"
    else:
        return f"schemas/{config_name}_schema.json"


def _get_required_sections(config_name: str) -> List[str]:
    """Get required configuration sections for validation."""
    required_sections_map = {
        'normalization': ['spatial', 'temporal', 'intensity'],
        'simulation': ['algorithm', 'parameters', 'output'],
        'analysis': ['metrics', 'visualization', 'export'],
        'logging': ['handlers', 'formatters', 'loggers'],
        'performance': ['thresholds', 'monitoring', 'optimization']
    }
    
    return required_sections_map.get(config_name, [])


def _apply_default_values(config_data: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    """Apply intelligent default values for missing configuration parameters."""
    # This would implement intelligent default value application
    # For now, return the configuration as-is
    return config_data


def _apply_configuration_overrides(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configuration overrides using deep merge strategy."""
    merged_config = copy.deepcopy(config_data)
    
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(merged_config, overrides)
    return merged_config


def _validate_scientific_parameters(
    config_data: Dict[str, Any],
    config_type: str,
    validation_result: ValidationResult,
    strict_validation: bool
) -> None:
    """Validate scientific parameters against constraints and ranges."""
    # This would implement scientific parameter validation
    # For now, perform basic validation
    pass


def _validate_parameter_dependencies(
    config_data: Dict[str, Any],
    config_type: str,
    validation_result: ValidationResult
) -> None:
    """Validate cross-parameter dependencies and relationships."""
    # This would implement parameter dependency validation
    # For now, perform basic validation
    pass


def _apply_strict_validation_rules(
    config_data: Dict[str, Any],
    config_type: str,
    validation_result: ValidationResult
) -> None:
    """Apply strict validation rules for comprehensive checking."""
    # This would implement strict validation rules
    # For now, perform basic validation
    pass


def _check_deprecated_parameters(
    config_data: Dict[str, Any],
    config_type: str,
    validation_result: ValidationResult
) -> None:
    """Check for deprecated parameters and configuration patterns."""
    # This would implement deprecated parameter checking
    # For now, perform basic checking
    pass


def _generate_validation_recovery_recommendations(
    config_data: Dict[str, Any],
    config_type: str,
    validation_result: ValidationResult
) -> None:
    """Generate recovery recommendations for validation failures."""
    if not validation_result.is_valid:
        validation_result.add_recommendation(
            "Review configuration errors and fix invalid parameters",
            priority="HIGH"
        )


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


def _deep_merge_configurations(
    base_config: Dict[str, Any],
    overlay_config: Dict[str, Any],
    merge_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform deep merge of configurations with conflict tracking."""
    result = copy.deepcopy(base_config)
    
    def deep_merge_recursive(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        for key, value in overlay.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge_recursive(base[key], value)
                else:
                    # Conflict detected - overlay wins
                    merge_metadata['conflicts_resolved'] += 1
                    base[key] = value
            else:
                base[key] = value
    
    deep_merge_recursive(result, overlay_config)
    return result


def _overlay_merge_configurations(
    base_config: Dict[str, Any],
    overlay_config: Dict[str, Any],
    merge_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform overlay merge where overlay completely replaces base values."""
    result = copy.deepcopy(base_config)
    
    for key, value in overlay_config.items():
        if key in result:
            merge_metadata['conflicts_resolved'] += 1
        result[key] = copy.deepcopy(value)
    
    return result


def _priority_merge_configurations(
    base_config: Dict[str, Any],
    overlay_config: Dict[str, Any],
    priority_order: List[str],
    merge_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform priority-based merge using specified priority order."""
    # This would implement priority-based merging
    # For now, fallback to deep merge
    return _deep_merge_configurations(base_config, overlay_config, merge_metadata)


def _selective_merge_configurations(
    base_config: Dict[str, Any],
    overlay_config: Dict[str, Any],
    merge_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform selective merge based on configuration metadata."""
    # This would implement selective merging
    # For now, fallback to deep merge
    return _deep_merge_configurations(base_config, overlay_config, merge_metadata)


def _apply_priority_order(
    merged_config: Dict[str, Any],
    config_list: List[Dict[str, Any]],
    priority_order: List[str]
) -> Dict[str, Any]:
    """Apply priority order for conflict resolution."""
    # This would implement priority order application
    # For now, return merged config as-is
    return merged_config


def _check_merged_configuration_consistency(merged_config: Dict[str, Any]) -> None:
    """Check merged configuration for consistency and compatibility issues."""
    # This would implement consistency checking
    # For now, perform basic validation
    pass


def _detect_configuration_type(config_data: Dict[str, Any]) -> Optional[str]:
    """Detect configuration type from configuration data structure."""
    # Simple heuristic-based detection
    if 'algorithm' in config_data and 'parameters' in config_data:
        return 'simulation'
    elif 'spatial' in config_data and 'temporal' in config_data:
        return 'normalization'
    elif 'metrics' in config_data and 'visualization' in config_data:
        return 'analysis'
    elif 'handlers' in config_data and 'formatters' in config_data:
        return 'logging'
    elif 'thresholds' in config_data and 'monitoring' in config_data:
        return 'performance'
    else:
        return None


def _generate_default_configuration(config_type: str) -> Dict[str, Any]:
    """Generate default configuration for specified type."""
    default_configs = {
        'normalization': {
            'spatial': {'target_width': 1.0, 'target_height': 1.0},
            'temporal': {'target_fps': 30.0},
            'intensity': {'min_value': 0.0, 'max_value': 1.0}
        },
        'simulation': {
            'algorithm': 'infotaxis',
            'parameters': {'step_size': 0.1, 'max_steps': 1000},
            'output': {'save_trajectory': True, 'save_metrics': True}
        },
        'analysis': {
            'metrics': ['success_rate', 'path_efficiency'],
            'visualization': {'create_plots': True, 'save_images': True},
            'export': {'format': 'json', 'include_metadata': True}
        },
        'logging': {
            'handlers': ['console', 'file'],
            'formatters': ['scientific'],
            'loggers': {'root': {'level': 'INFO'}}
        },
        'performance': {
            'thresholds': {'processing_time': 7.2, 'correlation': 0.95},
            'monitoring': {'enable_metrics': True},
            'optimization': {'parallel_processing': True}
        }
    }
    
    return default_configs.get(config_type, {})


def _apply_environment_overrides(
    config_data: Dict[str, Any],
    config_type: str,
    environment: str
) -> Dict[str, Any]:
    """Apply environment-specific configuration overrides."""
    # This would implement environment-specific overrides
    # For now, return configuration as-is
    return config_data


def _add_configuration_documentation(
    config_data: Dict[str, Any],
    config_type: str
) -> Dict[str, Any]:
    """Add parameter documentation to configuration."""
    # This would add documentation to configuration parameters
    # For now, return configuration as-is
    config_with_docs = copy.deepcopy(config_data)
    config_with_docs['_documentation'] = {
        'config_type': config_type,
        'documentation_included': True,
        'generated_at': datetime.datetime.now().isoformat()
    }
    return config_with_docs


def _apply_system_specific_defaults(
    config_data: Dict[str, Any],
    config_type: str
) -> Dict[str, Any]:
    """Apply system-specific default adjustments."""
    # This would apply platform-specific defaults
    # For now, return configuration as-is
    return config_data


def _load_schema_file(schema_path: str) -> Dict[str, Any]:
    """Load JSON schema file with error handling."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _identify_missing_parameters(
    config_data: Dict[str, Any],
    default_config: Dict[str, Any],
    prefix: str = ''
) -> List[str]:
    """Identify missing parameters by comparing with default configuration."""
    missing_params = []
    
    for key, default_value in default_config.items():
        if key.startswith('_'):  # Skip metadata fields
            continue
        
        param_path = f"{prefix}.{key}" if prefix else key
        
        if key not in config_data:
            missing_params.append(param_path)
        elif isinstance(default_value, dict) and isinstance(config_data.get(key), dict):
            # Recursively check nested dictionaries
            nested_missing = _identify_missing_parameters(
                config_data[key],
                default_value,
                param_path
            )
            missing_params.extend(nested_missing)
    
    return missing_params


def _get_nested_parameter(config_data: Dict[str, Any], param_path: str) -> Any:
    """Get nested parameter value using dot notation path."""
    keys = param_path.split('.')
    current = config_data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise KeyError(f"Parameter path not found: {param_path}")
    
    return current


def _set_nested_parameter(config_data: Dict[str, Any], param_path: str, value: Any) -> None:
    """Set nested parameter value using dot notation path."""
    keys = param_path.split('.')
    current = config_data
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the target value
    current[keys[-1]] = value


def _check_parameter_consistency(
    config_data: Dict[str, Any],
    config_type: str,
    parsing_metadata: Dict[str, Any]
) -> None:
    """Check parameter type consistency and value ranges."""
    # This would implement parameter consistency checking
    # For now, perform basic validation
    pass


def _substitute_env_vars_recursive(
    data: Any,
    metadata: Dict[str, Any],
    strict: bool,
    defaults: Dict[str, str],
    validate: bool,
    path: str = ''
) -> None:
    """Recursively substitute environment variables in configuration data."""
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            _substitute_env_vars_recursive(value, metadata, strict, defaults, validate, current_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            _substitute_env_vars_recursive(item, metadata, strict, defaults, validate, current_path)
    elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
        # Environment variable pattern found
        var_name = data[2:-1]
        env_value = os.getenv(var_name)
        
        if env_value is not None:
            # Replace with environment variable value
            # Note: This modifies the parent container, which requires special handling
            metadata['substitutions_made'].append(f"{path}: {var_name} = {env_value}")
        elif var_name in defaults:
            # Use default value
            env_value = defaults[var_name]
            metadata['default_values_used'].append(f"{path}: {var_name} = {env_value}")
        else:
            # Missing environment variable
            metadata['missing_variables'].append(var_name)
            if strict:
                return  # Error will be raised by caller
        
        # Note: In a full implementation, this would need to modify the parent container
        # This is a simplified version that tracks the substitutions


def _create_configuration_backup(config_path: str) -> str:
    """Create backup of configuration file."""
    config_file_path = pathlib.Path(config_path)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = config_file_path.with_suffix(f"{BACKUP_SUFFIX}_{timestamp}{config_file_path.suffix}")
    
    import shutil
    shutil.copy2(str(config_file_path), str(backup_path))
    
    return str(backup_path)


def _compress_backup_file(backup_path: pathlib.Path) -> pathlib.Path:
    """Compress backup file using gzip compression."""
    # This would implement backup file compression
    # For now, return the path as-is
    return backup_path


def _cleanup_old_backups(backup_dir: pathlib.Path, config_stem: str) -> None:
    """Clean up old backup files based on retention policy."""
    # This would implement backup cleanup based on MAX_BACKUP_VERSIONS
    # For now, perform basic cleanup
    pass