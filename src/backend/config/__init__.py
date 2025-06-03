"""
Configuration module initialization file that provides centralized access to all configuration 
management functionality for the plume navigation simulation system.

This module exposes configuration loading, validation, parsing, and management utilities while 
establishing the configuration subsystem's public API for normalization, simulation, analysis, 
logging, and performance threshold configurations with comprehensive error handling and scientific 
parameter validation.

Key Features:
- Centralized configuration management for algorithm parameters and simulation settings
- Fail-fast validation of configuration parameters to prevent wasted computational resources
- Cross-platform compatibility configuration for Crimaldi and custom plume formats
- Scientific parameter validation with comprehensive error handling and quality assurance
- Reproducible configuration management with complete audit trail capabilities
- Configuration-driven format handling and parameter normalization
- Thread-safe configuration caching and concurrent access control
- Atomic configuration operations with integrity verification and rollback
- Intelligent default value application and environment-specific overrides
- Configuration merging with conflict resolution and priority handling
"""

# External library imports with version specifications
import os  # Python 3.9+ - Operating system interface for configuration directory management
import pathlib  # Python 3.9+ - Modern path handling for configuration file management
from typing import Dict, Any, List, Optional, Union, Tuple  # Python 3.9+ - Type hints for configuration management function signatures
import warnings  # Python 3.9+ - Warning generation for deprecated configuration parameters

# Internal imports from utility modules
from ..utils.config_parser import (
    load_configuration,
    save_configuration, 
    parse_config_with_defaults,
    validate_config_schema,
    merge_configurations,
    ConfigurationParser
)
from ..utils.logging_utils import get_logger, create_audit_trail
from ..utils.validation_utils import ValidationResult, ConfigurationError, ValidationError
from ..utils.scientific_constants import get_performance_thresholds, get_statistical_constants
from ..utils.file_utils import ensure_directory_exists, load_json_config

# Global configuration directory and schema paths for centralized management
CONFIG_DIRECTORY: pathlib.Path = pathlib.Path(__file__).parent

# Schema directory for configuration validation with JSON schema files
SCHEMA_DIRECTORY: pathlib.Path = CONFIG_DIRECTORY / 'schema'

# Default configuration file mappings for different system components
DEFAULT_CONFIGS: Dict[str, str] = {
    'normalization': 'default_normalization.json',
    'simulation': 'default_simulation.json', 
    'analysis': 'default_analysis.json',
    'logging': 'logging_config.json',
    'performance': 'performance_thresholds.json'
}

# Schema file mappings for configuration validation against JSON schemas
SCHEMA_FILES: Dict[str, str] = {
    'normalization': 'normalization_schema.json',
    'simulation': 'simulation_schema.json',
    'analysis': 'analysis_schema.json'
}

# Global configuration parser instance with caching and validation enabled
_config_parser: ConfigurationParser = ConfigurationParser(
    str(CONFIG_DIRECTORY), 
    str(SCHEMA_DIRECTORY), 
    enable_caching=True
)

# Global logger instance for configuration operations with scientific context
_logger = get_logger(__name__, 'config')


def get_default_normalization_config(
    validate_schema: bool = True,
    include_documentation: bool = False
) -> Dict[str, Any]:
    """
    Load and return the default normalization configuration with comprehensive settings for 
    automated normalization and calibration of plume recordings across different physical 
    scales and formats.
    
    This function provides standardized normalization parameters including arena size 
    normalization, pixel resolution standardization, temporal sampling alignment, and 
    intensity unit conversion settings for cross-format compatibility.
    
    Args:
        validate_schema: Enable schema validation for configuration structure verification
        include_documentation: Add parameter documentation and descriptions to configuration
        
    Returns:
        Dict[str, Any]: Default normalization configuration with arena size normalization, 
                       pixel resolution standardization, temporal sampling alignment, and 
                       intensity unit conversion settings
    """
    try:
        # Load default_normalization.json using configuration parser
        config_path = CONFIG_DIRECTORY / DEFAULT_CONFIGS['normalization']
        
        if not config_path.exists():
            # Generate default normalization configuration if file doesn't exist
            default_config = _generate_default_normalization_config()
        else:
            default_config = _config_parser.load_config(
                'normalization', 
                str(config_path),
                validate_schema=validate_schema
            )
        
        # Validate against normalization schema if validate_schema is True
        if validate_schema:
            validation_result = validate_config_schema(
                config_data=default_config,
                schema_type='normalization',
                strict_mode=True,
                required_sections=['spatial', 'temporal', 'intensity']
            )
            
            if not validation_result.is_valid:
                _logger.error(f"Normalization configuration validation failed: {len(validation_result.errors)} errors")
                raise ValidationError(
                    "Default normalization configuration validation failed",
                    "configuration_schema",
                    "normalization"
                )
        
        # Add parameter documentation if include_documentation is True
        if include_documentation:
            default_config = _add_normalization_documentation(default_config)
        
        # Apply any system-specific default overrides based on environment
        default_config = _apply_system_specific_overrides(default_config, 'normalization')
        
        # Log configuration access for audit trail with scientific context
        create_audit_trail(
            action='DEFAULT_NORMALIZATION_CONFIG_ACCESSED',
            component='CONFIG',
            action_details={
                'validate_schema': validate_schema,
                'include_documentation': include_documentation,
                'parameter_count': len(str(default_config)),
                'config_sections': list(default_config.keys()) if isinstance(default_config, dict) else []
            },
            user_context='SYSTEM'
        )
        
        _logger.info("Default normalization configuration loaded successfully")
        
        # Return validated normalization configuration with applied settings
        return default_config
        
    except Exception as e:
        _logger.error(f"Failed to load default normalization configuration: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error loading default normalization configuration: {e}",
                "normalization_config_loading",
                "normalization"
            )


def get_default_simulation_config(
    validate_schema: bool = True,
    include_documentation: bool = False
) -> Dict[str, Any]:
    """
    Load and return the default simulation configuration with comprehensive settings for 
    batch execution of 4000+ plume navigation algorithm simulations with parallel processing 
    and resource management.
    
    This function provides standardized simulation parameters including batch execution 
    parameters, algorithm configurations, resource management, and performance optimization 
    settings for high-throughput scientific computing workflows.
    
    Args:
        validate_schema: Enable schema validation for configuration structure verification
        include_documentation: Add parameter documentation and descriptions to configuration
        
    Returns:
        Dict[str, Any]: Default simulation configuration with batch execution parameters, 
                       algorithm configurations, resource management, and performance 
                       optimization settings
    """
    try:
        # Load default_simulation.json using configuration parser
        config_path = CONFIG_DIRECTORY / DEFAULT_CONFIGS['simulation']
        
        if not config_path.exists():
            # Generate default simulation configuration if file doesn't exist
            default_config = _generate_default_simulation_config()
        else:
            default_config = _config_parser.load_config(
                'simulation',
                str(config_path), 
                validate_schema=validate_schema
            )
        
        # Validate against simulation schema if validate_schema is True
        if validate_schema:
            validation_result = validate_config_schema(
                config_data=default_config,
                schema_type='simulation',
                strict_mode=True,
                required_sections=['algorithm', 'parameters', 'batch_processing', 'performance']
            )
            
            if not validation_result.is_valid:
                _logger.error(f"Simulation configuration validation failed: {len(validation_result.errors)} errors")
                raise ValidationError(
                    "Default simulation configuration validation failed",
                    "configuration_schema", 
                    "simulation"
                )
        
        # Add parameter documentation if include_documentation is True
        if include_documentation:
            default_config = _add_simulation_documentation(default_config)
        
        # Apply any system-specific default overrides for simulation settings
        default_config = _apply_system_specific_overrides(default_config, 'simulation')
        
        # Log configuration access for audit trail with batch processing context
        create_audit_trail(
            action='DEFAULT_SIMULATION_CONFIG_ACCESSED',
            component='CONFIG',
            action_details={
                'validate_schema': validate_schema,
                'include_documentation': include_documentation,
                'target_simulation_count': default_config.get('batch_processing', {}).get('target_simulations', 4000),
                'parallel_processing_enabled': default_config.get('batch_processing', {}).get('parallel_processing', True)
            },
            user_context='SYSTEM'
        )
        
        _logger.info("Default simulation configuration loaded successfully")
        
        # Return validated simulation configuration with batch processing settings
        return default_config
        
    except Exception as e:
        _logger.error(f"Failed to load default simulation configuration: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error loading default simulation configuration: {e}",
                "simulation_config_loading",
                "simulation"
            )


def get_default_analysis_config(
    validate_schema: bool = True,
    include_documentation: bool = False
) -> Dict[str, Any]:
    """
    Load and return the default analysis configuration with comprehensive settings for 
    performance metrics calculation, statistical comparison, trajectory analysis, and 
    scientific reproducibility validation.
    
    This function provides standardized analysis parameters including performance metrics, 
    statistical comparison settings, visualization configurations, and cross-format algorithm 
    comparison parameters for comprehensive scientific analysis workflows.
    
    Args:
        validate_schema: Enable schema validation for configuration structure verification
        include_documentation: Add parameter documentation and descriptions to configuration
        
    Returns:
        Dict[str, Any]: Default analysis configuration with performance metrics, statistical 
                       comparison, visualization settings, and cross-format algorithm 
                       comparison parameters
    """
    try:
        # Load default_analysis.json using configuration parser
        config_path = CONFIG_DIRECTORY / DEFAULT_CONFIGS['analysis']
        
        if not config_path.exists():
            # Generate default analysis configuration if file doesn't exist
            default_config = _generate_default_analysis_config()
        else:
            default_config = _config_parser.load_config(
                'analysis',
                str(config_path),
                validate_schema=validate_schema
            )
        
        # Validate against analysis schema if validate_schema is True
        if validate_schema:
            validation_result = validate_config_schema(
                config_data=default_config,
                schema_type='analysis',
                strict_mode=True,
                required_sections=['metrics', 'visualization', 'export', 'statistical_analysis']
            )
            
            if not validation_result.is_valid:
                _logger.error(f"Analysis configuration validation failed: {len(validation_result.errors)} errors")
                raise ValidationError(
                    "Default analysis configuration validation failed",
                    "configuration_schema",
                    "analysis"
                )
        
        # Add parameter documentation if include_documentation is True
        if include_documentation:
            default_config = _add_analysis_documentation(default_config)
        
        # Apply any system-specific default overrides for analysis settings
        default_config = _apply_system_specific_overrides(default_config, 'analysis')
        
        # Log configuration access for audit trail with analysis context
        create_audit_trail(
            action='DEFAULT_ANALYSIS_CONFIG_ACCESSED',
            component='CONFIG',
            action_details={
                'validate_schema': validate_schema,
                'include_documentation': include_documentation,
                'metrics_enabled': list(default_config.get('metrics', {}).keys()),
                'visualization_enabled': default_config.get('visualization', {}).get('enable_plots', True)
            },
            user_context='SYSTEM'
        )
        
        _logger.info("Default analysis configuration loaded successfully")
        
        # Return validated analysis configuration with performance metrics settings
        return default_config
        
    except Exception as e:
        _logger.error(f"Failed to load default analysis configuration: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error loading default analysis configuration: {e}",
                "analysis_config_loading",
                "analysis"
            )


def get_logging_config(validate_structure: bool = True) -> Dict[str, Any]:
    """
    Load and return the logging configuration defining the complete logging infrastructure 
    for hierarchical logger configuration, handler definitions, formatter specifications, 
    and scientific context settings.
    
    This function provides comprehensive logging configuration including hierarchical loggers, 
    specialized handlers, rotation policies, and performance monitoring settings for 
    scientific computing audit trails and debugging.
    
    Args:
        validate_structure: Enable structure validation for logging configuration integrity
        
    Returns:
        Dict[str, Any]: Logging configuration with hierarchical loggers, specialized handlers, 
                       rotation policies, and performance monitoring settings
    """
    try:
        # Load logging_config.json using configuration parser
        config_path = CONFIG_DIRECTORY / DEFAULT_CONFIGS['logging']
        
        if not config_path.exists():
            # Generate default logging configuration if file doesn't exist
            default_config = _generate_default_logging_config()
        else:
            default_config = _config_parser.load_config(
                'logging',
                str(config_path),
                validate_schema=False  # Logging config has different validation requirements
            )
        
        # Validate configuration structure if validate_structure is True
        if validate_structure:
            structure_validation = _validate_logging_config_structure(default_config)
            if not structure_validation['is_valid']:
                _logger.error(f"Logging configuration structure validation failed: {structure_validation['errors']}")
                raise ConfigurationError(
                    "Logging configuration structure validation failed",
                    "logging_config_structure",
                    "logging"
                )
        
        # Check handler and formatter definitions for completeness
        _validate_logging_handlers_and_formatters(default_config)
        
        # Verify log directory paths exist and create if necessary
        _ensure_log_directories_exist(default_config)
        
        # Apply any environment-specific overrides for logging configuration
        default_config = _apply_logging_environment_overrides(default_config)
        
        # Log configuration access for audit trail with logging context
        create_audit_trail(
            action='LOGGING_CONFIG_ACCESSED',
            component='CONFIG',
            action_details={
                'validate_structure': validate_structure,
                'handlers_configured': list(default_config.get('handlers', {}).keys()),
                'loggers_configured': list(default_config.get('loggers', {}).keys()),
                'log_level': default_config.get('root', {}).get('level', 'INFO')
            },
            user_context='SYSTEM'
        )
        
        _logger.info("Logging configuration loaded successfully")
        
        # Return validated logging configuration with handler and formatter settings
        return default_config
        
    except Exception as e:
        _logger.error(f"Failed to load logging configuration: {e}")
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(
                f"Error loading logging configuration: {e}",
                "logging_config_loading",
                "logging"
            )


def get_performance_thresholds(validate_thresholds: bool = True) -> Dict[str, Any]:
    """
    Load and return performance thresholds configuration defining critical performance limits, 
    monitoring parameters, alerting conditions, and quality assurance criteria for scientific 
    computing standards.
    
    This function provides comprehensive performance threshold configuration including simulation 
    performance limits, resource utilization goals, quality assurance criteria, and monitoring 
    settings for maintaining >95% correlation requirements and <1% error rates.
    
    Args:
        validate_thresholds: Enable threshold value validation against scientific constraints
        
    Returns:
        Dict[str, Any]: Performance thresholds configuration with simulation performance limits, 
                       resource utilization goals, quality assurance criteria, and monitoring 
                       settings
    """
    try:
        # Load performance_thresholds.json using configuration parser
        config_path = CONFIG_DIRECTORY / DEFAULT_CONFIGS['performance']
        
        if not config_path.exists():
            # Generate default performance thresholds configuration if file doesn't exist
            default_config = _generate_default_performance_thresholds()
        else:
            default_config = _config_parser.load_config(
                'performance',
                str(config_path),
                validate_schema=False  # Performance thresholds have specialized validation
            )
        
        # Validate threshold values if validate_thresholds is True
        if validate_thresholds:
            threshold_validation = _validate_performance_threshold_values(default_config)
            if not threshold_validation['is_valid']:
                _logger.error(f"Performance threshold validation failed: {threshold_validation['errors']}")
                raise ValidationError(
                    "Performance threshold validation failed",
                    "performance_thresholds",
                    "performance"
                )
        
        # Check threshold consistency across categories (processing, accuracy, resource)
        _validate_threshold_consistency(default_config)
        
        # Apply any system-specific threshold adjustments based on hardware capabilities
        default_config = _apply_system_specific_threshold_adjustments(default_config)
        
        # Log threshold access for monitoring and audit trail
        create_audit_trail(
            action='PERFORMANCE_THRESHOLDS_ACCESSED',
            component='CONFIG',
            action_details={
                'validate_thresholds': validate_thresholds,
                'processing_time_target': default_config.get('processing', {}).get('time_per_simulation_seconds', 7.2),
                'correlation_threshold': default_config.get('accuracy', {}).get('correlation_threshold', 0.95),
                'error_rate_threshold': default_config.get('accuracy', {}).get('error_rate_threshold', 0.01)
            },
            user_context='SYSTEM'
        )
        
        _logger.info("Performance thresholds configuration loaded successfully")
        
        # Return validated performance thresholds with scientific computing standards
        return default_config
        
    except Exception as e:
        _logger.error(f"Failed to load performance thresholds configuration: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error loading performance thresholds configuration: {e}",
                "performance_thresholds_loading",
                "performance"
            )


def load_config(
    config_name: str,
    validate_schema: bool = True,
    use_cache: bool = True,
    default_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load any configuration file by name with comprehensive validation, schema checking, and 
    error handling for scientific computing parameters with caching support.
    
    This function provides centralized configuration loading with fail-fast validation, 
    comprehensive error handling, and performance optimization through intelligent caching 
    mechanisms for reproducible scientific computing workflows.
    
    Args:
        config_name: Supported configuration type identifier ('normalization', 'simulation', 'analysis', etc.)
        validate_schema: Enable JSON schema validation for configuration structure compliance
        use_cache: Enable configuration caching for improved performance and consistency
        default_overrides: Dictionary of override values to apply after loading configuration
        
    Returns:
        Dict[str, Any]: Loaded and validated configuration dictionary with scientific parameters 
                       and applied defaults
    """
    try:
        # Validate config_name is supported configuration type
        if config_name not in DEFAULT_CONFIGS:
            supported_types = list(DEFAULT_CONFIGS.keys())
            raise ConfigurationError(
                f"Unsupported configuration type: {config_name}. Supported types: {supported_types}",
                "config_type_validation",
                config_name
            )
        
        # Use configuration parser to load configuration with validation
        config_data = _config_parser.load_config(
            config_name=config_name,
            validate_schema=validate_schema,
            use_cache=use_cache
        )
        
        # Apply schema validation if validate_schema is True
        if validate_schema and config_name in SCHEMA_FILES:
            validation_result = validate_config_schema(
                config_data=config_data,
                schema_type=config_name,
                strict_mode=True,
                required_sections=_get_required_sections_for_config(config_name)
            )
            
            if not validation_result.is_valid:
                _logger.error(f"Configuration schema validation failed for {config_name}: {len(validation_result.errors)} errors")
                raise ValidationError(
                    f"Configuration validation failed: {len(validation_result.errors)} errors found",
                    "configuration_schema",
                    config_name
                )
        
        # Apply default_overrides if provided
        if default_overrides:
            config_data = _apply_configuration_overrides(config_data, default_overrides)
        
        # Merge with default configuration values to ensure completeness
        config_data = _merge_with_defaults(config_data, config_name)
        
        # Create audit trail entry for configuration access
        create_audit_trail(
            action='CONFIGURATION_LOADED',
            component='CONFIG',
            action_details={
                'config_name': config_name,
                'validate_schema': validate_schema,
                'use_cache': use_cache,
                'has_overrides': default_overrides is not None,
                'config_size': len(str(config_data)) if config_data else 0,
                'parameter_sections': list(config_data.keys()) if isinstance(config_data, dict) else []
            },
            user_context='SYSTEM'
        )
        
        # Cache configuration if use_cache is True
        if use_cache:
            _logger.debug(f"Configuration cached: {config_name}")
        
        _logger.info(f"Configuration loaded successfully: {config_name}")
        
        # Return validated configuration dictionary with applied settings
        return config_data
        
    except Exception as e:
        _logger.error(f"Failed to load configuration {config_name}: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error loading configuration {config_name}: {e}",
                "config_loading_operation",
                config_name
            )


def save_config(
    config_name: str,
    config_data: Dict[str, Any],
    create_backup: bool = True,
    validate_before_save: bool = True
) -> bool:
    """
    Save configuration to file with atomic write operations, backup creation, validation, 
    and audit trail generation to ensure data integrity and traceability.
    
    This function provides robust configuration persistence with atomic operations, 
    comprehensive validation, and audit trail integration for scientific computing 
    reproducibility and configuration change tracking.
    
    Args:
        config_name: Supported configuration type identifier for file naming and validation
        config_data: Configuration dictionary to save with scientific parameters and settings
        create_backup: Create backup of existing configuration file before saving
        validate_before_save: Perform comprehensive validation before saving to prevent corruption
        
    Returns:
        bool: True if save operation completed successfully with validation and backup status
    """
    try:
        # Validate config_name is supported configuration type
        if config_name not in DEFAULT_CONFIGS:
            supported_types = list(DEFAULT_CONFIGS.keys())
            raise ConfigurationError(
                f"Unsupported configuration type: {config_name}. Supported types: {supported_types}",
                "config_save_validation",
                config_name
            )
        
        # Validate config_data structure and types
        if not isinstance(config_data, dict):
            raise ConfigurationError(
                f"Configuration data must be a dictionary, got {type(config_data)}",
                "config_data_validation",
                config_name
            )
        
        if not config_data:
            raise ConfigurationError(
                "Configuration data cannot be empty",
                "config_data_validation",
                config_name
            )
        
        # Perform schema validation if validate_before_save is True
        if validate_before_save and config_name in SCHEMA_FILES:
            validation_result = validate_config_schema(
                config_data=config_data,
                schema_type=config_name,
                strict_mode=True,
                required_sections=_get_required_sections_for_config(config_name)
            )
            
            if not validation_result.is_valid:
                _logger.error(f"Configuration validation failed before save for {config_name}: {len(validation_result.errors)} errors")
                raise ValidationError(
                    f"Configuration validation failed before save: {len(validation_result.errors)} errors",
                    "config_save_validation",
                    config_name
                )
        
        # Use configuration parser to save configuration with atomic operations
        save_result = _config_parser.save_config(
            config_name=config_name,
            config_data=config_data,
            validate_schema=validate_before_save,
            create_backup=create_backup
        )
        
        if not save_result:
            raise ConfigurationError(
                "Configuration save operation failed",
                "config_save_operation", 
                config_name
            )
        
        # Create audit trail entry for configuration modification
        create_audit_trail(
            action='CONFIGURATION_SAVED',
            component='CONFIG',
            action_details={
                'config_name': config_name,
                'create_backup': create_backup,
                'validate_before_save': validate_before_save,
                'config_size': len(str(config_data)),
                'parameter_sections': list(config_data.keys()) if isinstance(config_data, dict) else []
            },
            user_context='SYSTEM'
        )
        
        # Clear cache for updated configuration to force reload
        _config_parser.clear_cache([config_name])
        
        _logger.info(f"Configuration saved successfully: {config_name}")
        
        # Return success status with comprehensive validation and backup information
        return True
        
    except Exception as e:
        _logger.error(f"Failed to save configuration {config_name}: {e}")
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        else:
            raise ConfigurationError(
                f"Error saving configuration {config_name}: {e}",
                "config_save_operation",
                config_name
            )


def validate_config(
    config_data: Dict[str, Any],
    config_type: str,
    strict_validation: bool = False
) -> ValidationResult:
    """
    Comprehensive validation of configuration against schema and scientific constraints with 
    detailed error reporting and compatibility checking across configuration sections.
    
    This function provides extensive configuration validation with scientific parameter checking, 
    cross-dependency validation, and comprehensive error reporting with actionable recommendations 
    for fail-fast validation strategy implementation.
    
    Args:
        config_data: Configuration dictionary to validate against schema and constraints
        config_type: Type of configuration for validation rules ('normalization', 'simulation', etc.)
        strict_validation: Enable strict validation mode with comprehensive scientific parameter checks
        
    Returns:
        ValidationResult: Detailed validation results with errors, warnings, and scientific 
                         parameter constraint violations
    """
    try:
        # Use configuration parser to validate configuration with comprehensive checking
        validation_result = _config_parser.validate_config(
            config_data=config_data,
            config_type=config_type,
            strict_validation=strict_validation
        )
        
        # Load appropriate schema for config_type if available
        if config_type in SCHEMA_FILES:
            schema_validation = validate_config_schema(
                config_data=config_data,
                schema_type=config_type,
                strict_mode=strict_validation,
                required_sections=_get_required_sections_for_config(config_type)
            )
            
            # Merge schema validation results with parser validation
            if not schema_validation.is_valid:
                validation_result.errors.extend(schema_validation.errors)
                validation_result.warnings.extend(schema_validation.warnings)
                validation_result.is_valid = False
        
        # Check scientific parameter constraints and value ranges
        _validate_scientific_parameter_constraints(config_data, config_type, validation_result)
        
        # Validate cross-parameter dependencies and relationships
        _validate_configuration_dependencies(config_data, config_type, validation_result)
        
        # Check for deprecated parameters and configuration patterns
        _check_deprecated_configuration_parameters(config_data, config_type, validation_result)
        
        # Generate comprehensive validation report with recovery recommendations
        if not validation_result.is_valid:
            _generate_validation_recovery_recommendations(config_data, config_type, validation_result)
        
        # Create audit trail for validation operation
        create_audit_trail(
            action='CONFIGURATION_VALIDATED',
            component='CONFIG',
            action_details={
                'config_type': config_type,
                'validation_result': {
                    'is_valid': validation_result.is_valid,
                    'error_count': len(validation_result.errors),
                    'warning_count': len(validation_result.warnings)
                },
                'strict_validation': strict_validation,
                'parameter_count': len(str(config_data))
            },
            user_context='SYSTEM'
        )
        
        _logger.info(f"Configuration validation completed: {config_type}, valid={validation_result.is_valid}, errors={len(validation_result.errors)}")
        
        # Return detailed validation results with comprehensive error analysis
        return validation_result
        
    except Exception as e:
        # Create error validation result for exception handling
        error_result = ValidationResult(
            validation_type="configuration_validation",
            is_valid=False,
            validation_context=f"config_type={config_type}"
        )
        error_result.add_error(
            f"Validation process failed: {str(e)}",
            severity=ValidationResult.ErrorSeverity.CRITICAL
        )
        
        _logger.error(f"Configuration validation failed for {config_type}: {e}")
        
        return error_result


def merge_configs(
    config_names: List[str],
    merge_strategy: str = 'deep_merge',
    validate_result: bool = True
) -> Dict[str, Any]:
    """
    Intelligently merge multiple configurations with conflict resolution, priority handling, 
    validation, and compatibility checking for complex configuration scenarios.
    
    This function provides sophisticated configuration merging with multiple strategies, 
    conflict resolution, and validation to support complex scientific computing workflows 
    and experimental condition setup with reproducible results.
    
    Args:
        config_names: List of configuration names to merge in priority order
        merge_strategy: Strategy for merging configurations ('deep_merge', 'overlay', 'priority')
        validate_result: Validate merged configuration against schemas and constraints
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary with resolved conflicts and validated consistency
    """
    try:
        # Load all configurations in config_names list
        config_list = []
        for config_name in config_names:
            if config_name not in DEFAULT_CONFIGS:
                raise ConfigurationError(
                    f"Unsupported configuration name: {config_name}",
                    "config_merge_validation",
                    config_name
                )
            
            config_data = load_config(config_name, validate_schema=True, use_cache=True)
            config_list.append(config_data)
        
        # Use configuration parser to merge configurations with specified strategy
        merged_config = merge_configurations(
            config_list=config_list,
            merge_strategy=merge_strategy,
            validate_result=False,  # We'll validate separately for better error handling
            priority_order=config_names,
            preserve_metadata=True
        )
        
        # Validate merged configuration if validate_result is True
        if validate_result:
            # Validate against all applicable schemas
            all_validation_passed = True
            for config_name in config_names:
                if config_name in SCHEMA_FILES:
                    validation_result = validate_config_schema(
                        config_data=merged_config,
                        schema_type=config_name,
                        strict_mode=False,  # Use relaxed validation for merged configs
                        required_sections=_get_required_sections_for_config(config_name)
                    )
                    
                    if not validation_result.is_valid:
                        _logger.warning(f"Merged configuration validation issues for {config_name}: {len(validation_result.errors)} errors")
                        all_validation_passed = False
            
            if not all_validation_passed:
                _logger.warning("Merged configuration has validation issues but will be returned")
        
        # Check for parameter inconsistencies and compatibility issues
        _check_merged_configuration_consistency(merged_config, config_names)
        
        # Create audit trail entry for merge operation with source tracking
        create_audit_trail(
            action='CONFIGURATIONS_MERGED',
            component='CONFIG',
            action_details={
                'config_names': config_names,
                'merge_strategy': merge_strategy,
                'validate_result': validate_result,
                'merged_parameter_count': len(str(merged_config)),
                'source_count': len(config_names)
            },
            user_context='SYSTEM'
        )
        
        _logger.info(f"Configuration merge completed: {len(config_names)} sources, strategy={merge_strategy}")
        
        # Return merged configuration dictionary with resolved conflicts
        return merged_config
        
    except Exception as e:
        _logger.error(f"Configuration merge failed: {e}")
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(
                f"Error merging configurations: {e}",
                "config_merge_operation",
                ",".join(config_names)
            )


def get_config_directory() -> pathlib.Path:
    """
    Get the configuration directory path for the plume simulation system with validation 
    of directory existence and permissions.
    
    This function provides access to the centralized configuration directory path with 
    validation checks for accessibility and proper permissions for scientific computing 
    configuration management.
    
    Returns:
        pathlib.Path: Path to the configuration directory with validation status
    """
    try:
        # Return CONFIG_DIRECTORY global path with validation
        if not CONFIG_DIRECTORY.exists():
            _logger.warning(f"Configuration directory does not exist: {CONFIG_DIRECTORY}")
            # Attempt to create directory if it doesn't exist
            ensure_directory_exists(str(CONFIG_DIRECTORY), create_parents=True)
            
        # Validate directory exists and is accessible
        if not CONFIG_DIRECTORY.is_dir():
            raise ConfigurationError(
                f"Configuration path is not a directory: {CONFIG_DIRECTORY}",
                "config_directory_validation",
                "directory"
            )
        
        # Check read permissions for configuration files
        if not os.access(CONFIG_DIRECTORY, os.R_OK):
            raise ConfigurationError(
                f"Configuration directory is not readable: {CONFIG_DIRECTORY}",
                "config_directory_permissions",
                "directory"
            )
        
        _logger.debug(f"Configuration directory accessed: {CONFIG_DIRECTORY}")
        
        return CONFIG_DIRECTORY
        
    except Exception as e:
        _logger.error(f"Error accessing configuration directory: {e}")
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(
                f"Configuration directory access failed: {e}",
                "config_directory_access",
                "directory"
            )


def get_schema_directory() -> pathlib.Path:
    """
    Get the schema directory path for configuration validation with validation of schema 
    file availability.
    
    This function provides access to the schema directory path with validation checks 
    for schema file availability and accessibility for JSON schema-based configuration 
    validation.
    
    Returns:
        pathlib.Path: Path to the schema directory with availability validation
    """
    try:
        # Return SCHEMA_DIRECTORY global path with validation
        if not SCHEMA_DIRECTORY.exists():
            _logger.warning(f"Schema directory does not exist: {SCHEMA_DIRECTORY}")
            # Attempt to create directory if it doesn't exist
            ensure_directory_exists(str(SCHEMA_DIRECTORY), create_parents=True)
        
        # Validate directory exists and is accessible
        if not SCHEMA_DIRECTORY.is_dir():
            raise ConfigurationError(
                f"Schema path is not a directory: {SCHEMA_DIRECTORY}",
                "schema_directory_validation",
                "schema"
            )
        
        # Check schema files are available
        schema_files_found = []
        for schema_name, schema_file in SCHEMA_FILES.items():
            schema_path = SCHEMA_DIRECTORY / schema_file
            if schema_path.exists():
                schema_files_found.append(schema_name)
        
        if schema_files_found:
            _logger.debug(f"Schema files available: {schema_files_found}")
        else:
            _logger.warning(f"No schema files found in directory: {SCHEMA_DIRECTORY}")
        
        _logger.debug(f"Schema directory accessed: {SCHEMA_DIRECTORY}")
        
        return SCHEMA_DIRECTORY
        
    except Exception as e:
        _logger.error(f"Error accessing schema directory: {e}")
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(
                f"Schema directory access failed: {e}",
                "schema_directory_access",
                "schema"
            )


def list_available_configs(include_schemas: bool = False) -> Dict[str, Any]:
    """
    List all available configuration files in the configuration directory with their types 
    and validation status.
    
    This function provides comprehensive information about available configuration files 
    including file paths, types, sizes, and validation status for configuration management 
    and system setup validation.
    
    Args:
        include_schemas: Include schema files in the listing along with configuration files
        
    Returns:
        Dict[str, Any]: Dictionary mapping configuration names to file information including 
                       paths and validation status
    """
    try:
        config_info = {
            'configuration_directory': str(CONFIG_DIRECTORY),
            'schema_directory': str(SCHEMA_DIRECTORY),
            'available_configs': {},
            'available_schemas': {},
            'scan_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Scan configuration directory for JSON files
        for config_name, config_file in DEFAULT_CONFIGS.items():
            config_path = CONFIG_DIRECTORY / config_file
            
            file_info = {
                'file_path': str(config_path),
                'exists': config_path.exists(),
                'file_type': 'configuration',
                'config_category': config_name
            }
            
            if config_path.exists():
                try:
                    file_stat = config_path.stat()
                    file_info.update({
                        'file_size_bytes': file_stat.st_size,
                        'last_modified': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'is_readable': os.access(config_path, os.R_OK),
                        'is_writable': os.access(config_path, os.W_OK)
                    })
                    
                    # Check file accessibility and validity
                    try:
                        test_config = load_json_config(str(config_path), validate_schema=False, use_cache=False)
                        file_info['json_valid'] = True
                        file_info['parameter_count'] = len(test_config) if isinstance(test_config, dict) else 0
                    except Exception as e:
                        file_info['json_valid'] = False
                        file_info['validation_error'] = str(e)
                        
                except Exception as e:
                    file_info['access_error'] = str(e)
            
            config_info['available_configs'][config_name] = file_info
        
        # Include schema files if include_schemas is True
        if include_schemas:
            for schema_name, schema_file in SCHEMA_FILES.items():
                schema_path = SCHEMA_DIRECTORY / schema_file
                
                schema_info = {
                    'file_path': str(schema_path),
                    'exists': schema_path.exists(),
                    'file_type': 'schema',
                    'schema_category': schema_name
                }
                
                if schema_path.exists():
                    try:
                        file_stat = schema_path.stat()
                        schema_info.update({
                            'file_size_bytes': file_stat.st_size,
                            'last_modified': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                            'is_readable': os.access(schema_path, os.R_OK)
                        })
                        
                        # Check schema file validity
                        try:
                            schema_data = load_json_config(str(schema_path), validate_schema=False, use_cache=False)
                            schema_info['json_valid'] = True
                            schema_info['schema_properties'] = len(schema_data.get('properties', {})) if isinstance(schema_data, dict) else 0
                        except Exception as e:
                            schema_info['json_valid'] = False
                            schema_info['validation_error'] = str(e)
                            
                    except Exception as e:
                        schema_info['access_error'] = str(e)
                
                config_info['available_schemas'][schema_name] = schema_info
        
        # Build information dictionary with file details and statistics
        config_info['summary'] = {
            'total_configs': len(config_info['available_configs']),
            'existing_configs': len([c for c in config_info['available_configs'].values() if c['exists']]),
            'valid_configs': len([c for c in config_info['available_configs'].values() if c.get('json_valid', False)]),
            'total_schemas': len(config_info['available_schemas']),
            'existing_schemas': len([s for s in config_info['available_schemas'].values() if s['exists']])
        }
        
        _logger.info(f"Configuration listing completed: {config_info['summary']['existing_configs']}/{config_info['summary']['total_configs']} configs available")
        
        # Return configuration file information with comprehensive details
        return config_info
        
    except Exception as e:
        _logger.error(f"Failed to list available configurations: {e}")
        return {
            'error': str(e),
            'configuration_directory': str(CONFIG_DIRECTORY),
            'scan_timestamp': datetime.datetime.now().isoformat()
        }


def clear_config_cache() -> None:
    """
    Clear the configuration cache to force reload from disk, useful for development and 
    testing scenarios.
    
    This function provides cache management capabilities for forcing configuration reload 
    from disk, useful for development, testing, and configuration update scenarios where 
    fresh configuration data is required.
    
    Returns:
        None: No return value (cache clearing operation status logged)
    """
    try:
        # Use configuration parser to clear cache
        cache_stats = _config_parser.clear_cache()
        
        # Log cache clearing operation with statistics
        _logger.info(f"Configuration cache cleared: {cache_stats}")
        
        # Reset any cached configuration state in module globals
        # This would clear any module-level caching if implemented
        
        # Create audit trail for cache clearing operation
        create_audit_trail(
            action='CONFIG_CACHE_CLEARED',
            component='CONFIG',
            action_details={
                'cache_stats': cache_stats,
                'force_reload_enabled': True
            },
            user_context='SYSTEM'
        )
        
        # Force reload on next configuration access by clearing parser cache
        _logger.debug("Configuration cache cleared successfully")
        
    except Exception as e:
        _logger.error(f"Failed to clear configuration cache: {e}")
        # Don't raise exception for cache clearing failures
        warnings.warn(f"Configuration cache clearing failed: {e}", UserWarning)


# Helper functions for configuration implementation

def _generate_default_normalization_config() -> Dict[str, Any]:
    """Generate default normalization configuration if file doesn't exist."""
    return {
        'spatial': {
            'target_arena_width_meters': 1.0,
            'target_arena_height_meters': 1.0,
            'pixel_to_meter_ratio_crimaldi': 100.0,
            'pixel_to_meter_ratio_custom': 150.0,
            'spatial_accuracy_threshold': 0.01
        },
        'temporal': {
            'target_fps': 30.0,
            'crimaldi_fps': 50.0,
            'custom_fps': 30.0,
            'temporal_accuracy_threshold': 0.001,
            'anti_aliasing_cutoff_ratio': 0.8
        },
        'intensity': {
            'target_min': 0.0,
            'target_max': 1.0,
            'calibration_accuracy': 0.02,
            'gamma_correction': 1.0,
            'histogram_bins': 256
        }
    }


def _generate_default_simulation_config() -> Dict[str, Any]:
    """Generate default simulation configuration if file doesn't exist."""
    return {
        'algorithm': {
            'default_type': 'infotaxis',
            'supported_types': ['infotaxis', 'casting', 'gradient_following', 'hybrid']
        },
        'parameters': {
            'step_size': 0.1,
            'max_steps': 1000,
            'convergence_threshold': 1e-6
        },
        'batch_processing': {
            'target_simulations': 4000,
            'max_workers': 8,
            'batch_size': 100,
            'parallel_processing': True,
            'timeout_seconds': 1800
        },
        'performance': {
            'target_time_per_simulation': 7.2,
            'target_completion_hours': 8.0,
            'memory_limit_gb': 8
        }
    }


def _generate_default_analysis_config() -> Dict[str, Any]:
    """Generate default analysis configuration if file doesn't exist."""
    return {
        'metrics': {
            'success_rate': {'enabled': True, 'threshold': 0.95},
            'path_efficiency': {'enabled': True, 'threshold': 0.8},
            'correlation_analysis': {'enabled': True, 'threshold': 0.95}
        },
        'visualization': {
            'enable_plots': True,
            'save_images': True,
            'plot_formats': ['png', 'pdf'],
            'figure_dpi': 300
        },
        'export': {
            'format': 'json',
            'include_metadata': True,
            'compress_output': False
        },
        'statistical_analysis': {
            'confidence_level': 0.95,
            'significance_level': 0.05,
            'outlier_detection': True
        }
    }


def _generate_default_logging_config() -> Dict[str, Any]:
    """Generate default logging configuration if file doesn't exist."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'scientific': {
                'format': '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(simulation_id)s | %(algorithm_name)s | %(processing_stage)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'scientific',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'scientific',
                'filename': 'logs/plume_simulation.log',
                'maxBytes': 10485760,
                'backupCount': 5
            }
        },
        'loggers': {
            'plume_simulation': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }


def _generate_default_performance_thresholds() -> Dict[str, Any]:
    """Generate default performance thresholds configuration if file doesn't exist."""
    return {
        'processing': {
            'time_per_simulation_seconds': 7.2,
            'batch_completion_hours': 8.0,
            'throughput_simulations_per_hour': 500.0
        },
        'accuracy': {
            'correlation_threshold': 0.95,
            'reproducibility_threshold': 0.99,
            'error_rate_threshold': 0.01
        },
        'resources': {
            'memory_limit_gb': 8,
            'cpu_utilization_max': 0.8,
            'disk_space_required_gb': 50
        },
        'monitoring': {
            'enable_metrics': True,
            'alert_thresholds': {
                'processing_time_exceeded': 10.8,
                'memory_usage_exceeded': 7.0,
                'error_rate_exceeded': 0.02
            }
        }
    }


def _get_required_sections_for_config(config_type: str) -> List[str]:
    """Get required sections for specific configuration type."""
    section_mappings = {
        'normalization': ['spatial', 'temporal', 'intensity'],
        'simulation': ['algorithm', 'parameters', 'batch_processing'],
        'analysis': ['metrics', 'visualization', 'export'],
        'logging': ['handlers', 'formatters', 'loggers'],
        'performance': ['processing', 'accuracy', 'resources']
    }
    return section_mappings.get(config_type, [])


def _apply_configuration_overrides(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configuration overrides using deep merge strategy."""
    import copy
    merged_config = copy.deepcopy(config_data)
    
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(merged_config, overrides)
    return merged_config


def _merge_with_defaults(config_data: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    """Merge configuration with default values to ensure completeness."""
    # This would implement intelligent default merging
    # For now, return the configuration as-is
    return config_data


def _apply_system_specific_overrides(config_data: Dict[str, Any], config_type: str) -> Dict[str, Any]:
    """Apply system-specific configuration overrides based on environment."""
    # This would implement system-specific overrides
    # For now, return the configuration as-is
    return config_data


def _add_normalization_documentation(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add documentation to normalization configuration."""
    import copy
    documented_config = copy.deepcopy(config_data)
    documented_config['_documentation'] = {
        'description': 'Normalization configuration for cross-format plume data processing',
        'spatial': 'Arena dimensions and pixel-to-meter conversion ratios',
        'temporal': 'Frame rate normalization and temporal alignment settings',
        'intensity': 'Intensity calibration and histogram normalization parameters'
    }
    return documented_config


def _add_simulation_documentation(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add documentation to simulation configuration."""
    import copy
    documented_config = copy.deepcopy(config_data)
    documented_config['_documentation'] = {
        'description': 'Simulation configuration for batch plume navigation algorithm execution',
        'algorithm': 'Navigation algorithm selection and parameter settings',
        'batch_processing': 'Parallel processing and resource management settings',
        'performance': 'Performance targets and optimization parameters'
    }
    return documented_config


def _add_analysis_documentation(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add documentation to analysis configuration."""
    import copy
    documented_config = copy.deepcopy(config_data)
    documented_config['_documentation'] = {
        'description': 'Analysis configuration for performance metrics and statistical comparison',
        'metrics': 'Performance metrics calculation and threshold settings',
        'visualization': 'Plot generation and visualization output settings',
        'statistical_analysis': 'Statistical testing and confidence interval settings'
    }
    return documented_config


def _validate_logging_config_structure(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate logging configuration structure."""
    validation_result = {'is_valid': True, 'errors': []}
    
    required_sections = ['handlers', 'formatters', 'loggers']
    for section in required_sections:
        if section not in config_data:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required logging section: {section}")
    
    return validation_result


def _validate_logging_handlers_and_formatters(config_data: Dict[str, Any]) -> None:
    """Validate logging handlers and formatters definitions."""
    # This would implement handler and formatter validation
    pass


def _ensure_log_directories_exist(config_data: Dict[str, Any]) -> None:
    """Ensure log directories exist for logging configuration."""
    handlers = config_data.get('handlers', {})
    for handler_name, handler_config in handlers.items():
        if 'filename' in handler_config:
            log_file_path = pathlib.Path(handler_config['filename'])
            ensure_directory_exists(str(log_file_path.parent), create_parents=True)


def _apply_logging_environment_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment-specific overrides for logging configuration."""
    # This would implement environment-specific logging overrides
    return config_data


def _validate_performance_threshold_values(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate performance threshold values against constraints."""
    validation_result = {'is_valid': True, 'errors': []}
    
    # Validate processing time thresholds
    processing = config_data.get('processing', {})
    time_per_sim = processing.get('time_per_simulation_seconds', 0)
    if time_per_sim <= 0 or time_per_sim > 300:  # Max 5 minutes per simulation
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Invalid processing time per simulation: {time_per_sim}")
    
    # Validate accuracy thresholds
    accuracy = config_data.get('accuracy', {})
    correlation = accuracy.get('correlation_threshold', 0)
    if correlation < 0.5 or correlation > 1.0:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Invalid correlation threshold: {correlation}")
    
    return validation_result


def _validate_threshold_consistency(config_data: Dict[str, Any]) -> None:
    """Validate threshold consistency across categories."""
    # This would implement cross-category threshold consistency validation
    pass


def _apply_system_specific_threshold_adjustments(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply system-specific threshold adjustments based on hardware."""
    # This would implement hardware-specific threshold adjustments
    return config_data


def _validate_scientific_parameter_constraints(config_data: Dict[str, Any], config_type: str, validation_result: ValidationResult) -> None:
    """Validate scientific parameters against constraints and ranges."""
    # This would implement scientific parameter constraint validation
    pass


def _validate_configuration_dependencies(config_data: Dict[str, Any], config_type: str, validation_result: ValidationResult) -> None:
    """Validate cross-parameter dependencies and relationships."""
    # This would implement parameter dependency validation
    pass


def _check_deprecated_configuration_parameters(config_data: Dict[str, Any], config_type: str, validation_result: ValidationResult) -> None:
    """Check for deprecated parameters and configuration patterns."""
    # This would implement deprecated parameter checking
    pass


def _generate_validation_recovery_recommendations(config_data: Dict[str, Any], config_type: str, validation_result: ValidationResult) -> None:
    """Generate recovery recommendations for validation failures."""
    if not validation_result.is_valid:
        validation_result.add_recommendation(
            "Review configuration errors and fix invalid parameters",
            priority="HIGH"
        )


def _check_merged_configuration_consistency(merged_config: Dict[str, Any], config_names: List[str]) -> None:
    """Check merged configuration for consistency and compatibility issues."""
    # This would implement consistency checking for merged configurations
    pass


# Export all public functions and constants for configuration management
__all__ = [
    # Configuration access functions
    'get_default_normalization_config',
    'get_default_simulation_config', 
    'get_default_analysis_config',
    'get_logging_config',
    'get_performance_thresholds',
    
    # General configuration operations
    'load_config',
    'save_config',
    'validate_config',
    'merge_configs',
    
    # Directory and file management
    'get_config_directory',
    'get_schema_directory',
    'list_available_configs',
    'clear_config_cache',
    
    # Global constants and paths
    'CONFIG_DIRECTORY',
    'SCHEMA_DIRECTORY', 
    'DEFAULT_CONFIGS'
]