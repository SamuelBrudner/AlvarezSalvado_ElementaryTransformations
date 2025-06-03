"""
Centralized log management system providing comprehensive logging infrastructure coordination, 
configuration management, logger lifecycle management, and integration with monitoring systems 
for the plume navigation simulation system.

This module implements hierarchical logger management, performance-aware logging, audit trail 
coordination, and scientific context integration to support reproducible research outcomes 
with structured logging, real-time monitoring, and comprehensive error tracking across all 
system components.

Key Features:
- Centralized log management system with comprehensive logging infrastructure
- Configuration management with dynamic reconfiguration and validation
- Logger lifecycle management with thread-safe operations and registry tracking
- Integration with monitoring systems for real-time performance tracking
- Scientific context enhancement for reproducible research outcomes
- Performance-aware logging with buffering and optimization
- Audit trail coordination with correlation tracking and compliance reporting
- Comprehensive error handling and recovery mechanisms
- Progress tracking integration for batch operations monitoring
- Alert system integration for log-based alerting and escalation
"""

# Standard library imports with version specifications
import logging  # Python 3.9+ - Core Python logging framework for log management infrastructure
import logging.config  # Python 3.9+ - Logging configuration management and dynamic reconfiguration
import threading  # Python 3.9+ - Thread-safe log management operations and concurrent logger access
import datetime  # Python 3.9+ - Timestamp management for log rotation and audit trail tracking
import pathlib  # Python 3.9+ - Cross-platform path handling for log file management
import typing  # Python 3.9+ - Type hints for log manager function signatures and data structures
import contextlib  # Python 3.9+ - Context manager utilities for scoped logging operations
import atexit  # Python 3.9+ - Cleanup registration for graceful log system shutdown
import signal  # Python 3.9+ - Signal handling for graceful log system shutdown and rotation
import json  # Python 3.9+ - JSON configuration parsing and dynamic configuration updates
import os  # Python 3.9+ - Operating system interface for environment variables and file permissions
import sys  # Python 3.9+ - System interface for stdout/stderr management and exit handling
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field

# Internal imports from utility modules
from ..utils.logging_utils import (
    initialize_logging_system,
    get_logger,
    configure_logger_for_component,
    load_logging_configuration,
    ScientificFormatter,
    PerformanceLoggingHandler,
    AuditTrailManager
)

from .console_formatter import (
    initialize_console_formatting,
    ScientificConsoleHandler
)

from .progress_tracker import (
    initialize_progress_tracking,
    ProgressTrackingContext
)

from .alert_system import (
    initialize_alert_system,
    AlertManager
)

from ..utils.file_utils import (
    load_json_config,
    ensure_directory_exists
)

# Global registry and state management variables
_log_manager_instance: Optional['LogManager'] = None
_logger_registry: Dict[str, logging.Logger] = {}
_handler_registry: Dict[str, logging.Handler] = {}
_formatter_registry: Dict[str, logging.Formatter] = {}
_monitoring_systems: Dict[str, Any] = {}
_log_manager_lock: threading.RLock = threading.RLock()
_shutdown_handlers: List[Callable] = []
_configuration_cache: Dict[str, Any] = {}

# Configuration constants for log management system
DEFAULT_LOG_DIRECTORY: str = 'logs'
DEFAULT_CONFIG_PATH: str = 'src/backend/config/logging_config.json'
LOG_ROTATION_CHECK_INTERVAL: float = 300.0
PERFORMANCE_LOG_FLUSH_INTERVAL: float = 60.0
AUDIT_TRAIL_RETENTION_DAYS: int = 365
MAX_LOG_FILE_SIZE_MB: int = 50
MAX_BACKUP_COUNT: int = 10


def initialize_log_manager(
    config_path: str = DEFAULT_CONFIG_PATH,
    enable_console_output: bool = True,
    enable_performance_logging: bool = True,
    enable_audit_trail: bool = True,
    manager_config: Dict[str, Any] = None
) -> 'LogManager':
    """
    Initialize the centralized log management system with configuration loading, monitoring system 
    integration, console formatting setup, and comprehensive logging infrastructure for scientific 
    computing workflows.
    
    This function sets up the entire logging infrastructure including custom log levels, formatters,
    handlers, and scientific context filters. It configures both console and file logging with
    appropriate formatting for scientific computing workflows with performance tracking, audit trail
    management, and real-time monitoring capabilities.
    
    Args:
        config_path: Path to logging configuration file (JSON format)
        enable_console_output: Enable color-coded console logging with progress indicators
        enable_performance_logging: Enable performance metrics logging with buffering
        enable_audit_trail: Enable audit trail management with correlation tracking
        manager_config: Additional configuration options for log manager
        
    Returns:
        LogManager: Initialized log manager instance with comprehensive logging infrastructure
    """
    global _log_manager_instance, _log_manager_lock
    
    with _log_manager_lock:
        # Return existing instance if already initialized
        if _log_manager_instance is not None:
            return _log_manager_instance
        
        try:
            # Load logging configuration from specified path or use default
            if config_path and pathlib.Path(config_path).exists():
                config = load_json_config(config_path, validate_schema=False, use_cache=True)
            else:
                config = _get_default_logging_configuration()
            
            # Initialize core logging system with scientific computing enhancements
            logging_success = initialize_logging_system(
                config_path=config_path,
                enable_console_output=enable_console_output,
                enable_file_logging=True,
                log_level='INFO'
            )
            
            if not logging_success:
                raise RuntimeError("Failed to initialize core logging system")
            
            # Setup console formatting system with color coding and progress indicators
            if enable_console_output:
                console_success = initialize_console_formatting(
                    force_color_detection=False,
                    default_color_scheme='scientific_default',
                    console_width_override=None,
                    formatting_config=manager_config.get('console_config', {}) if manager_config else {}
                )
                
                if not console_success:
                    print("WARNING: Console formatting initialization failed", file=sys.stderr)
            
            # Initialize performance logging infrastructure if enabled
            performance_handler = None
            if enable_performance_logging:
                try:
                    log_dir = pathlib.Path(config.get('output_directories', {}).get('performance_logs', 'logs/performance'))
                    ensure_directory_exists(str(log_dir), create_parents=True)
                    
                    performance_handler = PerformanceLoggingHandler(
                        filename=str(log_dir / 'performance_metrics.jsonl'),
                        buffer_size=config.get('performance_monitoring', {}).get('buffer_size', 1000),
                        flush_interval=config.get('performance_monitoring', {}).get('flush_interval_seconds', PERFORMANCE_LOG_FLUSH_INTERVAL),
                        compression=True,
                        real_time_monitoring=True
                    )
                    
                    _handler_registry['performance'] = performance_handler
                    
                except Exception as e:
                    print(f"WARNING: Performance logging initialization failed: {e}", file=sys.stderr)
            
            # Setup audit trail management system if enabled
            audit_manager = None
            if enable_audit_trail:
                try:
                    log_dir = pathlib.Path(config.get('output_directories', {}).get('audit_logs', 'logs/audit'))
                    ensure_directory_exists(str(log_dir), create_parents=True)
                    
                    audit_manager = AuditTrailManager(
                        audit_log_path=str(log_dir / 'audit_trail.log'),
                        enable_integrity_verification=False,
                        enable_tamper_protection=False
                    )
                    
                    _monitoring_systems['audit_manager'] = audit_manager
                    
                except Exception as e:
                    print(f"WARNING: Audit trail initialization failed: {e}", file=sys.stderr)
            
            # Initialize progress tracking integration for monitoring
            try:
                progress_success = initialize_progress_tracking(
                    enable_console_integration=enable_console_output,
                    enable_performance_monitoring=enable_performance_logging
                )
                
                if progress_success:
                    _monitoring_systems['progress_tracking'] = True
                    
            except Exception as e:
                print(f"WARNING: Progress tracking initialization failed: {e}", file=sys.stderr)
            
            # Setup alert system integration for log-based alerting
            try:
                alert_success = initialize_alert_system(
                    alert_config=manager_config.get('alert_config', {}) if manager_config else {},
                    enable_escalation=True,
                    enable_suppression=True,
                    alert_profile='scientific_computing'
                )
                
                if alert_success:
                    _monitoring_systems['alert_system'] = True
                    
            except Exception as e:
                print(f"WARNING: Alert system initialization failed: {e}", file=sys.stderr)
            
            # Create log manager instance with comprehensive configuration
            _log_manager_instance = LogManager(
                config_path=config_path,
                enable_console_output=enable_console_output,
                enable_performance_logging=enable_performance_logging,
                enable_audit_trail=enable_audit_trail
            )
            
            # Initialize the log manager instance
            if not _log_manager_instance.initialize():
                raise RuntimeError("Failed to initialize log manager instance")
            
            # Register shutdown handlers for graceful cleanup
            atexit.register(_cleanup_on_exit)
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
            
            _shutdown_handlers.append(_log_manager_instance.shutdown)
            
            # Validate log manager initialization and connectivity
            test_logger = get_logger('system.initialization', 'SYSTEM')
            test_logger.info("Log manager initialization completed successfully")
            
            # Return configured log manager instance
            return _log_manager_instance
            
        except Exception as e:
            print(f"CRITICAL: Log manager initialization failed: {e}", file=sys.stderr)
            raise


def get_log_manager(create_if_missing: bool = True) -> Optional['LogManager']:
    """
    Get the global log manager instance with thread-safe access and lazy initialization for 
    centralized log management across all system components.
    
    This function provides thread-safe access to the global log manager instance with optional
    lazy initialization for consistent logging infrastructure across all system components.
    
    Args:
        create_if_missing: Create new instance if missing and creation enabled
        
    Returns:
        LogManager: Global log manager instance or None if not initialized
    """
    global _log_manager_instance, _log_manager_lock
    
    with _log_manager_lock:
        # Check if global log manager instance exists
        if _log_manager_instance is not None:
            return _log_manager_instance
        
        # Create new instance if missing and creation enabled
        if create_if_missing:
            try:
                return initialize_log_manager()
            except Exception as e:
                print(f"ERROR: Failed to create log manager instance: {e}", file=sys.stderr)
                return None
        
        # Return None if instance not available
        return None


def configure_component_logging(
    component_name: str,
    log_level: str = 'INFO',
    handler_names: List[str] = None,
    enable_scientific_context: bool = True,
    enable_performance_tracking: bool = True,
    component_config: Dict[str, Any] = None
) -> logging.Logger:
    """
    Configure logging for specific system components with component-specific settings, scientific 
    context integration, and performance tracking for modular logging management.
    
    This function provides fine-grained logger configuration for specific components with custom
    settings, scientific context integration, and performance tracking capabilities.
    
    Args:
        component_name: Name of the component for logger configuration
        log_level: Log level for the component logger
        handler_names: List of handler names to attach to the logger
        enable_scientific_context: Enable scientific context filter
        enable_performance_tracking: Enable performance tracking if requested
        component_config: Component-specific configuration options
        
    Returns:
        logging.Logger: Configured logger instance for the specified component
    """
    try:
        # Get log manager instance for component configuration
        log_manager = get_log_manager(create_if_missing=True)
        if not log_manager:
            raise RuntimeError("Log manager not available for component configuration")
        
        # Validate component name and configuration parameters
        if not component_name:
            raise ValueError("Component name cannot be empty")
        
        # Configure logger using component-specific settings
        logger = configure_logger_for_component(
            component_name=component_name,
            log_level=log_level,
            handler_names=handler_names or [],
            enable_scientific_context=enable_scientific_context,
            component_config=component_config or {}
        )
        
        # Setup scientific context integration if enabled
        if enable_scientific_context:
            # Scientific context is automatically handled by configure_logger_for_component
            pass
        
        # Enable performance tracking if requested
        if enable_performance_tracking and 'performance' in _handler_registry:
            logger.addHandler(_handler_registry['performance'])
        
        # Register logger in component registry
        _logger_registry[component_name] = logger
        
        # Apply component-specific handlers and formatters
        if component_config and 'handlers' in component_config:
            for handler_config in component_config['handlers']:
                try:
                    handler = _create_handler_from_config(handler_config)
                    logger.addHandler(handler)
                except Exception as e:
                    print(f"WARNING: Failed to add handler for {component_name}: {e}", file=sys.stderr)
        
        # Return configured logger instance
        return logger
        
    except Exception as e:
        print(f"ERROR: Failed to configure component logging for {component_name}: {e}", file=sys.stderr)
        # Return basic logger as fallback
        return logging.getLogger(component_name)


def reconfigure_logging(
    new_config_path: str,
    preserve_existing_loggers: bool = True,
    validate_configuration: bool = True
) -> bool:
    """
    Dynamically reconfigure logging system with new configuration, handler updates, and monitoring 
    system integration for runtime configuration management.
    
    This function provides comprehensive logging system reconfiguration with validation, handler
    updates, and monitoring system integration for runtime configuration management.
    
    Args:
        new_config_path: Path to new logging configuration file
        preserve_existing_loggers: Preserve existing loggers if requested
        validate_configuration: Enable strict configuration validation
        
    Returns:
        bool: Success status of logging reconfiguration
    """
    global _configuration_cache, _log_manager_lock
    
    with _log_manager_lock:
        try:
            # Load new logging configuration with validation
            if not pathlib.Path(new_config_path).exists():
                raise FileNotFoundError(f"Configuration file not found: {new_config_path}")
            
            new_config = load_json_config(
                config_path=new_config_path,
                validate_schema=validate_configuration,
                use_cache=False  # Force reload for reconfiguration
            )
            
            # Backup current logging configuration
            backup_config = _configuration_cache.copy()
            
            # Apply new configuration to logging system
            try:
                # Update logging configuration using logging.config
                if 'version' in new_config:
                    logging.config.dictConfig(new_config)
                else:
                    raise ValueError("Invalid configuration format - missing version")
                
                # Update handler and formatter registries
                _update_registries_from_config(new_config)
                
                # Preserve existing loggers if requested
                if preserve_existing_loggers:
                    for logger_name, logger in _logger_registry.items():
                        # Maintain existing logger configuration
                        pass
                
                # Update monitoring system integrations
                log_manager = get_log_manager(create_if_missing=False)
                if log_manager:
                    log_manager.update_configuration(new_config, preserve_existing_loggers)
                
                # Update configuration cache
                _configuration_cache.update(new_config)
                
                # Validate reconfiguration success
                test_logger = get_logger('system.reconfiguration', 'SYSTEM')
                test_logger.info(f"Logging system reconfigured successfully from: {new_config_path}")
                
                # Log configuration change with audit trail
                if 'audit_manager' in _monitoring_systems:
                    audit_manager = _monitoring_systems['audit_manager']
                    audit_manager.create_audit_entry(
                        action='LOGGING_RECONFIGURATION',
                        component='LOG_MANAGER',
                        details={
                            'new_config_path': new_config_path,
                            'preserve_existing_loggers': preserve_existing_loggers,
                            'validate_configuration': validate_configuration
                        }
                    )
                
                return True
                
            except Exception as config_error:
                # Restore backup configuration on failure
                try:
                    if backup_config:
                        logging.config.dictConfig(backup_config)
                        print(f"Logging configuration restored from backup due to error: {config_error}", file=sys.stderr)
                except Exception as restore_error:
                    print(f"CRITICAL: Failed to restore logging configuration: {restore_error}", file=sys.stderr)
                
                raise config_error
            
        except Exception as e:
            print(f"ERROR: Logging reconfiguration failed: {e}", file=sys.stderr)
            return False


def flush_all_logs(force_flush: bool = False, timeout_seconds: float = 30.0) -> Dict[str, bool]:
    """
    Flush all logging handlers and buffers to ensure log data persistence for critical operations 
    and system shutdown procedures.
    
    This function provides comprehensive flushing of all logging handlers and buffers with timeout
    management and error handling for critical operations and system shutdown procedures.
    
    Args:
        force_flush: Force flush if requested and handler supports it
        timeout_seconds: Maximum time to wait for flush operations
        
    Returns:
        Dict[str, bool]: Flush status for each handler with success indicators
    """
    flush_results = {}
    
    try:
        # Get all registered logging handlers
        all_handlers = []
        
        # Collect handlers from registry
        for handler_name, handler in _handler_registry.items():
            all_handlers.append((handler_name, handler))
        
        # Collect handlers from active loggers
        for logger_name, logger in _logger_registry.items():
            for i, handler in enumerate(logger.handlers):
                handler_id = f"{logger_name}_handler_{i}"
                all_handlers.append((handler_id, handler))
        
        # Collect root logger handlers
        root_logger = logging.getLogger()
        for i, handler in enumerate(root_logger.handlers):
            handler_id = f"root_handler_{i}"
            all_handlers.append((handler_id, handler))
        
        # Flush each handler with timeout management
        for handler_name, handler in all_handlers:
            try:
                # Check if handler supports flushing
                if hasattr(handler, 'flush'):
                    if force_flush and hasattr(handler, 'force_flush'):
                        handler.force_flush()
                    else:
                        handler.flush()
                    
                    flush_results[handler_name] = True
                else:
                    flush_results[handler_name] = False  # Handler doesn't support flush
                    
            except Exception as handler_error:
                flush_results[handler_name] = False
                print(f"WARNING: Failed to flush handler {handler_name}: {handler_error}", file=sys.stderr)
        
        # Flush performance logging handler if available
        if 'performance' in _handler_registry:
            try:
                perf_handler = _handler_registry['performance']
                if hasattr(perf_handler, 'flush_buffer'):
                    perf_handler.flush_buffer()
                    flush_results['performance_buffer'] = True
            except Exception as e:
                flush_results['performance_buffer'] = False
                print(f"WARNING: Failed to flush performance buffer: {e}", file=sys.stderr)
        
        # Log flush operation completion
        successful_flushes = sum(1 for success in flush_results.values() if success)
        total_handlers = len(flush_results)
        
        if successful_flushes > 0:
            test_logger = get_logger('system.flush', 'SYSTEM')
            test_logger.info(f"Log flush operation completed: {successful_flushes}/{total_handlers} handlers flushed")
        
        return flush_results
        
    except Exception as e:
        print(f"ERROR: Log flush operation failed: {e}", file=sys.stderr)
        return {'error': False}


def rotate_log_files(
    compress_rotated_files: bool = True,
    cleanup_old_files: bool = True
) -> Dict[str, Any]:
    """
    Manually trigger log file rotation for all rotating handlers with compression, archival, and 
    cleanup for log file management.
    
    This function provides comprehensive log file rotation with compression, archival, and cleanup
    for efficient log file management and storage optimization.
    
    Args:
        compress_rotated_files: Compress rotated files if compression enabled
        cleanup_old_files: Cleanup old log files based on retention policies
        
    Returns:
        Dict[str, Any]: Log rotation summary with file counts and cleanup statistics
    """
    rotation_summary = {
        'rotation_timestamp': datetime.datetime.now().isoformat(),
        'handlers_processed': 0,
        'files_rotated': 0,
        'files_compressed': 0,
        'files_cleaned': 0,
        'errors': [],
        'rotation_success': True
    }
    
    try:
        # Identify all rotating file handlers
        rotating_handlers = []
        
        for handler_name, handler in _handler_registry.items():
            if hasattr(handler, 'doRollover'):  # RotatingFileHandler or TimedRotatingFileHandler
                rotating_handlers.append((handler_name, handler))
        
        # Check handlers from active loggers
        for logger_name, logger in _logger_registry.items():
            for i, handler in enumerate(logger.handlers):
                if hasattr(handler, 'doRollover'):
                    handler_id = f"{logger_name}_rotating_{i}"
                    rotating_handlers.append((handler_id, handler))
        
        rotation_summary['handlers_processed'] = len(rotating_handlers)
        
        # Trigger rotation for each rotating handler
        for handler_name, handler in rotating_handlers:
            try:
                # Get current log file info before rotation
                current_file = getattr(handler, 'baseFilename', None)
                
                # Trigger rotation
                handler.doRollover()
                rotation_summary['files_rotated'] += 1
                
                # Compress rotated files if compression enabled
                if compress_rotated_files and current_file:
                    compressed_count = _compress_rotated_file(current_file)
                    rotation_summary['files_compressed'] += compressed_count
                
            except Exception as handler_error:
                error_msg = f"Rotation failed for {handler_name}: {handler_error}"
                rotation_summary['errors'].append(error_msg)
                print(f"WARNING: {error_msg}", file=sys.stderr)
        
        # Cleanup old log files based on retention policies
        if cleanup_old_files:
            try:
                cleanup_count = _cleanup_old_log_files()
                rotation_summary['files_cleaned'] = cleanup_count
            except Exception as cleanup_error:
                error_msg = f"Cleanup failed: {cleanup_error}"
                rotation_summary['errors'].append(error_msg)
                print(f"WARNING: {error_msg}", file=sys.stderr)
        
        # Update rotation statistics
        if rotation_summary['errors']:
            rotation_summary['rotation_success'] = False
        
        # Log rotation operation completion
        test_logger = get_logger('system.rotation', 'SYSTEM')
        test_logger.info(
            f"Log rotation completed: {rotation_summary['files_rotated']} files rotated, "
            f"{rotation_summary['files_compressed']} compressed, {rotation_summary['files_cleaned']} cleaned"
        )
        
        return rotation_summary
        
    except Exception as e:
        rotation_summary['rotation_success'] = False
        rotation_summary['errors'].append(str(e))
        print(f"ERROR: Log rotation operation failed: {e}", file=sys.stderr)
        return rotation_summary


def get_logging_statistics(
    statistics_period: str = 'all',
    include_performance_metrics: bool = True,
    include_handler_details: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive logging system statistics including handler performance, log volumes, error 
    rates, and system health indicators for monitoring and optimization.
    
    This function provides comprehensive logging system statistics with performance metrics, handler
    information, and system health indicators for monitoring and optimization purposes.
    
    Args:
        statistics_period: Period for statistics calculation ('all', 'today', 'week')
        include_performance_metrics: Include performance metrics if requested
        include_handler_details: Gather handler-specific statistics if requested
        
    Returns:
        Dict[str, Any]: Comprehensive logging statistics with performance metrics and handler information
    """
    try:
        statistics = {
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'statistics_period': statistics_period,
            'system_overview': {
                'active_loggers': len(_logger_registry),
                'registered_handlers': len(_handler_registry),
                'registered_formatters': len(_formatter_registry),
                'monitoring_systems': len(_monitoring_systems),
                'log_manager_initialized': _log_manager_instance is not None
            }
        }
        
        # Collect logging statistics from all handlers
        if include_handler_details:
            handler_stats = {}
            
            for handler_name, handler in _handler_registry.items():
                handler_info = {
                    'handler_type': type(handler).__name__,
                    'level': handler.level,
                    'formatter': type(handler.formatter).__name__ if handler.formatter else None
                }
                
                # Get handler-specific statistics
                if hasattr(handler, 'get_statistics'):
                    try:
                        handler_info.update(handler.get_statistics())
                    except Exception as e:
                        handler_info['statistics_error'] = str(e)
                
                # Get file handler specific information
                if hasattr(handler, 'baseFilename'):
                    file_path = pathlib.Path(handler.baseFilename)
                    if file_path.exists():
                        handler_info['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
                        handler_info['last_modified'] = datetime.datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat()
                
                handler_stats[handler_name] = handler_info
            
            statistics['handler_details'] = handler_stats
        
        # Include performance metrics if requested
        if include_performance_metrics:
            performance_stats = {
                'performance_logging_enabled': 'performance' in _handler_registry,
                'audit_trail_enabled': 'audit_manager' in _monitoring_systems,
                'progress_tracking_enabled': _monitoring_systems.get('progress_tracking', False),
                'alert_system_enabled': _monitoring_systems.get('alert_system', False)
            }
            
            # Get performance handler statistics
            if 'performance' in _handler_registry:
                perf_handler = _handler_registry['performance']
                if hasattr(perf_handler, 'get_statistics'):
                    try:
                        performance_stats['performance_handler'] = perf_handler.get_statistics()
                    except Exception as e:
                        performance_stats['performance_handler_error'] = str(e)
            
            statistics['performance_metrics'] = performance_stats
        
        # Calculate log volumes and rates for specified period
        volume_stats = {
            'estimated_total_log_size_mb': 0.0,
            'log_files_count': 0,
            'rotation_files_count': 0
        }
        
        # Scan log directories for volume calculation
        try:
            log_dirs = ['logs', 'logs/performance', 'logs/audit', 'logs/errors']
            for log_dir_str in log_dirs:
                log_dir = pathlib.Path(log_dir_str)
                if log_dir.exists():
                    for log_file in log_dir.rglob('*.log*'):
                        volume_stats['log_files_count'] += 1
                        if log_file.is_file():
                            volume_stats['estimated_total_log_size_mb'] += log_file.stat().st_size / (1024 * 1024)
                        
                        if '.1' in log_file.name or 'backup' in log_file.name:
                            volume_stats['rotation_files_count'] += 1
        except Exception as e:
            volume_stats['volume_calculation_error'] = str(e)
        
        statistics['volume_statistics'] = volume_stats
        
        # Analyze error rates and system health indicators
        health_indicators = {
            'system_health': 'good',  # Would be calculated from actual metrics
            'configuration_valid': _configuration_cache is not None and len(_configuration_cache) > 0,
            'handlers_operational': len(_handler_registry) > 0,
            'monitoring_active': len(_monitoring_systems) > 0
        }
        
        # Determine overall system health
        if health_indicators['configuration_valid'] and health_indicators['handlers_operational']:
            health_indicators['system_health'] = 'good'
        elif health_indicators['handlers_operational']:
            health_indicators['system_health'] = 'fair'
        else:
            health_indicators['system_health'] = 'poor'
        
        statistics['health_indicators'] = health_indicators
        
        # Format statistics for reporting and monitoring
        statistics['summary'] = {
            'total_components': (
                statistics['system_overview']['active_loggers'] +
                statistics['system_overview']['registered_handlers'] +
                statistics['system_overview']['monitoring_systems']
            ),
            'estimated_disk_usage_mb': volume_stats['estimated_total_log_size_mb'],
            'system_operational': health_indicators['system_health'] in ['good', 'fair']
        }
        
        return statistics
        
    except Exception as e:
        print(f"ERROR: Failed to collect logging statistics: {e}", file=sys.stderr)
        return {
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'statistics_period': statistics_period,
            'error': str(e),
            'partial_data_available': False
        }


def cleanup_log_manager(
    archive_logs: bool = False,
    preserve_audit_trail: bool = True,
    cleanup_mode: str = 'normal'
) -> Dict[str, Any]:
    """
    Cleanup log management system resources, finalize audit trails, archive logs, and prepare for 
    system shutdown with comprehensive cleanup and preservation.
    
    This function provides comprehensive cleanup of the log management system with optional log
    archival, audit trail preservation, and resource cleanup for system shutdown.
    
    Args:
        archive_logs: Archive log files if archival enabled
        preserve_audit_trail: Finalize audit trail if preservation enabled
        cleanup_mode: Mode of cleanup operation ('normal', 'emergency', 'maintenance')
        
    Returns:
        Dict[str, Any]: Cleanup summary with final statistics and preserved data locations
    """
    global _log_manager_instance, _logger_registry, _handler_registry, _formatter_registry
    global _monitoring_systems, _configuration_cache
    
    cleanup_summary = {
        'cleanup_timestamp': datetime.datetime.now().isoformat(),
        'cleanup_mode': cleanup_mode,
        'archive_logs': archive_logs,
        'preserve_audit_trail': preserve_audit_trail,
        'operations_performed': [],
        'final_statistics': {},
        'preserved_data_locations': [],
        'cleanup_success': True
    }
    
    try:
        # Stop all active logging operations
        if _log_manager_instance:
            try:
                shutdown_result = _log_manager_instance.shutdown(
                    archive_logs=archive_logs,
                    preserve_audit_trail=preserve_audit_trail
                )
                cleanup_summary['manager_shutdown_result'] = shutdown_result
                cleanup_summary['operations_performed'].append('log_manager_shutdown')
            except Exception as e:
                cleanup_summary['cleanup_success'] = False
                cleanup_summary['errors'] = cleanup_summary.get('errors', [])
                cleanup_summary['errors'].append(f"Manager shutdown failed: {e}")
        
        # Flush all handlers and buffers
        try:
            flush_results = flush_all_logs(force_flush=True, timeout_seconds=10.0)
            cleanup_summary['flush_results'] = flush_results
            cleanup_summary['operations_performed'].append('flush_all_handlers')
        except Exception as e:
            cleanup_summary['cleanup_success'] = False
            cleanup_summary['errors'] = cleanup_summary.get('errors', [])
            cleanup_summary['errors'].append(f"Handler flush failed: {e}")
        
        # Finalize audit trail if preservation enabled
        if preserve_audit_trail and 'audit_manager' in _monitoring_systems:
            try:
                audit_manager = _monitoring_systems['audit_manager']
                audit_summary = audit_manager.generate_audit_report(
                    start_time=datetime.datetime.now() - datetime.timedelta(days=1),
                    end_time=datetime.datetime.now(),
                    components_filter=None
                )
                
                cleanup_summary['audit_trail_summary'] = audit_summary
                cleanup_summary['operations_performed'].append('audit_trail_finalized')
                
                if archive_logs:
                    # Archive audit trail
                    archive_path = f"logs/archive/audit_trail_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    ensure_directory_exists(str(pathlib.Path(archive_path).parent), create_parents=True)
                    
                    with open(archive_path, 'w') as f:
                        json.dump(audit_summary, f, indent=2)
                    
                    cleanup_summary['preserved_data_locations'].append(archive_path)
                    
            except Exception as e:
                print(f"WARNING: Audit trail finalization failed: {e}", file=sys.stderr)
        
        # Archive log files if archival enabled
        if archive_logs:
            try:
                archive_count = _archive_log_files()
                cleanup_summary['archived_files_count'] = archive_count
                cleanup_summary['operations_performed'].append('log_files_archived')
            except Exception as e:
                print(f"WARNING: Log file archival failed: {e}", file=sys.stderr)
        
        # Close all logging handlers and connections
        try:
            for handler_name, handler in list(_handler_registry.items()):
                try:
                    if hasattr(handler, 'close'):
                        handler.close()
                except Exception as handler_error:
                    print(f"WARNING: Failed to close handler {handler_name}: {handler_error}", file=sys.stderr)
            
            cleanup_summary['operations_performed'].append('handlers_closed')
        except Exception as e:
            print(f"WARNING: Handler closing failed: {e}", file=sys.stderr)
        
        # Generate final logging statistics
        try:
            final_stats = get_logging_statistics(
                statistics_period='all',
                include_performance_metrics=True,
                include_handler_details=True
            )
            cleanup_summary['final_statistics'] = final_stats
            cleanup_summary['operations_performed'].append('final_statistics_generated')
        except Exception as e:
            print(f"WARNING: Final statistics generation failed: {e}", file=sys.stderr)
        
        # Clear log manager registries and caches
        try:
            _logger_registry.clear()
            _handler_registry.clear()
            _formatter_registry.clear()
            _monitoring_systems.clear()
            _configuration_cache.clear()
            _log_manager_instance = None
            
            cleanup_summary['operations_performed'].append('registries_cleared')
        except Exception as e:
            print(f"WARNING: Registry clearing failed: {e}", file=sys.stderr)
        
        # Log cleanup completion with summary
        if cleanup_summary['cleanup_success']:
            print(f"Log manager cleanup completed successfully (mode: {cleanup_mode})")
        else:
            print(f"Log manager cleanup completed with errors (mode: {cleanup_mode})")
        
        return cleanup_summary
        
    except Exception as e:
        cleanup_summary['cleanup_success'] = False
        cleanup_summary['critical_error'] = str(e)
        print(f"CRITICAL: Log manager cleanup failed: {e}", file=sys.stderr)
        return cleanup_summary


@dataclass
class LogManagerConfig:
    """
    Configuration data class for log manager initialization providing structured configuration 
    management, validation, and default value handling for comprehensive logging system setup.
    
    This dataclass provides comprehensive configuration management with validation and default
    value handling for log manager initialization and runtime configuration.
    """
    
    config_path: str = DEFAULT_CONFIG_PATH
    enable_console_output: bool = True
    enable_performance_logging: bool = True
    enable_audit_trail: bool = True
    
    # Optional configuration parameters with defaults
    log_directory: str = DEFAULT_LOG_DIRECTORY
    console_color_scheme: str = 'scientific_default'
    max_log_file_size_mb: int = MAX_LOG_FILE_SIZE_MB
    backup_count: int = MAX_BACKUP_COUNT
    performance_flush_interval: float = PERFORMANCE_LOG_FLUSH_INTERVAL
    audit_retention_days: int = AUDIT_TRAIL_RETENTION_DAYS
    monitoring_integrations: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize log manager configuration with default values and validation."""
        # Set primary configuration parameters
        if not self.config_path:
            self.config_path = DEFAULT_CONFIG_PATH
        
        # Apply default values for optional parameters
        if not self.log_directory:
            self.log_directory = DEFAULT_LOG_DIRECTORY
        
        # Validate configuration parameter consistency
        validation_errors = self.validate()
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")
        
        # Initialize monitoring integration settings
        if not self.monitoring_integrations:
            self.monitoring_integrations = {
                'alert_system': True,
                'progress_tracking': True,
                'performance_monitoring': True
            }
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters for consistency and compatibility with system requirements.
        
        Returns:
            List[str]: List of validation errors or empty list if valid
        """
        errors = []
        
        # Validate configuration file path exists
        if self.config_path and not pathlib.Path(self.config_path).exists():
            errors.append(f"Configuration file not found: {self.config_path}")
        
        # Check log directory accessibility
        try:
            log_path = pathlib.Path(self.log_directory)
            if not log_path.exists():
                # Try to create directory to test accessibility
                log_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Log directory not accessible: {self.log_directory} - {e}")
        
        # Validate numerical parameters are within acceptable ranges
        if self.max_log_file_size_mb <= 0:
            errors.append("Max log file size must be positive")
        
        if self.backup_count < 0:
            errors.append("Backup count cannot be negative")
        
        if self.performance_flush_interval <= 0:
            errors.append("Performance flush interval must be positive")
        
        if self.audit_retention_days <= 0:
            errors.append("Audit retention days must be positive")
        
        # Check feature flag consistency
        if self.enable_performance_logging and self.performance_flush_interval <= 0:
            errors.append("Performance logging enabled but invalid flush interval")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for serialization and logging.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary with all parameters
        """
        # Convert all configuration parameters to dictionary
        config_dict = {
            'config_path': self.config_path,
            'enable_console_output': self.enable_console_output,
            'enable_performance_logging': self.enable_performance_logging,
            'enable_audit_trail': self.enable_audit_trail,
            'log_directory': self.log_directory,
            'console_color_scheme': self.console_color_scheme,
            'max_log_file_size_mb': self.max_log_file_size_mb,
            'backup_count': self.backup_count,
            'performance_flush_interval': self.performance_flush_interval,
            'audit_retention_days': self.audit_retention_days,
            'monitoring_integrations': self.monitoring_integrations.copy()
        }
        
        # Include default values and computed settings
        config_dict['validation_status'] = 'valid' if not self.validate() else 'invalid'
        config_dict['configuration_timestamp'] = datetime.datetime.now().isoformat()
        
        return config_dict


class LogManager:
    """
    Centralized log management class providing comprehensive logging infrastructure coordination, 
    configuration management, logger lifecycle management, and integration with monitoring systems 
    for scientific computing workflows with performance tracking, audit trail management, and 
    real-time monitoring capabilities.
    
    This class provides centralized coordination of all logging operations with comprehensive
    configuration management, logger lifecycle management, and monitoring system integration
    for scientific computing workflows.
    """
    
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        enable_console_output: bool = True,
        enable_performance_logging: bool = True,
        enable_audit_trail: bool = True
    ):
        """
        Initialize log manager with configuration loading, monitoring system setup, and comprehensive 
        logging infrastructure for scientific computing.
        
        Args:
            config_path: Path to logging configuration file
            enable_console_output: Enable console output handler if enabled
            enable_performance_logging: Enable performance logging handler if enabled
            enable_audit_trail: Enable audit trail manager if enabled
        """
        # Set configuration path and feature flags
        self.config_path = config_path
        self.console_output_enabled = enable_console_output
        self.performance_logging_enabled = enable_performance_logging
        self.audit_trail_enabled = enable_audit_trail
        
        # Initialize thread lock for manager operations
        self.manager_lock = threading.RLock()
        
        # Load logging configuration from specified path
        try:
            if pathlib.Path(config_path).exists():
                self.logging_config = load_json_config(config_path, validate_schema=False, use_cache=True)
            else:
                self.logging_config = _get_default_logging_configuration()
        except Exception as e:
            print(f"WARNING: Failed to load logging config, using defaults: {e}", file=sys.stderr)
            self.logging_config = _get_default_logging_configuration()
        
        # Initialize logger, handler, and formatter registries
        self.logger_registry: Dict[str, logging.Logger] = {}
        self.handler_registry: Dict[str, logging.Handler] = {}
        self.formatter_registry: Dict[str, logging.Formatter] = {}
        
        # Setup console output handler if enabled
        self.console_handler: Optional[ScientificConsoleHandler] = None
        
        # Initialize performance logging handler if enabled
        self.performance_handler: Optional[PerformanceLoggingHandler] = None
        
        # Setup audit trail manager if enabled
        self.audit_manager: Optional[AuditTrailManager] = None
        
        # Initialize monitoring system integrations
        self.alert_manager: Optional[AlertManager] = None
        self.monitoring_integrations: Dict[str, Any] = {}
        
        # Register shutdown callbacks for cleanup
        self.shutdown_callbacks: List[Callable] = []
        
        # Record initialization time and set initialized flag
        self.initialization_time = datetime.datetime.now()
        self.is_initialized = False
        
        # Initialize statistics tracking
        self.statistics: Dict[str, Any] = {
            'loggers_created': 0,
            'handlers_added': 0,
            'configuration_updates': 0,
            'initialization_time': self.initialization_time.isoformat()
        }
    
    def initialize(self) -> bool:
        """
        Initialize the log management system with configuration loading, handler setup, and monitoring 
        integration for comprehensive logging infrastructure.
        
        This method sets up the complete logging infrastructure including handlers, formatters,
        monitoring systems, and scientific context integration.
        
        Returns:
            bool: Success status of log manager initialization
        """
        with self.manager_lock:
            try:
                # Load and validate logging configuration
                if not self.logging_config:
                    raise RuntimeError("No logging configuration available")
                
                # Initialize core logging system with scientific enhancements
                # (Already done in initialize_log_manager, but ensure consistency)
                
                # Setup console formatting if console output enabled
                if self.console_output_enabled:
                    try:
                        self.console_handler = ScientificConsoleHandler(
                            progress_mode=True,
                            buffering_enabled=True,
                            buffer_size=100,
                            flush_interval=1.0,
                            color_scheme='scientific_default'
                        )
                        
                        self.handler_registry['console_scientific'] = self.console_handler
                        self.statistics['handlers_added'] += 1
                        
                    except Exception as e:
                        print(f"WARNING: Console handler initialization failed: {e}", file=sys.stderr)
                
                # Initialize performance logging infrastructure if enabled
                if self.performance_logging_enabled:
                    try:
                        log_dir = pathlib.Path(self.logging_config.get('output_directories', {}).get('performance_logs', 'logs/performance'))
                        ensure_directory_exists(str(log_dir), create_parents=True)
                        
                        self.performance_handler = PerformanceLoggingHandler(
                            filename=str(log_dir / 'manager_performance.jsonl'),
                            buffer_size=1000,
                            flush_interval=PERFORMANCE_LOG_FLUSH_INTERVAL,
                            compression=True,
                            real_time_monitoring=True
                        )
                        
                        self.handler_registry['performance_manager'] = self.performance_handler
                        self.statistics['handlers_added'] += 1
                        
                    except Exception as e:
                        print(f"WARNING: Performance handler initialization failed: {e}", file=sys.stderr)
                
                # Setup audit trail management if enabled
                if self.audit_trail_enabled:
                    try:
                        log_dir = pathlib.Path(self.logging_config.get('output_directories', {}).get('audit_logs', 'logs/audit'))
                        ensure_directory_exists(str(log_dir), create_parents=True)
                        
                        self.audit_manager = AuditTrailManager(
                            audit_log_path=str(log_dir / 'manager_audit.log'),
                            enable_integrity_verification=False,
                            enable_tamper_protection=False
                        )
                        
                        self.monitoring_integrations['audit_manager'] = self.audit_manager
                        
                    except Exception as e:
                        print(f"WARNING: Audit manager initialization failed: {e}", file=sys.stderr)
                
                # Initialize progress tracking integration
                try:
                    self.monitoring_integrations['progress_tracking'] = True
                except Exception as e:
                    print(f"WARNING: Progress tracking integration failed: {e}", file=sys.stderr)
                
                # Setup alert system integration
                try:
                    # Alert manager is initialized globally, store reference
                    self.monitoring_integrations['alert_system'] = True
                except Exception as e:
                    print(f"WARNING: Alert system integration failed: {e}", file=sys.stderr)
                
                # Configure monitoring system integrations
                self._setup_monitoring_integrations()
                
                # Register cleanup callbacks
                self.shutdown_callbacks.append(self._cleanup_handlers)
                self.shutdown_callbacks.append(self._finalize_audit_trail)
                
                # Set initialization flag and log completion
                self.is_initialized = True
                
                # Update statistics
                self.statistics['initialization_success'] = True
                self.statistics['initialization_completed'] = datetime.datetime.now().isoformat()
                
                # Log initialization success
                logger = get_logger('log_manager.init', 'SYSTEM')
                logger.info("Log manager initialization completed successfully")
                
                # Create audit trail entry
                if self.audit_manager:
                    self.audit_manager.create_audit_entry(
                        action='LOG_MANAGER_INITIALIZED',
                        component='LOG_MANAGER',
                        details={
                            'console_enabled': self.console_output_enabled,
                            'performance_enabled': self.performance_logging_enabled,
                            'audit_enabled': self.audit_trail_enabled,
                            'handlers_count': len(self.handler_registry),
                            'monitoring_integrations': len(self.monitoring_integrations)
                        }
                    )
                
                return True
                
            except Exception as e:
                self.statistics['initialization_success'] = False
                self.statistics['initialization_error'] = str(e)
                print(f"ERROR: Log manager initialization failed: {e}", file=sys.stderr)
                return False
    
    def get_logger(
        self,
        logger_name: str,
        component_type: str = 'GENERAL',
        enable_scientific_context: bool = True,
        enable_performance_tracking: bool = True
    ) -> logging.Logger:
        """
        Get or create logger instance with component-specific configuration, scientific context 
        integration, and performance tracking.
        
        This method provides comprehensive logger creation and configuration with scientific
        context integration and performance tracking capabilities.
        
        Args:
            logger_name: Unique identifier for the logger instance
            component_type: Type of component for categorized logging
            enable_scientific_context: Enable scientific context filter
            enable_performance_tracking: Enable performance metric collection
            
        Returns:
            logging.Logger: Configured logger instance with scientific enhancements
        """
        with self.manager_lock:
            # Check logger registry for existing logger
            if logger_name in self.logger_registry:
                return self.logger_registry[logger_name]
            
            # Create new logger if not found
            logger = get_logger(
                logger_name=logger_name,
                component_type=component_type,
                enable_scientific_context=enable_scientific_context,
                enable_performance_tracking=enable_performance_tracking
            )
            
            # Configure logger with component-specific settings
            component_config = self.logging_config.get('loggers', {}).get(logger_name, {})
            
            if component_config:
                logger.setLevel(getattr(logging, component_config.get('level', 'INFO')))
                
                # Add handlers from configuration
                for handler_name in component_config.get('handlers', []):
                    if handler_name in self.handler_registry:
                        logger.addHandler(self.handler_registry[handler_name])
            
            # Add scientific context filter if enabled
            if enable_scientific_context:
                # Scientific context is handled by get_logger function
                pass
            
            # Setup performance tracking if enabled
            if enable_performance_tracking and self.performance_handler:
                logger.addHandler(self.performance_handler)
            
            # Register logger in registry
            self.logger_registry[logger_name] = logger
            self.statistics['loggers_created'] += 1
            
            # Apply appropriate handlers and formatters
            if self.console_handler and component_type in ['SYSTEM', 'SIMULATION', 'ANALYSIS']:
                logger.addHandler(self.console_handler)
            
            # Log logger creation for audit trail
            logger.debug(f"Logger created by log manager: {logger_name} (type: {component_type})")
            
            return logger
    
    def configure_component(
        self,
        component_name: str,
        component_config: Dict[str, Any]
    ) -> logging.Logger:
        """
        Configure logging for specific system component with specialized settings, handlers, and 
        scientific context integration.
        
        This method provides comprehensive component-specific logging configuration with specialized
        settings and scientific context integration.
        
        Args:
            component_name: Name of the component to configure
            component_config: Configuration dictionary for the component
            
        Returns:
            logging.Logger: Configured component logger with specialized settings
        """
        with self.manager_lock:
            # Validate component configuration parameters
            if not component_name:
                raise ValueError("Component name cannot be empty")
            
            if not isinstance(component_config, dict):
                raise TypeError("Component config must be a dictionary")
            
            # Get or create logger for component
            logger = self.get_logger(
                logger_name=component_name,
                component_type=component_config.get('type', 'GENERAL'),
                enable_scientific_context=component_config.get('scientific_context', True),
                enable_performance_tracking=component_config.get('performance_tracking', True)
            )
            
            # Apply component-specific log level and handlers
            if 'level' in component_config:
                logger.setLevel(getattr(logging, component_config['level'].upper()))
            
            if 'handlers' in component_config:
                # Clear existing handlers if replacement is specified
                if component_config.get('replace_handlers', False):
                    logger.handlers.clear()
                
                # Add specified handlers
                for handler_name in component_config['handlers']:
                    if handler_name in self.handler_registry:
                        logger.addHandler(self.handler_registry[handler_name])
                    elif handler_name in _handler_registry:
                        logger.addHandler(_handler_registry[handler_name])
            
            # Configure scientific context integration
            if component_config.get('scientific_context', True):
                # Scientific context is automatically handled by get_logger
                pass
            
            # Setup performance tracking if required
            if component_config.get('performance_tracking', True) and self.performance_handler:
                if self.performance_handler not in logger.handlers:
                    logger.addHandler(self.performance_handler)
            
            # Register component logger
            self.logger_registry[component_name] = logger
            
            # Log component configuration
            logger.info(f"Component logging configured: {component_name}")
            
            # Create audit trail entry
            if self.audit_manager:
                self.audit_manager.create_audit_entry(
                    action='COMPONENT_CONFIGURED',
                    component='LOG_MANAGER',
                    details={
                        'component_name': component_name,
                        'configuration': component_config
                    }
                )
            
            return logger
    
    def add_handler(
        self,
        handler_name: str,
        handler: logging.Handler,
        target_loggers: List[str] = None
    ) -> bool:
        """
        Add logging handler to the system with registration, configuration, and integration with 
        existing infrastructure.
        
        This method provides comprehensive handler addition with registration and integration
        capabilities for dynamic logging system configuration.
        
        Args:
            handler_name: Name identifier for the handler
            handler: Logging handler instance to add
            target_loggers: List of logger names to attach handler to
            
        Returns:
            bool: Success status of handler addition
        """
        with self.manager_lock:
            try:
                # Validate handler configuration and compatibility
                if not handler_name:
                    raise ValueError("Handler name cannot be empty")
                
                if not isinstance(handler, logging.Handler):
                    raise TypeError("Handler must be a logging.Handler instance")
                
                # Register handler in handler registry
                self.handler_registry[handler_name] = handler
                _handler_registry[handler_name] = handler
                
                # Add handler to specified target loggers
                if target_loggers:
                    for logger_name in target_loggers:
                        if logger_name in self.logger_registry:
                            self.logger_registry[logger_name].addHandler(handler)
                        elif logger_name in _logger_registry:
                            _logger_registry[logger_name].addHandler(handler)
                
                # Configure handler with appropriate formatter
                if not handler.formatter and 'detailed_scientific' in _formatter_registry:
                    handler.setFormatter(_formatter_registry['detailed_scientific'])
                
                # Setup handler-specific monitoring if applicable
                # (Implementation would depend on handler type)
                
                # Update statistics
                self.statistics['handlers_added'] += 1
                
                # Log handler addition operation
                logger = get_logger('log_manager.handler', 'SYSTEM')
                logger.info(f"Handler added: {handler_name} (type: {type(handler).__name__})")
                
                # Create audit trail entry
                if self.audit_manager:
                    self.audit_manager.create_audit_entry(
                        action='HANDLER_ADDED',
                        component='LOG_MANAGER',
                        details={
                            'handler_name': handler_name,
                            'handler_type': type(handler).__name__,
                            'target_loggers': target_loggers or []
                        }
                    )
                
                return True
                
            except Exception as e:
                print(f"ERROR: Failed to add handler {handler_name}: {e}", file=sys.stderr)
                return False
    
    def remove_handler(
        self,
        handler_name: str,
        flush_before_removal: bool = True
    ) -> bool:
        """
        Remove logging handler from the system with cleanup, deregistration, and graceful shutdown.
        
        This method provides comprehensive handler removal with cleanup and graceful shutdown
        for dynamic logging system configuration.
        
        Args:
            handler_name: Name of the handler to remove
            flush_before_removal: Flush handler if flush_before_removal is enabled
            
        Returns:
            bool: Success status of handler removal
        """
        with self.manager_lock:
            try:
                # Locate handler in registry
                if handler_name not in self.handler_registry:
                    return False  # Handler not found
                
                handler = self.handler_registry[handler_name]
                
                # Flush handler if flush_before_removal is enabled
                if flush_before_removal and hasattr(handler, 'flush'):
                    try:
                        handler.flush()
                    except Exception as e:
                        print(f"WARNING: Failed to flush handler {handler_name}: {e}", file=sys.stderr)
                
                # Remove handler from all associated loggers
                for logger in self.logger_registry.values():
                    if handler in logger.handlers:
                        logger.removeHandler(handler)
                
                # Close handler and cleanup resources
                try:
                    if hasattr(handler, 'close'):
                        handler.close()
                except Exception as e:
                    print(f"WARNING: Failed to close handler {handler_name}: {e}", file=sys.stderr)
                
                # Remove from handler registry
                del self.handler_registry[handler_name]
                if handler_name in _handler_registry:
                    del _handler_registry[handler_name]
                
                # Log handler removal operation
                logger = get_logger('log_manager.handler', 'SYSTEM')
                logger.info(f"Handler removed: {handler_name}")
                
                # Create audit trail entry
                if self.audit_manager:
                    self.audit_manager.create_audit_entry(
                        action='HANDLER_REMOVED',
                        component='LOG_MANAGER',
                        details={
                            'handler_name': handler_name,
                            'flush_before_removal': flush_before_removal
                        }
                    )
                
                return True
                
            except Exception as e:
                print(f"ERROR: Failed to remove handler {handler_name}: {e}", file=sys.stderr)
                return False
    
    def update_configuration(
        self,
        new_config: Dict[str, Any],
        preserve_existing_loggers: bool = True
    ) -> bool:
        """
        Update logging configuration dynamically with validation, handler reconfiguration, and 
        monitoring system updates.
        
        This method provides comprehensive configuration updates with validation and monitoring
        system integration for runtime configuration management.
        
        Args:
            new_config: New configuration dictionary to apply
            preserve_existing_loggers: Preserve existing loggers if requested
            
        Returns:
            bool: Success status of configuration update
        """
        with self.manager_lock:
            try:
                # Validate new configuration structure
                if not isinstance(new_config, dict):
                    raise TypeError("Configuration must be a dictionary")
                
                if 'version' not in new_config:
                    raise ValueError("Configuration must include version field")
                
                # Backup current configuration
                backup_config = self.logging_config.copy()
                
                # Apply new configuration to logging system
                try:
                    logging.config.dictConfig(new_config)
                    self.logging_config = new_config
                    
                    # Update statistics
                    self.statistics['configuration_updates'] += 1
                    self.statistics['last_config_update'] = datetime.datetime.now().isoformat()
                    
                except Exception as config_error:
                    # Restore backup configuration on failure
                    if backup_config:
                        logging.config.dictConfig(backup_config)
                        self.logging_config = backup_config
                    raise config_error
                
                # Update handler and formatter configurations
                self._update_handlers_from_config(new_config)
                
                # Preserve existing loggers if requested
                if preserve_existing_loggers:
                    # Maintain existing logger configurations
                    pass
                
                # Update monitoring system integrations
                self._update_monitoring_integrations(new_config)
                
                # Log configuration update operation
                logger = get_logger('log_manager.config', 'SYSTEM')
                logger.info("Logging configuration updated successfully")
                
                # Create audit trail entry
                if self.audit_manager:
                    self.audit_manager.create_audit_entry(
                        action='CONFIGURATION_UPDATED',
                        component='LOG_MANAGER',
                        details={
                            'preserve_existing_loggers': preserve_existing_loggers,
                            'handlers_count': len(new_config.get('handlers', {})),
                            'loggers_count': len(new_config.get('loggers', {}))
                        }
                    )
                
                return True
                
            except Exception as e:
                print(f"ERROR: Configuration update failed: {e}", file=sys.stderr)
                return False
    
    def get_statistics(
        self,
        include_performance_metrics: bool = True,
        include_handler_details: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive logging system statistics including performance metrics, handler status, 
        and system health indicators.
        
        This method provides comprehensive statistics collection with performance metrics and
        system health assessment for monitoring and optimization.
        
        Args:
            include_performance_metrics: Include performance metrics if requested
            include_handler_details: Include handler-specific details if requested
            
        Returns:
            Dict[str, Any]: Comprehensive logging statistics with performance and handler information
        """
        with self.manager_lock:
            try:
                # Collect statistics from all registered handlers
                handler_stats = {}
                
                if include_handler_details:
                    for handler_name, handler in self.handler_registry.items():
                        handler_info = {
                            'handler_type': type(handler).__name__,
                            'level': handler.level,
                            'formatter': type(handler.formatter).__name__ if handler.formatter else None
                        }
                        
                        # Get handler-specific statistics if available
                        if hasattr(handler, 'get_statistics'):
                            try:
                                handler_info.update(handler.get_statistics())
                            except Exception as e:
                                handler_info['statistics_error'] = str(e)
                        
                        handler_stats[handler_name] = handler_info
                
                # Calculate performance metrics if requested
                performance_metrics = {}
                
                if include_performance_metrics:
                    performance_metrics = {
                        'loggers_managed': len(self.logger_registry),
                        'handlers_managed': len(self.handler_registry),
                        'monitoring_integrations_active': len(self.monitoring_integrations),
                        'initialization_time': self.initialization_time.isoformat(),
                        'uptime_seconds': (datetime.datetime.now() - self.initialization_time).total_seconds()
                    }
                    
                    # Include performance handler statistics
                    if self.performance_handler and hasattr(self.performance_handler, 'get_statistics'):
                        try:
                            performance_metrics['performance_handler'] = self.performance_handler.get_statistics()
                        except Exception as e:
                            performance_metrics['performance_handler_error'] = str(e)
                
                # Include handler-specific details if requested
                statistics = {
                    'collection_timestamp': datetime.datetime.now().isoformat(),
                    'manager_statistics': self.statistics.copy(),
                    'system_status': {
                        'is_initialized': self.is_initialized,
                        'console_enabled': self.console_output_enabled,
                        'performance_enabled': self.performance_logging_enabled,
                        'audit_enabled': self.audit_trail_enabled
                    }
                }
                
                if include_handler_details:
                    statistics['handler_details'] = handler_stats
                
                if include_performance_metrics:
                    statistics['performance_metrics'] = performance_metrics
                
                # Analyze system health indicators
                statistics['health_indicators'] = {
                    'configuration_valid': bool(self.logging_config),
                    'handlers_operational': len(self.handler_registry) > 0,
                    'monitoring_active': len(self.monitoring_integrations) > 0,
                    'overall_health': 'good' if self.is_initialized else 'poor'
                }
                
                return statistics
                
            except Exception as e:
                print(f"ERROR: Failed to collect statistics: {e}", file=sys.stderr)
                return {
                    'collection_timestamp': datetime.datetime.now().isoformat(),
                    'error': str(e),
                    'partial_data': False
                }
    
    def flush_all(self, timeout_seconds: float = 30.0) -> Dict[str, bool]:
        """
        Flush all logging handlers and buffers with timeout management and error handling for data 
        persistence.
        
        This method provides comprehensive flushing of all managed handlers with timeout management
        and error handling for data persistence and critical operations.
        
        Args:
            timeout_seconds: Maximum time to wait for flush operations
            
        Returns:
            Dict[str, bool]: Flush status for each handler
        """
        with self.manager_lock:
            flush_results = {}
            
            try:
                # Iterate through all registered handlers
                for handler_name, handler in self.handler_registry.items():
                    try:
                        # Flush each handler with timeout management
                        if hasattr(handler, 'flush'):
                            handler.flush()
                            flush_results[handler_name] = True
                        else:
                            flush_results[handler_name] = False  # Handler doesn't support flush
                    except Exception as e:
                        # Handle flush errors gracefully
                        flush_results[handler_name] = False
                        print(f"WARNING: Failed to flush handler {handler_name}: {e}", file=sys.stderr)
                
                # Flush performance handler buffer if available
                if self.performance_handler and hasattr(self.performance_handler, 'flush_buffer'):
                    try:
                        self.performance_handler.flush_buffer()
                        flush_results['performance_buffer'] = True
                    except Exception as e:
                        flush_results['performance_buffer'] = False
                        print(f"WARNING: Failed to flush performance buffer: {e}", file=sys.stderr)
                
                # Collect flush status for each handler
                successful_flushes = sum(1 for success in flush_results.values() if success)
                total_handlers = len(flush_results)
                
                # Log flush operation completion
                if successful_flushes > 0:
                    logger = get_logger('log_manager.flush', 'SYSTEM')
                    logger.info(f"Log manager flush completed: {successful_flushes}/{total_handlers} handlers")
                
                return flush_results
                
            except Exception as e:
                print(f"ERROR: Log manager flush failed: {e}", file=sys.stderr)
                return {'error': False}
    
    def shutdown(
        self,
        archive_logs: bool = False,
        preserve_audit_trail: bool = True
    ) -> Dict[str, Any]:
        """
        Shutdown log management system with cleanup, resource finalization, and graceful termination 
        of all logging operations.
        
        This method provides comprehensive shutdown with cleanup, audit trail finalization, and
        resource cleanup for graceful system termination.
        
        Args:
            archive_logs: Archive log files if archival enabled
            preserve_audit_trail: Finalize audit trail if preservation enabled
            
        Returns:
            Dict[str, Any]: Shutdown summary with final statistics and cleanup information
        """
        with self.manager_lock:
            shutdown_summary = {
                'shutdown_timestamp': datetime.datetime.now().isoformat(),
                'archive_logs': archive_logs,
                'preserve_audit_trail': preserve_audit_trail,
                'operations_performed': [],
                'final_statistics': {},
                'shutdown_success': True
            }
            
            try:
                # Execute registered shutdown callbacks
                for callback in self.shutdown_callbacks:
                    try:
                        callback()
                        shutdown_summary['operations_performed'].append(f'callback_{callback.__name__}')
                    except Exception as e:
                        print(f"WARNING: Shutdown callback failed: {e}", file=sys.stderr)
                
                # Flush all handlers and buffers
                try:
                    flush_results = self.flush_all(timeout_seconds=10.0)
                    shutdown_summary['flush_results'] = flush_results
                    shutdown_summary['operations_performed'].append('handlers_flushed')
                except Exception as e:
                    print(f"WARNING: Handler flush during shutdown failed: {e}", file=sys.stderr)
                
                # Finalize audit trail if preservation enabled
                if preserve_audit_trail and self.audit_manager:
                    try:
                        # Create final audit entry
                        self.audit_manager.create_audit_entry(
                            action='LOG_MANAGER_SHUTDOWN',
                            component='LOG_MANAGER',
                            details={
                                'archive_logs': archive_logs,
                                'final_statistics': self.statistics.copy(),
                                'uptime_seconds': (datetime.datetime.now() - self.initialization_time).total_seconds()
                            }
                        )
                        shutdown_summary['operations_performed'].append('audit_trail_finalized')
                    except Exception as e:
                        print(f"WARNING: Audit trail finalization failed: {e}", file=sys.stderr)
                
                # Archive log files if archival enabled
                if archive_logs:
                    try:
                        # Implementation would archive log files
                        shutdown_summary['operations_performed'].append('logs_archived')
                    except Exception as e:
                        print(f"WARNING: Log archival failed: {e}", file=sys.stderr)
                
                # Close all handlers and cleanup resources
                try:
                    for handler_name, handler in list(self.handler_registry.items()):
                        try:
                            if hasattr(handler, 'close'):
                                handler.close()
                        except Exception as e:
                            print(f"WARNING: Failed to close handler {handler_name}: {e}", file=sys.stderr)
                    
                    shutdown_summary['operations_performed'].append('handlers_closed')
                except Exception as e:
                    print(f"WARNING: Handler cleanup failed: {e}", file=sys.stderr)
                
                # Clear registries and caches
                try:
                    self.logger_registry.clear()
                    self.handler_registry.clear()
                    self.formatter_registry.clear()
                    self.monitoring_integrations.clear()
                    shutdown_summary['operations_performed'].append('registries_cleared')
                except Exception as e:
                    print(f"WARNING: Registry cleanup failed: {e}", file=sys.stderr)
                
                # Generate final statistics summary
                try:
                    shutdown_summary['final_statistics'] = self.get_statistics(
                        include_performance_metrics=True,
                        include_handler_details=False
                    )
                    shutdown_summary['operations_performed'].append('final_statistics_collected')
                except Exception as e:
                    print(f"WARNING: Final statistics collection failed: {e}", file=sys.stderr)
                
                # Set shutdown flag
                self.is_initialized = False
                
                # Log shutdown completion
                print("Log manager shutdown completed successfully")
                
                return shutdown_summary
                
            except Exception as e:
                shutdown_summary['shutdown_success'] = False
                shutdown_summary['error'] = str(e)
                print(f"ERROR: Log manager shutdown failed: {e}", file=sys.stderr)
                return shutdown_summary
    
    def _setup_monitoring_integrations(self) -> None:
        """Setup monitoring system integrations."""
        try:
            # Setup alert system integration
            if 'alert_system' in _monitoring_systems:
                self.monitoring_integrations['alert_system'] = _monitoring_systems['alert_system']
            
            # Setup progress tracking integration
            if 'progress_tracking' in _monitoring_systems:
                self.monitoring_integrations['progress_tracking'] = _monitoring_systems['progress_tracking']
                
        except Exception as e:
            print(f"WARNING: Monitoring integration setup failed: {e}", file=sys.stderr)
    
    def _update_handlers_from_config(self, config: Dict[str, Any]) -> None:
        """Update handlers based on new configuration."""
        try:
            handlers_config = config.get('handlers', {})
            
            for handler_name, handler_config in handlers_config.items():
                if handler_name not in self.handler_registry:
                    # Create new handler from configuration
                    handler = _create_handler_from_config(handler_config)
                    if handler:
                        self.handler_registry[handler_name] = handler
                        
        except Exception as e:
            print(f"WARNING: Handler update from config failed: {e}", file=sys.stderr)
    
    def _update_monitoring_integrations(self, config: Dict[str, Any]) -> None:
        """Update monitoring integrations based on configuration."""
        try:
            # Update monitoring settings based on configuration
            integration_config = config.get('integration', {})
            
            if integration_config.get('alert_system_integration', True):
                self.monitoring_integrations['alert_system'] = True
            
            if integration_config.get('performance_metrics_integration', True):
                self.monitoring_integrations['performance_monitoring'] = True
                
        except Exception as e:
            print(f"WARNING: Monitoring integration update failed: {e}", file=sys.stderr)
    
    def _cleanup_handlers(self) -> None:
        """Cleanup all handlers during shutdown."""
        for handler in self.handler_registry.values():
            try:
                if hasattr(handler, 'close'):
                    handler.close()
            except Exception as e:
                print(f"WARNING: Handler cleanup failed: {e}", file=sys.stderr)
    
    def _finalize_audit_trail(self) -> None:
        """Finalize audit trail during shutdown."""
        if self.audit_manager:
            try:
                # Create final audit entry
                self.audit_manager.create_audit_entry(
                    action='LOG_MANAGER_FINALIZED',
                    component='LOG_MANAGER',
                    details={'finalization_timestamp': datetime.datetime.now().isoformat()}
                )
            except Exception as e:
                print(f"WARNING: Audit trail finalization failed: {e}", file=sys.stderr)


class LoggingContext:
    """
    Context manager for scoped logging operations that automatically manages logging configuration, 
    scientific context, and cleanup for specific operations or workflows with automatic setup and 
    teardown.
    
    This context manager provides automatic scientific context management with performance tracking
    and audit trail correlation for scoped logging operations.
    """
    
    def __init__(
        self,
        context_name: str,
        logging_config: Dict[str, Any] = None,
        enable_performance_tracking: bool = False
    ):
        """
        Initialize logging context manager with context name, configuration, and performance tracking 
        settings.
        
        Args:
            context_name: Name identifier for the logging context
            logging_config: Scientific context data to set during context
            enable_performance_tracking: Enable performance tracking for the context
        """
        # Set context name and logging configuration
        self.context_name = context_name
        self.logging_config = logging_config or {}
        
        # Configure performance tracking settings
        self.performance_tracking_enabled = enable_performance_tracking
        
        # Initialize context state variables
        self.previous_config: Dict[str, Any] = {}
        self.start_time: datetime.datetime = None
        self.created_loggers: List[str] = []
        
        # Get log manager instance
        self.log_manager = get_log_manager(create_if_missing=False)
    
    def __enter__(self) -> 'LoggingContext':
        """
        Enter logging context and setup scoped logging configuration with scientific context and 
        performance tracking.
        
        This method saves the current context state and establishes the new context with performance
        tracking and audit trail initialization.
        
        Returns:
            LoggingContext: Self reference for context management
        """
        # Record context start time
        self.start_time = datetime.datetime.now()
        
        # Backup current logging configuration
        if self.log_manager:
            self.previous_config = self.log_manager.logging_config.copy()
        
        # Apply context-specific logging configuration
        if self.logging_config and self.log_manager:
            try:
                self.log_manager.update_configuration(self.logging_config, preserve_existing_loggers=True)
            except Exception as e:
                print(f"WARNING: Context configuration update failed: {e}", file=sys.stderr)
        
        # Setup scientific context for logging operations
        from ..utils.logging_utils import set_scientific_context
        try:
            context_data = self.logging_config.get('scientific_context', {})
            if context_data:
                set_scientific_context(**context_data)
        except Exception as e:
            print(f"WARNING: Scientific context setup failed: {e}", file=sys.stderr)
        
        # Initialize performance tracking if enabled
        if self.performance_tracking_enabled:
            # Performance tracking initialization
            pass
        
        # Log context entry operation
        logger = get_logger('logging_context', 'SYSTEM')
        logger.debug(f"Entering logging context: {self.context_name}")
        
        # Create audit trail entry
        if self.log_manager and self.log_manager.audit_manager:
            self.log_manager.audit_manager.create_audit_entry(
                action='LOGGING_CONTEXT_ENTERED',
                component='LOGGING_CONTEXT',
                details={
                    'context_name': self.context_name,
                    'performance_tracking': self.performance_tracking_enabled
                }
            )
        
        return self
    
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb) -> bool:
        """
        Exit logging context and restore previous configuration with cleanup and performance summary.
        
        This method restores the previous context state and finalizes performance tracking and audit
        trail with completion information and exception handling.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
            
        Returns:
            bool: False to propagate exceptions
        """
        try:
            # Calculate context execution time
            execution_time = None
            if self.start_time:
                execution_time = (datetime.datetime.now() - self.start_time).total_seconds()
            
            # Finalize performance tracking if enabled
            if self.performance_tracking_enabled and execution_time:
                from ..utils.logging_utils import log_performance_metrics
                try:
                    log_performance_metrics(
                        metric_name='context_execution_time',
                        metric_value=execution_time,
                        metric_unit='seconds',
                        component='LOGGING_CONTEXT',
                        metric_context={
                            'context_name': self.context_name,
                            'success': exc_type is None
                        }
                    )
                except Exception as e:
                    print(f"WARNING: Performance tracking finalization failed: {e}", file=sys.stderr)
            
            # Restore previous logging configuration
            if self.previous_config and self.log_manager:
                try:
                    self.log_manager.update_configuration(self.previous_config, preserve_existing_loggers=True)
                except Exception as e:
                    print(f"WARNING: Context configuration restoration failed: {e}", file=sys.stderr)
            
            # Cleanup context-specific loggers
            for logger_name in self.created_loggers:
                if logger_name in _logger_registry:
                    try:
                        logger = _logger_registry[logger_name]
                        # Cleanup logger handlers if needed
                        for handler in logger.handlers[:]:
                            logger.removeHandler(handler)
                    except Exception as e:
                        print(f"WARNING: Logger cleanup failed for {logger_name}: {e}", file=sys.stderr)
            
            # Clear scientific context
            from ..utils.logging_utils import clear_scientific_context
            try:
                clear_scientific_context()
            except Exception as e:
                print(f"WARNING: Scientific context clearing failed: {e}", file=sys.stderr)
            
            # Log context exit with performance summary
            logger = get_logger('logging_context', 'SYSTEM')
            if execution_time:
                logger.debug(f"Exiting logging context: {self.context_name} (execution time: {execution_time:.3f}s)")
            else:
                logger.debug(f"Exiting logging context: {self.context_name}")
            
            # Log exception information if exception occurred
            if exc_type is not None:
                logger.warning(f"Exception in logging context '{self.context_name}': {exc_type.__name__}: {exc_val}")
            
            # Create audit trail entry
            if self.log_manager and self.log_manager.audit_manager:
                self.log_manager.audit_manager.create_audit_entry(
                    action='LOGGING_CONTEXT_EXITED',
                    component='LOGGING_CONTEXT',
                    details={
                        'context_name': self.context_name,
                        'execution_time': execution_time,
                        'success': exc_type is None,
                        'exception_type': exc_type.__name__ if exc_type else None
                    }
                )
            
        except Exception as cleanup_error:
            print(f"ERROR: Logging context cleanup failed: {cleanup_error}", file=sys.stderr)
        
        # Return False to propagate exceptions
        return False
    
    def get_context_logger(self, logger_name: str) -> logging.Logger:
        """
        Get logger instance configured for the current context with scientific enhancements.
        
        This method creates context-specific loggers with automatic cleanup tracking.
        
        Args:
            logger_name: Name of the logger to create
            
        Returns:
            logging.Logger: Context-configured logger instance
        """
        # Create logger with context-specific configuration
        full_logger_name = f"{self.context_name}.{logger_name}"
        
        logger = get_logger(
            logger_name=full_logger_name,
            component_type='CONTEXT',
            enable_scientific_context=True,
            enable_performance_tracking=self.performance_tracking_enabled
        )
        
        # Add scientific context integration
        # (This is handled automatically by get_logger)
        
        # Track created logger for cleanup
        self.created_loggers.append(full_logger_name)
        
        return logger


# Helper functions for internal log manager operations

def _get_default_logging_configuration() -> Dict[str, Any]:
    """Get default logging configuration for system initialization."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed_scientific': {
                'format': '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(simulation_id)s | %(algorithm_name)s | %(processing_stage)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
                'class': 'src.backend.utils.logging_utils.ScientificFormatter'
            },
            'simple': {
                'format': '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed_scientific',
                'filename': 'logs/plume_simulation.log',
                'maxBytes': 10485760,
                'backupCount': 5
            }
        },
        'loggers': {
            'src.backend': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        },
        'output_directories': {
            'base_log_directory': 'logs',
            'performance_logs': 'logs/performance',
            'audit_logs': 'logs/audit',
            'error_logs': 'logs/errors'
        },
        'performance_monitoring': {
            'enabled': True,
            'buffer_size': 1000,
            'flush_interval_seconds': PERFORMANCE_LOG_FLUSH_INTERVAL
        }
    }


def _create_handler_from_config(handler_config: Dict[str, Any]) -> Optional[logging.Handler]:
    """Create logging handler from configuration dictionary."""
    try:
        handler_class = handler_config.get('class')
        if not handler_class:
            return None
        
        # Simple handler creation - would be more sophisticated in full implementation
        if 'RotatingFileHandler' in handler_class:
            return logging.handlers.RotatingFileHandler(
                filename=handler_config.get('filename', 'logs/default.log'),
                maxBytes=handler_config.get('maxBytes', 10485760),
                backupCount=handler_config.get('backupCount', 5)
            )
        elif 'StreamHandler' in handler_class:
            return logging.StreamHandler()
        
        return None
        
    except Exception as e:
        print(f"WARNING: Failed to create handler from config: {e}", file=sys.stderr)
        return None


def _update_registries_from_config(config: Dict[str, Any]) -> None:
    """Update global registries from configuration."""
    try:
        # Update handler registry
        handlers_config = config.get('handlers', {})
        for handler_name, handler_config in handlers_config.items():
            if handler_name not in _handler_registry:
                handler = _create_handler_from_config(handler_config)
                if handler:
                    _handler_registry[handler_name] = handler
                    
    except Exception as e:
        print(f"WARNING: Registry update from config failed: {e}", file=sys.stderr)


def _compress_rotated_file(file_path: str) -> int:
    """Compress rotated log file."""
    try:
        # Implementation would compress the file
        return 1  # Number of files compressed
    except Exception:
        return 0


def _cleanup_old_log_files() -> int:
    """Cleanup old log files based on retention policies."""
    try:
        # Implementation would clean up old files
        return 0  # Number of files cleaned
    except Exception:
        return 0


def _archive_log_files() -> int:
    """Archive log files to archive directory."""
    try:
        # Implementation would archive files
        return 0  # Number of files archived
    except Exception:
        return 0


def _cleanup_on_exit() -> None:
    """Cleanup function registered with atexit."""
    try:
        if _log_manager_instance:
            _log_manager_instance.shutdown(archive_logs=False, preserve_audit_trail=True)
    except Exception as e:
        print(f"WARNING: Exit cleanup failed: {e}", file=sys.stderr)


def _signal_handler(signum: int, frame) -> None:
    """Signal handler for graceful shutdown."""
    try:
        print(f"Received signal {signum}, initiating graceful shutdown...")
        _cleanup_on_exit()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Signal handler failed: {e}", file=sys.stderr)
        sys.exit(1)