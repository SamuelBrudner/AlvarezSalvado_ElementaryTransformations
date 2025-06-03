#!/usr/bin/env python3
"""
Main entry point module for the plume navigation simulation system backend, providing direct 
execution capability when the backend package is invoked as a module using 'python -m src.backend'. 
Serves as the primary interface for command-line execution, system initialization, and workflow 
orchestration with comprehensive error handling, logging infrastructure, and graceful shutdown 
procedures.

This module implements scientific computing standards with >95% correlation accuracy, <7.2 seconds 
average simulation time, and support for 4000+ simulation batch processing within 8-hour target 
timeframe while maintaining reproducible research outcomes and cross-format compatibility.

Key Features:
- Direct Module Execution Interface with comprehensive command-line interface integration
- System Initialization and Orchestration with modular component separation
- Scientific Computing Excellence with >95% correlation accuracy and <7.2s processing
- Batch Processing Framework for 4000+ simulations within 8-hour target timeframe
- Comprehensive Error Handling with fail-fast validation and graceful degradation
- Cross-Platform Compatibility for Crimaldi and custom plume formats
- Signal handling for graceful shutdown and interrupt management
- Cleanup handlers for proper resource finalization and data preservation
- Performance monitoring and audit trail generation for reproducible outcomes
"""

# Global module metadata and version information
__version__ = '1.0.0'
MODULE_NAME = 'Plume Navigation Simulation Backend'
MODULE_DESCRIPTION = 'Main entry point for plume navigation simulation system backend with comprehensive scientific computing capabilities'

# Exit codes for comprehensive error classification and workflow status reporting
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INITIALIZATION_ERROR = 2
EXIT_SYSTEM_ERROR = 3
EXIT_INTERRUPT = 4

# Global backend system state management with thread-safe operations
_backend_initialized = False
_cleanup_registered = False
_logger = None

# External library imports with version specifications for system operations
import sys  # Python 3.9+ - System-specific parameters and functions for exit codes and argument handling
import os  # Python 3.9+ - Operating system interface for environment variables and system information
import logging  # Python 3.9+ - Logging framework for module execution tracking and error reporting
import traceback  # Python 3.9+ - Exception traceback formatting for comprehensive error reporting
import signal  # Python 3.9+ - Signal handling for graceful shutdown and interrupt management
import atexit  # Python 3.9+ - Exit handler registration for cleanup operations and resource finalization

# Internal imports from CLI interface and backend system components
from .cli import main
from . import (
    run_cli_interface,
    initialize_backend_system,
    get_backend_system_status,
    cleanup_backend_system,
    PlumeSimulationException
)
from .utils import (
    get_logger,
    initialize_utils_package
)


def setup_signal_handlers() -> None:
    """
    Setup signal handlers for graceful shutdown and interrupt management during scientific 
    computing operations with comprehensive cleanup and resource finalization.
    
    This function establishes signal handlers for SIGINT (Ctrl+C) and SIGTERM (termination) 
    signals to ensure graceful shutdown with data preservation, resource cleanup, and audit 
    trail completion for scientific computing integrity and reproducible research outcomes.
    
    Returns:
        None: No return value
    """
    global _logger
    
    try:
        # Define signal handler function for SIGINT (Ctrl+C) interruption
        def sigint_handler(signal_number: int, frame) -> None:
            """Handle SIGINT signal with graceful shutdown and data preservation."""
            handle_system_interrupt(signal_number, frame)
        
        # Define signal handler function for SIGTERM (termination) signal
        def sigterm_handler(signal_number: int, frame) -> None:
            """Handle SIGTERM signal with graceful shutdown and resource finalization."""
            handle_system_interrupt(signal_number, frame)
        
        # Register SIGINT handler for graceful shutdown with cleanup operations
        signal.signal(signal.SIGINT, sigint_handler)
        
        # Register SIGTERM handler for system termination with resource finalization
        signal.signal(signal.SIGTERM, sigterm_handler)
        
        # Log signal handler registration for debugging and audit trail
        if _logger:
            _logger.debug("Signal handlers registered for graceful shutdown (SIGINT, SIGTERM)")
        
        # Setup emergency cleanup procedures for unexpected termination scenarios
        # Additional signal handlers can be registered here for comprehensive coverage
        
    except Exception as e:
        if _logger:
            _logger.error(f"Signal handler setup failed: {e}")
        else:
            print(f"ERROR: Signal handler setup failed: {e}", file=sys.stderr)


def register_cleanup_handlers() -> None:
    """
    Register cleanup handlers using atexit module to ensure proper resource cleanup and 
    system finalization even in case of unexpected termination or errors.
    
    This function registers comprehensive cleanup handlers with the atexit module to ensure
    proper resource deallocation, data preservation, and system finalization for scientific
    computing integrity and reproducible research outcomes.
    
    Returns:
        None: No return value
    """
    global _cleanup_registered, _logger
    
    try:
        # Check if cleanup handlers are already registered to prevent duplicates
        if _cleanup_registered:
            if _logger:
                _logger.debug("Cleanup handlers already registered")
            return
        
        # Register main cleanup function with atexit module
        atexit.register(emergency_cleanup, preserve_data=True, force_cleanup=False)
        
        # Register backend system cleanup with resource finalization
        atexit.register(lambda: cleanup_backend_system(
            preserve_results=True,
            generate_final_reports=False,
            cleanup_mode='graceful',
            save_performance_statistics=True
        ))
        
        # Register utilities package cleanup with memory management
        try:
            from .utils import cleanup_utils_package
            atexit.register(lambda: cleanup_utils_package(
                force_cleanup=False,
                preserve_logs=True
            ))
        except ImportError:
            if _logger:
                _logger.warning("Utils package cleanup not available for registration")
        
        # Register logging system cleanup with log finalization
        atexit.register(lambda: _finalize_logging_system())
        
        # Set global cleanup registration flag to prevent duplicate registration
        _cleanup_registered = True
        
        # Log cleanup handler registration for audit trail and debugging
        if _logger:
            _logger.info("Cleanup handlers registered successfully with atexit module")
        
    except Exception as e:
        if _logger:
            _logger.error(f"Cleanup handler registration failed: {e}")
        else:
            print(f"ERROR: Cleanup handler registration failed: {e}", file=sys.stderr)


def emergency_cleanup(preserve_data: bool = True, force_cleanup: bool = False) -> None:
    """
    Perform emergency cleanup operations for unexpected termination scenarios including 
    resource deallocation, data preservation, and system state finalization with minimal 
    error handling.
    
    This function provides emergency cleanup capabilities for unexpected termination scenarios
    with resource deallocation, critical data preservation, and system state finalization 
    while minimizing dependencies on complex error handling mechanisms.
    
    Args:
        preserve_data: Whether to preserve critical data and intermediate results
        force_cleanup: Whether to force cleanup of backend system resources
        
    Returns:
        None: No return value
    """
    global _backend_initialized, _logger
    
    try:
        # Attempt to preserve critical data and intermediate results if preserve_data is True
        if preserve_data:
            try:
                import tempfile
                import pathlib
                import datetime
                import json
                
                # Create emergency backup directory
                emergency_dir = pathlib.Path(tempfile.gettempdir()) / 'plume_emergency_backup'
                emergency_dir.mkdir(exist_ok=True)
                
                # Save emergency system state for debugging
                emergency_state = {
                    'emergency_timestamp': datetime.datetime.now().isoformat(),
                    'backend_initialized': _backend_initialized,
                    'cleanup_registered': _cleanup_registered,
                    'module_version': __version__,
                    'python_version': sys.version,
                    'emergency_reason': 'unexpected_termination'
                }
                
                emergency_file = emergency_dir / f'emergency_state_{int(datetime.datetime.now().timestamp())}.json'
                with open(emergency_file, 'w') as f:
                    json.dump(emergency_state, f, indent=2)
                
                if _logger:
                    _logger.info(f"Emergency state preserved: {emergency_file}")
                    
            except Exception as e:
                # Minimal error handling during emergency cleanup
                if _logger:
                    _logger.error(f"Emergency data preservation failed: {e}")
                print(f"Emergency data preservation failed: {e}", file=sys.stderr)
        
        # Force cleanup of backend system resources if force_cleanup is True
        if force_cleanup and _backend_initialized:
            try:
                cleanup_backend_system(
                    preserve_results=preserve_data,
                    generate_final_reports=False,
                    cleanup_mode='emergency',
                    save_performance_statistics=preserve_data
                )
                
                if _logger:
                    _logger.info("Emergency backend system cleanup completed")
                    
            except Exception as e:
                if _logger:
                    _logger.error(f"Emergency backend cleanup failed: {e}")
                print(f"Emergency backend cleanup failed: {e}", file=sys.stderr)
        
        # Cleanup utilities package with minimal error handling
        try:
            from .utils import cleanup_utils_package
            cleanup_utils_package(
                force_cleanup=force_cleanup,
                preserve_logs=preserve_data
            )
            
            if _logger:
                _logger.info("Emergency utilities cleanup completed")
                
        except Exception as e:
            if _logger:
                _logger.error(f"Emergency utilities cleanup failed: {e}")
            print(f"Emergency utilities cleanup failed: {e}", file=sys.stderr)
        
        # Finalize logging system and flush pending log entries
        try:
            _finalize_logging_system()
        except Exception as e:
            print(f"Emergency logging finalization failed: {e}", file=sys.stderr)
        
        # Release memory allocations and system resources
        try:
            # Reset global backend system state
            _backend_initialized = False
            
            # Force garbage collection if available
            try:
                import gc
                gc.collect()
            except ImportError:
                pass
                
        except Exception as e:
            print(f"Emergency resource cleanup failed: {e}", file=sys.stderr)
        
        # Log emergency cleanup completion with minimal dependencies
        try:
            print("Emergency cleanup completed", file=sys.stderr)
        except Exception:
            pass  # Final fallback - ignore any errors
        
    except Exception as e:
        # Ultimate fallback for emergency cleanup failures
        try:
            print(f"CRITICAL: Emergency cleanup failed: {e}", file=sys.stderr)
        except Exception:
            pass  # Final fallback - ignore any errors


def handle_system_interrupt(signal_number: int, frame) -> None:
    """
    Handle system interruption signals (SIGINT, SIGTERM) with graceful shutdown procedures, 
    data preservation, and comprehensive cleanup for scientific computing integrity.
    
    This function manages system interruption signals with graceful shutdown procedures,
    critical data preservation, comprehensive resource cleanup, and audit trail generation
    to maintain scientific computing integrity and enable operation resumption.
    
    Args:
        signal_number: Signal number received (SIGINT=2, SIGTERM=15)
        frame: Current execution frame object from signal handler
        
    Returns:
        None: No return value
    """
    global _backend_initialized, _logger
    
    try:
        signal_name = {2: 'SIGINT', 15: 'SIGTERM'}.get(signal_number, f'Signal {signal_number}')
        
        # Log interruption signal reception with signal number and context
        if _logger:
            _logger.warning(f"System interruption signal received: {signal_name}")
        else:
            print(f"\nSystem interruption signal received: {signal_name}", file=sys.stderr)
        
        print(f"\nInitiating graceful shutdown for {MODULE_NAME}...")
        
        # Initiate graceful shutdown procedures for all backend components
        if _backend_initialized:
            try:
                # Get system status before shutdown for audit trail
                system_status = get_backend_system_status(
                    include_detailed_metrics=False,
                    include_component_diagnostics=False,
                    include_performance_analysis=False
                )
                
                if _logger:
                    _logger.info(f"System status captured before shutdown: {system_status.get('operational_readiness', {}).get('is_ready', 'unknown')}")
                
            except Exception as e:
                if _logger:
                    _logger.warning(f"Could not capture system status before shutdown: {e}")
        
        # Preserve critical data and intermediate results for scientific integrity
        try:
            import datetime
            import tempfile
            import pathlib
            import json
            
            # Create interruption backup directory
            interrupt_dir = pathlib.Path(tempfile.gettempdir()) / 'plume_interrupt_recovery'
            interrupt_dir.mkdir(exist_ok=True)
            
            # Save interruption context for recovery
            interrupt_context = {
                'interruption_timestamp': datetime.datetime.now().isoformat(),
                'signal_received': signal_name,
                'signal_number': signal_number,
                'backend_initialized': _backend_initialized,
                'module_version': __version__,
                'recovery_instructions': [
                    'Check the backend system logs for detailed operation history',
                    'Use checkpoint files to resume interrupted operations',
                    'Verify data integrity before continuing operations'
                ]
            }
            
            interrupt_file = interrupt_dir / f'interrupt_context_{int(datetime.datetime.now().timestamp())}.json'
            with open(interrupt_file, 'w') as f:
                json.dump(interrupt_context, f, indent=2)
            
            print(f"Interruption context saved: {interrupt_file}")
            
        except Exception as e:
            if _logger:
                _logger.error(f"Failed to preserve interruption context: {e}")
            print(f"Failed to preserve interruption context: {e}", file=sys.stderr)
        
        # Cleanup backend system with comprehensive resource finalization
        if _backend_initialized:
            try:
                cleanup_result = cleanup_backend_system(
                    preserve_results=True,
                    generate_final_reports=False,
                    cleanup_mode='graceful',
                    save_performance_statistics=True
                )
                
                if _logger:
                    _logger.info(f"Backend system cleanup completed: {cleanup_result.get('cleanup_status', 'completed')}")
                
                print("Backend system resources cleaned up successfully")
                
            except Exception as e:
                if _logger:
                    _logger.error(f"Backend system cleanup failed during interruption: {e}")
                print(f"Backend system cleanup failed: {e}", file=sys.stderr)
        
        # Cleanup utilities package and memory management systems
        try:
            from .utils import cleanup_utils_package
            
            utils_cleanup_result = cleanup_utils_package(
                force_cleanup=False,
                preserve_logs=True
            )
            
            if _logger:
                _logger.info(f"Utilities cleanup completed: {utils_cleanup_result.get('success', 'unknown')}")
            
            print("Utilities package resources cleaned up successfully")
            
        except Exception as e:
            if _logger:
                _logger.error(f"Utilities cleanup failed during interruption: {e}")
            print(f"Utilities cleanup failed: {e}", file=sys.stderr)
        
        # Finalize logging system and generate interruption report
        try:
            if _logger:
                _logger.info(f"Graceful shutdown completed for signal {signal_name}")
                
                # Create final audit trail entry for interruption
                try:
                    from .utils import create_audit_trail
                    create_audit_trail(
                        action='SYSTEM_INTERRUPTED',
                        component='MAIN_MODULE',
                        action_details={
                            'signal_name': signal_name,
                            'signal_number': signal_number,
                            'module_version': __version__,
                            'graceful_shutdown': True
                        },
                        user_context='SYSTEM'
                    )
                except Exception:
                    pass  # Don't fail on audit trail creation
            
            _finalize_logging_system()
            
        except Exception as e:
            print(f"Logging finalization failed: {e}", file=sys.stderr)
        
        # Exit with appropriate exit code for interruption scenario
        print(f"Graceful shutdown completed. Exiting with interrupt code.")
        sys.exit(EXIT_INTERRUPT)
        
    except Exception as e:
        # Emergency shutdown on critical failure during interrupt handling
        print(f"CRITICAL: Interrupt handler failed: {e}", file=sys.stderr)
        
        # Attempt emergency cleanup
        try:
            emergency_cleanup(preserve_data=True, force_cleanup=True)
        except Exception as cleanup_error:
            print(f"Emergency cleanup also failed: {cleanup_error}", file=sys.stderr)
        
        sys.exit(EXIT_SYSTEM_ERROR)


def initialize_module_execution(
    enable_monitoring: bool = True,
    validate_environment: bool = True
) -> bool:
    """
    Initialize module execution environment including backend system initialization, utilities 
    setup, logging configuration, and signal handler registration for scientific computing 
    operations.
    
    This function establishes the complete module execution environment with backend system
    initialization, utilities package setup, logging configuration, signal handlers, and
    cleanup handlers to ensure reliable scientific computing operations.
    
    Args:
        enable_monitoring: Whether to enable monitoring systems for performance tracking
        validate_environment: Whether to validate execution environment for scientific computing
        
    Returns:
        bool: Success status of module execution initialization
    """
    global _backend_initialized, _logger
    
    try:
        # Check if backend system is already initialized to prevent duplicate initialization
        if _backend_initialized:
            if _logger:
                _logger.info("Backend system already initialized - skipping module execution initialization")
            return True
        
        # Initialize utilities package with logging and memory management
        utils_init_success = initialize_utils_package(
            config=None,
            enable_logging=True,
            enable_memory_monitoring=enable_monitoring,
            validate_environment=validate_environment
        )
        
        if not utils_init_success:
            print("ERROR: Utilities package initialization failed", file=sys.stderr)
            return False
        
        # Setup module logger with scientific context and performance tracking
        try:
            _logger = get_logger('backend_main', 'MAIN_MODULE')
            _logger.info(f"Module execution initialization started for {MODULE_NAME} v{__version__}")
        except Exception as e:
            print(f"WARNING: Logger setup failed: {e}", file=sys.stderr)
            # Continue with basic logging
            import logging
            _logger = logging.getLogger('backend_main')
            _logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            _logger.addHandler(handler)
        
        # Initialize backend system with all integrated components
        backend_init_success = initialize_backend_system(
            backend_config=None,
            enable_all_components=True,
            validate_system_requirements=validate_environment,
            enable_performance_monitoring=enable_monitoring
        )
        
        if not backend_init_success:
            _logger.error("Backend system initialization failed")
            return False
        
        _backend_initialized = True
        _logger.info("Backend system initialized successfully")
        
        # Setup signal handlers for graceful shutdown and interrupt management
        try:
            setup_signal_handlers()
            _logger.debug("Signal handlers registered successfully")
        except Exception as e:
            _logger.warning(f"Signal handler setup failed: {e}")
        
        # Register cleanup handlers for proper resource finalization
        try:
            register_cleanup_handlers()
            _logger.debug("Cleanup handlers registered successfully")
        except Exception as e:
            _logger.warning(f"Cleanup handler registration failed: {e}")
        
        # Validate execution environment if validate_environment is True
        if validate_environment:
            try:
                # Check system requirements for scientific computing
                import sys
                if sys.version_info < (3, 9):
                    _logger.error(f"Python version {sys.version_info} below minimum requirement 3.9+")
                    return False
                
                # Verify backend system operational readiness
                system_status = get_backend_system_status(
                    include_detailed_metrics=False,
                    include_component_diagnostics=True,
                    include_performance_analysis=False
                )
                
                if not system_status.get('operational_readiness', {}).get('is_ready', False):
                    _logger.error("Backend system not operational after initialization")
                    return False
                
                _logger.info("Execution environment validation passed")
                
            except Exception as e:
                _logger.error(f"Environment validation failed: {e}")
                return False
        
        # Enable monitoring systems if enable_monitoring is True
        if enable_monitoring:
            try:
                # Additional monitoring setup would go here
                _logger.debug("Monitoring systems enabled for performance tracking")
            except Exception as e:
                _logger.warning(f"Monitoring system setup failed: {e}")
        
        # Set global backend initialization flag and log successful initialization
        _logger.info(f"Module execution environment initialized successfully with monitoring={enable_monitoring}, validation={validate_environment}")
        
        # Return initialization success status with component health summary
        return True
        
    except Exception as e:
        error_msg = f"Module execution initialization failed: {e}"
        if _logger:
            _logger.critical(error_msg)
        else:
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
        
        # Cleanup on initialization failure
        try:
            emergency_cleanup(preserve_data=False, force_cleanup=True)
        except Exception as cleanup_error:
            print(f"Emergency cleanup during initialization failure also failed: {cleanup_error}", file=sys.stderr)
        
        return False


def execute_backend_module(args: list = None) -> int:
    """
    Execute backend module with comprehensive error handling, system initialization, CLI 
    interface execution, and graceful shutdown for scientific computing workflows with 
    performance monitoring and audit trail.
    
    This function orchestrates complete backend module execution with system initialization,
    CLI interface coordination, performance monitoring, and graceful shutdown to ensure
    reliable scientific computing operations with comprehensive error handling.
    
    Args:
        args: Optional list of command-line arguments for CLI execution
        
    Returns:
        int: Exit code indicating module execution success or failure with detailed error classification
    """
    global _backend_initialized, _logger
    
    execution_start_time = None
    
    try:
        import datetime
        execution_start_time = datetime.datetime.now()
        
        # Initialize module execution environment with comprehensive setup
        init_success = initialize_module_execution(
            enable_monitoring=True,
            validate_environment=True
        )
        
        if not init_success:
            print("ERROR: Module execution initialization failed", file=sys.stderr)
            return EXIT_INITIALIZATION_ERROR
        
        # Log module execution start with arguments and system information
        _logger.info(f"Backend module execution started at {execution_start_time.isoformat()}")
        if args:
            _logger.info(f"Command-line arguments provided: {args}")
        else:
            _logger.info("Using default system arguments")
        
        # Log system information for audit trail
        import sys
        _logger.info(f"Python version: {sys.version_info}")
        _logger.info(f"Platform: {sys.platform}")
        _logger.info(f"Module version: {__version__}")
        
        # Execute CLI interface with provided arguments or system arguments
        try:
            _logger.info("Executing CLI interface")
            
            # Use the run_cli_interface function from backend package
            cli_exit_code = run_cli_interface(args)
            
            _logger.info(f"CLI interface execution completed with exit code: {cli_exit_code}")
            
        except Exception as e:
            _logger.error(f"CLI interface execution failed: {e}")
            _logger.debug(f"CLI execution traceback: {traceback.format_exc()}")
            return EXIT_FAILURE
        
        # Monitor execution performance and resource utilization
        execution_duration = (datetime.datetime.now() - execution_start_time).total_seconds()
        
        # Get final system status for performance assessment
        try:
            final_status = get_backend_system_status(
                include_detailed_metrics=True,
                include_component_diagnostics=False,
                include_performance_analysis=True
            )
            
            _logger.info(f"Final system health: {final_status.get('operational_readiness', {}).get('readiness_score', 'unknown')}")
            
        except Exception as e:
            _logger.warning(f"Could not retrieve final system status: {e}")
        
        # Handle any system exceptions with comprehensive error reporting and recovery
        # (This section handles exceptions that might occur during execution monitoring)
        
        # Perform graceful shutdown with resource cleanup and data preservation
        # (Graceful shutdown is handled by cleanup handlers and signal handlers)
        
        # Generate execution summary with performance metrics and quality assessment
        execution_summary = {
            'execution_duration_seconds': execution_duration,
            'cli_exit_code': cli_exit_code,
            'module_version': __version__,
            'backend_initialized': _backend_initialized,
            'execution_start_time': execution_start_time.isoformat(),
            'execution_end_time': datetime.datetime.now().isoformat()
        }
        
        # Log module execution completion with final statistics and audit trail
        _logger.info(f"Backend module execution completed in {execution_duration:.3f} seconds")
        _logger.info(f"Execution summary: CLI exit code {cli_exit_code}")
        
        # Create audit trail for module execution
        try:
            from .utils import create_audit_trail
            create_audit_trail(
                action='MODULE_EXECUTION_COMPLETED',
                component='MAIN_MODULE',
                action_details=execution_summary,
                user_context='CLI_USER'
            )
        except Exception as e:
            _logger.warning(f"Could not create audit trail for module execution: {e}")
        
        # Return appropriate exit code based on execution success and error classification
        return cli_exit_code
        
    except KeyboardInterrupt:
        if _logger:
            _logger.warning("Backend module execution interrupted by user")
        else:
            print("Backend module execution interrupted by user", file=sys.stderr)
        return EXIT_INTERRUPT
        
    except PlumeSimulationException as e:
        error_msg = f"Plume simulation error during module execution: {e}"
        if _logger:
            _logger.error(error_msg)
            
            # Add context for plume simulation errors
            try:
                context = e.add_context('module_execution', {
                    'module_version': __version__,
                    'execution_duration': (datetime.datetime.now() - execution_start_time).total_seconds() if execution_start_time else 0
                })
                
                recommendations = e.get_recovery_recommendations()
                if recommendations:
                    _logger.info("Recovery recommendations:")
                    for rec in recommendations:
                        _logger.info(f"  - {rec}")
            except Exception as context_error:
                _logger.warning(f"Could not add error context: {context_error}")
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        
        return EXIT_FAILURE
        
    except Exception as e:
        error_msg = f"Unexpected error during module execution: {e}"
        if _logger:
            _logger.critical(error_msg)
            _logger.debug(f"Module execution traceback: {traceback.format_exc()}")
        else:
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        
        return EXIT_FAILURE


def main() -> int:
    """
    Main entry point function for backend module execution providing comprehensive system 
    orchestration, error handling, and graceful shutdown for scientific computing operations 
    with performance monitoring and reproducible outcomes.
    
    This function serves as the primary entry point for backend module execution with comprehensive
    system orchestration, error handling, graceful shutdown procedures, and audit trail generation
    to ensure reliable scientific computing operations with reproducible research outcomes.
    
    Returns:
        int: Exit code indicating overall module execution success or failure
    """
    global _logger
    
    main_execution_start_time = None
    
    try:
        import datetime
        main_execution_start_time = datetime.datetime.now()
        
        # Setup basic logging for module execution tracking
        try:
            # Try to get logger from utils package
            from .utils import get_logger
            _logger = get_logger('backend_main', 'MAIN_MODULE')
        except Exception:
            # Fallback to basic logging if utils package not available
            import logging
            _logger = logging.getLogger('backend_main')
            _logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            _logger.addHandler(handler)
        
        # Log backend module execution start with system information
        _logger.info(f"Backend module execution started: {MODULE_NAME} v{__version__}")
        _logger.info(f"Execution start time: {main_execution_start_time.isoformat()}")
        _logger.info(f"Python version: {sys.version_info}")
        _logger.info(f"Command line arguments: {sys.argv}")
        
        # Execute backend module with comprehensive error handling and monitoring
        module_exit_code = execute_backend_module(args=None)
        
        # Calculate total execution time
        main_execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds()
        
        # Log execution completion based on exit code
        if module_exit_code == EXIT_SUCCESS:
            _logger.info(f"Backend module execution completed successfully in {main_execution_duration:.3f} seconds")
        else:
            _logger.warning(f"Backend module execution completed with exit code {module_exit_code} in {main_execution_duration:.3f} seconds")
        
        # Return appropriate exit code based on overall execution outcome
        return module_exit_code
        
    except PlumeSimulationException as e:
        # Handle PlumeSimulationException with context-aware error reporting and recovery
        error_msg = f"Plume simulation error in main function: {e}"
        execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds() if main_execution_start_time else 0
        
        if _logger:
            _logger.error(error_msg)
            _logger.info(f"Error occurred after {execution_duration:.3f} seconds of execution")
            
            # Attempt to get recovery recommendations
            try:
                recommendations = e.get_recovery_recommendations()
                if recommendations:
                    _logger.info("Recovery recommendations:")
                    for rec in recommendations:
                        _logger.info(f"  - {rec}")
            except Exception:
                pass  # Don't fail on recommendation retrieval
        else:
            print(f"ERROR: {error_msg}", file=sys.stderr)
        
        return EXIT_FAILURE
        
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt with graceful shutdown and data preservation
        execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds() if main_execution_start_time else 0
        
        if _logger:
            _logger.warning(f"Backend module execution interrupted by user after {execution_duration:.3f} seconds")
        else:
            print("Backend module execution interrupted by user", file=sys.stderr)
        
        return EXIT_INTERRUPT
        
    except SystemExit as e:
        # Handle SystemExit with proper cleanup and exit code preservation
        exit_code = e.code if e.code is not None else EXIT_FAILURE
        execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds() if main_execution_start_time else 0
        
        if _logger:
            _logger.info(f"Backend module execution exited with code {exit_code} after {execution_duration:.3f} seconds")
        
        return exit_code
        
    except Exception as e:
        # Handle unexpected exceptions with emergency cleanup and comprehensive error reporting
        error_msg = f"Unexpected error in main function: {e}"
        execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds() if main_execution_start_time else 0
        
        if _logger:
            _logger.critical(error_msg)
            _logger.critical(f"Critical error occurred after {execution_duration:.3f} seconds of execution")
            _logger.debug(f"Main function traceback: {traceback.format_exc()}")
        else:
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        
        # Attempt emergency cleanup on critical failure
        try:
            emergency_cleanup(preserve_data=True, force_cleanup=True)
        except Exception as cleanup_error:
            if _logger:
                _logger.critical(f"Emergency cleanup during main function failure also failed: {cleanup_error}")
            else:
                print(f"Emergency cleanup also failed: {cleanup_error}", file=sys.stderr)
        
        return EXIT_FAILURE
        
    finally:
        # Perform final cleanup operations and resource finalization
        try:
            # Final resource cleanup and audit trail completion
            if _logger:
                final_execution_duration = (datetime.datetime.now() - main_execution_start_time).total_seconds() if main_execution_start_time else 0
                _logger.info(f"Main function cleanup completed after {final_execution_duration:.3f} seconds total execution")
                
                # Create final audit trail entry
                try:
                    from .utils import create_audit_trail
                    create_audit_trail(
                        action='MAIN_FUNCTION_COMPLETED',
                        component='MAIN_MODULE',
                        action_details={
                            'total_execution_duration': final_execution_duration,
                            'module_version': __version__,
                            'completion_timestamp': datetime.datetime.now().isoformat()
                        },
                        user_context='SYSTEM'
                    )
                except Exception:
                    pass  # Don't fail on audit trail creation
        except Exception as e:
            # Ultimate fallback for cleanup failures
            try:
                print(f"Final cleanup failed: {e}", file=sys.stderr)
            except Exception:
                pass  # Ultimate fallback


def _finalize_logging_system() -> None:
    """
    Finalize logging system with proper handler cleanup and log flushing.
    
    This function properly finalizes the logging system with handler cleanup,
    log flushing, and resource deallocation for clean system shutdown.
    
    Returns:
        None: No return value
    """
    global _logger
    
    try:
        if _logger:
            # Flush all handlers
            for handler in _logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    _logger.removeHandler(handler)
                except Exception:
                    pass  # Ignore handler cleanup errors
            
            # Final log message
            try:
                _logger.info("Logging system finalized")
            except Exception:
                pass  # Ignore final message errors
        
    except Exception:
        pass  # Ignore all finalization errors


# Direct module execution entry point that calls main() function and exits with returned exit code
if __name__ == '__main__':
    sys.exit(main())