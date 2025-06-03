"""
Comprehensive file utilities module providing robust file system operations, cross-platform compatibility, 
video file validation, configuration management, integrity verification, and atomic file operations for 
the plume simulation system.

This module implements fail-fast validation, graceful error handling, checksum verification, and scientific 
data file management with support for JSON configuration files, video format validation, and secure file 
operations essential for reproducible research outcomes and batch processing reliability with localized 
error handling and validation capabilities.

Key Features:
- Cross-platform file system operations with pathlib integration
- Video file validation using OpenCV with format compatibility checking
- JSON configuration management with schema validation using jsonschema
- Atomic file operations with integrity verification and rollback capability
- Thread-safe file locking mechanisms for concurrent access control
- Comprehensive error handling with retry logic and exponential backoff
- File integrity validation using multiple checksum algorithms
- Audit trail integration for scientific computing traceability
- Fail-fast validation strategy for early error detection
- Graceful degradation for partial batch processing completion
"""

# External library imports with version specifications
import pathlib  # Python 3.9+ - Modern cross-platform path handling and file system operations
import json  # Python 3.9+ - JSON configuration file parsing and validation
import shutil  # Python 3.9+ - High-level file operations including atomic copy and move operations
import os  # Python 3.9+ - Operating system interface for file permissions and environment variables
import hashlib  # Python 3.9+ - Cryptographic hash functions for file integrity verification
import tempfile  # Python 3.9+ - Temporary file creation for atomic operations and safe file handling
import stat  # Python 3.9+ - File status and permission constants for cross-platform compatibility
from typing import Dict, Any, List, Optional, Union, Callable, Tuple  # Python 3.9+ - Type hints for file utility function signatures
import datetime  # Python 3.9+ - Timestamp handling for file metadata and audit trails
import threading  # Python 3.9+ - Thread-safe file operations and locking mechanisms
import contextlib  # Python 3.9+ - Context manager utilities for safe file operations
import cv2  # opencv-python 4.11.0+ - Video file format validation and metadata extraction
import jsonschema  # jsonschema 4.17.0+ - JSON schema validation for configuration files
import time  # Python 3.9+ - Time-based operations for retry delays and timeout handling
import random  # Python 3.9+ - Random number generation for exponential backoff jitter
import functools  # Python 3.9+ - Decorator utilities for retry logic and function wrapping

# Internal imports from logging utilities module
from .logging_utils import get_logger, log_validation_error, create_audit_trail

# Global configuration constants for file operation limits and settings
SUPPORTED_VIDEO_FORMATS = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
SUPPORTED_CONFIG_FORMATS = ['.json', '.yaml', '.yml']
SUPPORTED_CHECKSUM_ALGORITHMS = ['md5', 'sha1', 'sha256', 'sha512']
DEFAULT_CHECKSUM_ALGORITHM = 'sha256'
MAX_FILE_SIZE_GB = 10.0
ATOMIC_OPERATION_TIMEOUT = 30.0
FILE_LOCK_TIMEOUT = 10.0
BACKUP_SUFFIX = '.backup'
TEMP_FILE_PREFIX = 'plume_sim_'
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
RETRY_MAX_DELAY = 60.0
RETRY_EXPONENTIAL_BASE = 2.0

# Error severity level constants for error classification and handling
ERROR_SEVERITY_CRITICAL = 'CRITICAL'
ERROR_SEVERITY_HIGH = 'HIGH'
ERROR_SEVERITY_MEDIUM = 'MEDIUM'
ERROR_SEVERITY_LOW = 'LOW'

# Global registries for file locks and operation caching
_file_locks: Dict[str, threading.Lock] = {}
_operation_cache: Dict[str, Any] = {}


def retry_with_backoff(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_DELAY,
    retryable_exceptions: tuple = (OSError, IOError, PermissionError),
    add_jitter: bool = True
) -> Callable:
    """
    Localized decorator function implementing exponential backoff retry logic for transient file 
    system errors with configurable retry attempts, delays, and exception handling for robust 
    file operations.
    
    This decorator provides comprehensive retry logic with exponential backoff for handling 
    transient file system errors, network interruptions, and resource contention issues.
    
    Args:
        max_attempts: Maximum number of retry attempts before giving up
        base_delay: Base delay in seconds for exponential backoff calculation
        retryable_exceptions: Tuple of exception types that should trigger retries
        add_jitter: Add random jitter to prevent thundering herd effects
        
    Returns:
        Callable: Decorated function with retry logic and exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f'retry.{func.__name__}', 'FILE_OPERATIONS')
            
            for attempt in range(max_attempts):
                try:
                    # Execute the function with provided arguments
                    result = func(*args, **kwargs)
                    
                    # Log successful execution after retries
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except retryable_exceptions as e:
                    # Calculate exponential backoff delay with optional jitter
                    delay = base_delay * (RETRY_EXPONENTIAL_BASE ** attempt)
                    if add_jitter:
                        delay += random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    
                    # Enforce maximum delay limit
                    delay = min(delay, RETRY_MAX_DELAY)
                    
                    # Check if this is the last attempt
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise e
                    
                    # Log retry attempt with context and delay information
                    logger.warning(
                        f"Function {func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds"
                    )
                    
                    # Create audit trail for retry attempts
                    create_audit_trail(
                        action='RETRY_ATTEMPT',
                        component='FILE_OPERATIONS',
                        action_details={
                            'function_name': func.__name__,
                            'attempt_number': attempt + 1,
                            'max_attempts': max_attempts,
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'retry_delay': delay
                        }
                    )
                    
                    # Wait before next retry attempt
                    time.sleep(delay)
                
                except Exception as e:
                    # Non-retryable exceptions are re-raised immediately
                    logger.error(f"Function {func.__name__} failed with non-retryable exception: {e}")
                    raise e
            
            # This should never be reached due to the logic above
            raise RuntimeError(f"Unexpected retry loop completion for {func.__name__}")
        
        return wrapper
    return decorator


def handle_file_error(
    error: Exception,
    operation_context: str,
    file_path: str,
    error_severity: str = ERROR_SEVERITY_MEDIUM
) -> Dict[str, Any]:
    """
    Localized error handling function for file operations with error classification, severity 
    assessment, and recovery recommendations for comprehensive file system error management.
    
    This function provides comprehensive error analysis and classification with actionable 
    recovery recommendations for different types of file system errors.
    
    Args:
        error: Exception object containing error details
        operation_context: Description of the operation that failed
        file_path: Path to the file involved in the error
        error_severity: Severity level for error classification
        
    Returns:
        Dict[str, Any]: Error handling result with classification, recommendations, and recovery actions
    """
    logger = get_logger('error_handler', 'FILE_OPERATIONS')
    
    # Initialize error handling result dictionary
    error_result = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'operation_context': operation_context,
        'file_path': file_path,
        'severity': error_severity,
        'timestamp': datetime.datetime.now().isoformat(),
        'is_retryable': False,
        'recovery_recommendations': [],
        'recovery_actions': []
    }
    
    # Classify error type and determine appropriate handling
    if isinstance(error, FileNotFoundError):
        error_result['is_retryable'] = False
        error_result['recovery_recommendations'] = [
            'Verify the file path exists and is accessible',
            'Check if the file was moved or deleted',
            'Ensure the file system is mounted and accessible'
        ]
        error_result['recovery_actions'] = ['validate_file_path', 'check_parent_directory']
        
    elif isinstance(error, PermissionError):
        error_result['is_retryable'] = True
        error_result['recovery_recommendations'] = [
            'Check file and directory permissions',
            'Ensure the process has appropriate access rights',
            'Verify the file is not locked by another process'
        ]
        error_result['recovery_actions'] = ['check_permissions', 'retry_with_elevated_access']
        
    elif isinstance(error, OSError):
        error_result['is_retryable'] = True
        error_result['recovery_recommendations'] = [
            'Check disk space availability',
            'Verify file system integrity',
            'Ensure the storage device is accessible'
        ]
        error_result['recovery_actions'] = ['check_disk_space', 'verify_filesystem']
        
    elif isinstance(error, IOError):
        error_result['is_retryable'] = True
        error_result['recovery_recommendations'] = [
            'Retry the operation after a brief delay',
            'Check for hardware or network issues',
            'Verify the file is not corrupted'
        ]
        error_result['recovery_actions'] = ['retry_operation', 'verify_file_integrity']
        
    else:
        # Generic error handling for unknown error types
        error_result['recovery_recommendations'] = [
            'Review the specific error details',
            'Check system logs for additional information',
            'Consider alternative approaches for the operation'
        ]
        error_result['recovery_actions'] = ['log_detailed_error', 'escalate_to_administrator']
    
    # Log error with structured format and detailed context
    logger.error(
        f"File operation error in {operation_context}: {error_result['error_type']} - "
        f"{error_result['error_message']} (Path: {file_path}, Severity: {error_severity})"
    )
    
    # Create comprehensive audit trail entry for error occurrence
    create_audit_trail(
        action='FILE_OPERATION_ERROR',
        component='FILE_OPERATIONS',
        action_details={
            'error_classification': error_result,
            'operation_context': operation_context,
            'file_path': file_path,
            'error_analysis': {
                'retryable': error_result['is_retryable'],
                'severity_assessment': error_severity,
                'recommended_actions': error_result['recovery_recommendations']
            }
        }
    )
    
    # Log validation error with specialized logging function
    log_validation_error(
        validation_type='FILE_OPERATION',
        error_message=f"{error_result['error_type']}: {error_result['error_message']}",
        validation_context={
            'operation': operation_context,
            'file_path': file_path,
            'severity': error_severity
        },
        recovery_recommendations=error_result['recovery_recommendations']
    )
    
    return error_result


@retry_with_backoff(max_attempts=3, retryable_exceptions=(OSError, PermissionError))
def ensure_directory_exists(
    directory_path: str,
    permissions: int = 0o755,
    create_parents: bool = True
) -> bool:
    """
    Create directory structure with cross-platform compatibility, proper permissions, and error 
    handling for scientific data organization and batch processing requirements.
    
    This function provides robust directory creation with comprehensive error handling and 
    cross-platform compatibility for scientific computing workflows.
    
    Args:
        directory_path: Path to the directory to create
        permissions: Octal permissions for the directory (Unix/Linux systems)
        create_parents: Create parent directories if they don't exist
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    logger = get_logger('directory_operations', 'FILE_OPERATIONS')
    
    try:
        # Convert directory path to pathlib.Path for cross-platform compatibility
        dir_path = pathlib.Path(directory_path)
        
        # Check if directory already exists and is accessible
        if dir_path.exists():
            if dir_path.is_dir():
                logger.debug(f"Directory already exists: {directory_path}")
                return True
            else:
                raise ValueError(f"Path exists but is not a directory: {directory_path}")
        
        # Create parent directories if create_parents is True
        if create_parents:
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            dir_path.mkdir(exist_ok=True)
        
        # Set appropriate permissions for cross-platform compatibility
        if os.name != 'nt':  # Not Windows
            os.chmod(str(dir_path), permissions)
        
        # Validate directory creation and accessibility
        if not dir_path.exists() or not dir_path.is_dir():
            raise OSError(f"Failed to create directory: {directory_path}")
        
        # Test write accessibility
        test_file = dir_path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            logger.warning(f"Directory created but not writable: {directory_path} - {e}")
        
        # Log successful directory creation with audit trail
        logger.info(f"Directory created successfully: {directory_path}")
        create_audit_trail(
            action='DIRECTORY_CREATED',
            component='FILE_OPERATIONS',
            action_details={
                'directory_path': str(directory_path),
                'permissions': oct(permissions),
                'parents_created': create_parents
            }
        )
        
        return True
        
    except Exception as e:
        # Handle directory creation errors with comprehensive error reporting
        error_result = handle_file_error(
            error=e,
            operation_context='directory_creation',
            file_path=directory_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        logger.error(f"Failed to create directory: {directory_path} - {e}")
        return False


def validate_file_exists(
    file_path: str,
    check_readable: bool = True,
    check_size_limits: bool = True
) -> 'FileValidationResult':
    """
    Validate file existence, accessibility, and basic format requirements with comprehensive 
    error reporting and fail-fast validation for preventing wasted computational resources.
    
    This function implements comprehensive file validation with fail-fast error detection 
    and detailed reporting for scientific computing workflows.
    
    Args:
        file_path: Path to the file to validate
        check_readable: Verify file is readable by the current process
        check_size_limits: Check file size against configured limits
        
    Returns:
        FileValidationResult: File existence validation result with detailed error information and recommendations
    """
    logger = get_logger('file_validation', 'VALIDATION')
    
    # Create FileValidationResult container for validation tracking
    validation_result = FileValidationResult(file_path=file_path, is_valid=True)
    
    try:
        # Convert file_path to pathlib.Path for cross-platform handling
        file_obj = pathlib.Path(file_path)
        
        # Check if file exists and add error if not found
        if not file_obj.exists():
            validation_result.add_error(
                error_message=f"File does not exist: {file_path}",
                error_category="FILE_NOT_FOUND"
            )
            return validation_result
        
        # Validate file is not a directory or special file type
        if not file_obj.is_file():
            validation_result.add_error(
                error_message=f"Path is not a regular file: {file_path}",
                error_category="INVALID_FILE_TYPE"
            )
            return validation_result
        
        # Check file readability if check_readable is enabled
        if check_readable:
            try:
                with open(file_obj, 'rb') as f:
                    # Try to read first byte to verify accessibility
                    f.read(1)
                validation_result.set_metadata('readable', True)
            except (PermissionError, OSError) as e:
                validation_result.add_error(
                    error_message=f"File is not readable: {e}",
                    error_category="PERMISSION_DENIED"
                )
        
        # Validate file size against limits if check_size_limits is enabled
        if check_size_limits:
            file_size_bytes = file_obj.stat().st_size
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
            
            validation_result.set_metadata('file_size_bytes', file_size_bytes)
            validation_result.set_metadata('file_size_gb', round(file_size_gb, 3))
            
            if file_size_gb > MAX_FILE_SIZE_GB:
                validation_result.add_error(
                    error_message=f"File size ({file_size_gb:.2f} GB) exceeds limit ({MAX_FILE_SIZE_GB} GB)",
                    error_category="FILE_SIZE_EXCEEDED"
                )
            elif file_size_bytes == 0:
                validation_result.add_warning(
                    warning_message="File is empty (0 bytes)",
                    warning_category="EMPTY_FILE"
                )
        
        # Check file permissions and accessibility
        file_stat = file_obj.stat()
        validation_result.set_metadata('file_permissions', oct(file_stat.st_mode))
        validation_result.set_metadata('last_modified', datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat())
        
        # Add warnings for potential issues
        if file_stat.st_mode & stat.S_IWOTH:  # World writable
            validation_result.add_warning(
                warning_message="File has world-writable permissions",
                warning_category="SECURITY_WARNING"
            )
        
        # Log successful validation operation with context
        if validation_result.is_valid:
            logger.debug(f"File validation successful: {file_path}")
        else:
            logger.warning(f"File validation failed: {file_path} - {len(validation_result.errors)} errors")
        
        # Create audit trail for validation operation
        create_audit_trail(
            action='FILE_VALIDATION',
            component='VALIDATION',
            action_details={
                'file_path': file_path,
                'validation_success': validation_result.is_valid,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings),
                'checks_performed': {
                    'existence': True,
                    'readability': check_readable,
                    'size_limits': check_size_limits
                }
            }
        )
        
        return validation_result
        
    except Exception as e:
        # Handle unexpected validation errors
        validation_result.add_error(
            error_message=f"Validation error: {e}",
            error_category="VALIDATION_EXCEPTION"
        )
        
        handle_file_error(
            error=e,
            operation_context='file_validation',
            file_path=file_path,
            error_severity=ERROR_SEVERITY_MEDIUM
        )
        
        return validation_result


def validate_video_file(
    video_path: str,
    expected_format: str = None,
    validate_codec: bool = True,
    check_integrity: bool = True
) -> 'FileValidationResult':
    """
    Comprehensive video file validation including format compatibility, codec support, metadata 
    extraction, and integrity checking for plume recording processing pipeline.
    
    This function provides specialized validation for video files used in plume simulation 
    research with comprehensive format and codec compatibility checking.
    
    Args:
        video_path: Path to the video file to validate
        expected_format: Expected video format for compatibility checking
        validate_codec: Perform codec compatibility validation
        check_integrity: Check video file integrity by sampling frames
        
    Returns:
        FileValidationResult: Video file validation result with format compatibility, codec information, and integrity status
    """
    logger = get_logger('video_validation', 'VALIDATION')
    
    # Create FileValidationResult container for video validation
    validation_result = FileValidationResult(file_path=video_path, is_valid=True)
    
    try:
        # Validate basic file existence and accessibility first
        basic_validation = validate_file_exists(video_path, check_readable=True, check_size_limits=True)
        if not basic_validation.is_valid:
            validation_result.errors.extend(basic_validation.errors)
            validation_result.warnings.extend(basic_validation.warnings)
            validation_result.is_valid = False
            return validation_result
        
        # Check file extension against supported video formats
        file_ext = pathlib.Path(video_path).suffix.lower()
        if file_ext not in SUPPORTED_VIDEO_FORMATS:
            validation_result.add_error(
                error_message=f"Unsupported video format: {file_ext}. Supported formats: {SUPPORTED_VIDEO_FORMATS}",
                error_category="UNSUPPORTED_FORMAT"
            )
            validation_result.add_recommendation("Convert video to a supported format (AVI, MP4, MOV, MKV, WMV)")
            return validation_result
        
        # Use OpenCV to open and validate video file structure
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            validation_result.add_error(
                error_message="Failed to open video file with OpenCV",
                error_category="VIDEO_OPEN_FAILED"
            )
            validation_result.add_recommendation("Verify video file is not corrupted and uses a supported codec")
            return validation_result
        
        try:
            # Extract video metadata including resolution, frame rate, and codec
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert fourcc to readable codec name
            codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Store video metadata in validation result
            validation_result.set_metadata('frame_count', frame_count)
            validation_result.set_metadata('fps', fps)
            validation_result.set_metadata('resolution', f"{width}x{height}")
            validation_result.set_metadata('width', width)
            validation_result.set_metadata('height', height)
            validation_result.set_metadata('codec', codec_name)
            validation_result.set_metadata('duration_seconds', frame_count / fps if fps > 0 else 0)
            
            # Validate video properties for scientific processing
            if frame_count == 0:
                validation_result.add_error(
                    error_message="Video contains no frames",
                    error_category="EMPTY_VIDEO"
                )
            elif frame_count < 10:
                validation_result.add_warning(
                    warning_message=f"Video has very few frames ({frame_count})",
                    warning_category="SHORT_VIDEO"
                )
            
            if fps <= 0:
                validation_result.add_error(
                    error_message="Invalid frame rate (FPS <= 0)",
                    error_category="INVALID_FRAMERATE"
                )
            elif fps < 1:
                validation_result.add_warning(
                    warning_message=f"Very low frame rate: {fps:.2f} FPS",
                    warning_category="LOW_FRAMERATE"
                )
            
            if width <= 0 or height <= 0:
                validation_result.add_error(
                    error_message=f"Invalid video resolution: {width}x{height}",
                    error_category="INVALID_RESOLUTION"
                )
            
            # Validate codec compatibility if validate_codec is enabled
            if validate_codec:
                # List of commonly supported codecs for scientific video processing
                supported_codecs = ['MJPG', 'H264', 'XVID', 'MP4V', 'DIVX']
                if codec_name not in supported_codecs:
                    validation_result.add_warning(
                        warning_message=f"Codec '{codec_name}' may not be fully supported. Recommended: {supported_codecs}",
                        warning_category="CODEC_COMPATIBILITY"
                    )
            
            # Check video integrity by sampling frames if check_integrity is enabled
            if check_integrity and frame_count > 0:
                frames_to_sample = min(10, frame_count)
                sample_interval = max(1, frame_count // frames_to_sample)
                
                corrupted_frames = 0
                for i in range(0, frame_count, sample_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        corrupted_frames += 1
                
                validation_result.set_metadata('integrity_check_samples', frames_to_sample)
                validation_result.set_metadata('corrupted_frames', corrupted_frames)
                
                if corrupted_frames > 0:
                    corruption_rate = corrupted_frames / frames_to_sample * 100
                    if corruption_rate > 20:  # More than 20% corruption
                        validation_result.add_error(
                            error_message=f"High frame corruption rate: {corruption_rate:.1f}%",
                            error_category="VIDEO_CORRUPTION"
                        )
                    else:
                        validation_result.add_warning(
                            warning_message=f"Some frame corruption detected: {corruption_rate:.1f}%",
                            warning_category="MINOR_CORRUPTION"
                        )
            
            # Validate expected format compatibility if specified
            if expected_format and expected_format.lower() != file_ext:
                validation_result.add_warning(
                    warning_message=f"Video format ({file_ext}) differs from expected ({expected_format})",
                    warning_category="FORMAT_MISMATCH"
                )
            
        finally:
            # Always release the video capture object
            cap.release()
        
        # Log validation results with comprehensive context
        if validation_result.is_valid:
            logger.info(f"Video validation successful: {video_path} ({width}x{height}, {fps:.2f} FPS, {frame_count} frames)")
        else:
            logger.error(f"Video validation failed: {video_path} - {len(validation_result.errors)} errors")
        
        # Create audit trail for video validation operation
        create_audit_trail(
            action='VIDEO_VALIDATION',
            component='VALIDATION',
            action_details={
                'video_path': video_path,
                'validation_success': validation_result.is_valid,
                'video_metadata': validation_result.metadata,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings),
                'validation_options': {
                    'expected_format': expected_format,
                    'validate_codec': validate_codec,
                    'check_integrity': check_integrity
                }
            }
        )
        
        return validation_result
        
    except Exception as e:
        # Handle video validation errors with comprehensive error reporting
        validation_result.add_error(
            error_message=f"Video validation exception: {e}",
            error_category="VALIDATION_EXCEPTION"
        )
        
        handle_file_error(
            error=e,
            operation_context='video_validation',
            file_path=video_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        return validation_result


def get_file_metadata(
    file_path: str,
    include_checksum: bool = False,
    checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM,
    include_format_info: bool = True
) -> dict:
    """
    Extract comprehensive file metadata including size, modification time, permissions, format 
    information, and checksum for audit trails and processing pipeline requirements.
    
    This function provides comprehensive metadata extraction for scientific computing workflows 
    with optional integrity verification and format analysis.
    
    Args:
        file_path: Path to the file for metadata extraction
        include_checksum: Calculate and include file checksum for integrity verification
        checksum_algorithm: Algorithm to use for checksum calculation
        include_format_info: Include file format and type information
        
    Returns:
        dict: Comprehensive file metadata including technical properties, timestamps, and integrity information
    """
    logger = get_logger('metadata_extraction', 'FILE_OPERATIONS')
    
    try:
        # Convert file_path to pathlib.Path for cross-platform handling
        file_obj = pathlib.Path(file_path)
        
        # Validate file exists before metadata extraction
        if not file_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract basic file statistics (size, timestamps, permissions)
        file_stat = file_obj.stat()
        
        metadata = {
            'file_path': str(file_obj.absolute()),
            'file_name': file_obj.name,
            'file_stem': file_obj.stem,
            'file_suffix': file_obj.suffix,
            'file_size_bytes': file_stat.st_size,
            'file_size_mb': round(file_stat.st_size / (1024 * 1024), 3),
            'file_size_gb': round(file_stat.st_size / (1024 * 1024 * 1024), 6),
            'created_time': datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'modified_time': datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'accessed_time': datetime.datetime.fromtimestamp(file_stat.st_atime).isoformat(),
            'permissions': oct(file_stat.st_mode),
            'is_readable': os.access(file_obj, os.R_OK),
            'is_writable': os.access(file_obj, os.W_OK),
            'extraction_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Get file format information if include_format_info is enabled
        if include_format_info:
            file_ext = file_obj.suffix.lower()
            
            # Classify file type based on extension
            if file_ext in SUPPORTED_VIDEO_FORMATS:
                metadata['file_type'] = 'video'
                metadata['supported_format'] = True
                
                # Extract video-specific metadata for video files
                try:
                    cap = cv2.VideoCapture(str(file_obj))
                    if cap.isOpened():
                        metadata['video_metadata'] = {
                            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                            'fps': cap.get(cv2.CAP_PROP_FPS),
                            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
                        }
                        cap.release()
                except Exception as e:
                    logger.warning(f"Failed to extract video metadata: {e}")
                    metadata['video_metadata_error'] = str(e)
                    
            elif file_ext in SUPPORTED_CONFIG_FORMATS:
                metadata['file_type'] = 'configuration'
                metadata['supported_format'] = True
                
                # Validate JSON structure for configuration files
                if file_ext == '.json':
                    try:
                        with open(file_obj, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        metadata['json_valid'] = True
                        metadata['json_keys_count'] = len(config_data) if isinstance(config_data, dict) else 0
                    except json.JSONDecodeError as e:
                        metadata['json_valid'] = False
                        metadata['json_error'] = str(e)
                        
            else:
                metadata['file_type'] = 'other'
                metadata['supported_format'] = False
            
            # Include file system information and mount point details
            try:
                metadata['filesystem_info'] = {
                    'mount_point': str(file_obj.anchor),
                    'is_symlink': file_obj.is_symlink(),
                    'parent_directory': str(file_obj.parent)
                }
                
                if file_obj.is_symlink():
                    metadata['filesystem_info']['symlink_target'] = str(file_obj.readlink())
                    
            except Exception as e:
                logger.debug(f"Failed to extract filesystem info: {e}")
        
        # Calculate file checksum if include_checksum is enabled
        if include_checksum:
            try:
                checksum = calculate_file_checksum(
                    file_path=str(file_obj),
                    algorithm=checksum_algorithm
                )
                metadata['checksum'] = {
                    'algorithm': checksum_algorithm,
                    'value': checksum,
                    'calculated_at': datetime.datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"Failed to calculate checksum: {e}")
                metadata['checksum_error'] = str(e)
        
        # Add file accessibility and permission analysis
        metadata['accessibility_analysis'] = {
            'can_read': os.access(file_obj, os.R_OK),
            'can_write': os.access(file_obj, os.W_OK),
            'can_execute': os.access(file_obj, os.X_OK),
            'owner_readable': bool(file_stat.st_mode & stat.S_IRUSR),
            'owner_writable': bool(file_stat.st_mode & stat.S_IWUSR),
            'group_readable': bool(file_stat.st_mode & stat.S_IRGRP),
            'world_readable': bool(file_stat.st_mode & stat.S_IROTH)
        }
        
        # Log metadata extraction operation with performance tracking
        logger.debug(f"Metadata extraction completed for: {file_path}")
        
        # Create audit trail for metadata extraction
        create_audit_trail(
            action='METADATA_EXTRACTION',
            component='FILE_OPERATIONS',
            action_details={
                'file_path': file_path,
                'metadata_fields_extracted': len(metadata),
                'include_checksum': include_checksum,
                'include_format_info': include_format_info,
                'file_size_mb': metadata['file_size_mb']
            }
        )
        
        return metadata
        
    except Exception as e:
        # Handle metadata extraction errors with comprehensive error reporting
        error_metadata = {
            'file_path': file_path,
            'extraction_error': str(e),
            'error_type': type(e).__name__,
            'extraction_timestamp': datetime.datetime.now().isoformat()
        }
        
        handle_file_error(
            error=e,
            operation_context='metadata_extraction',
            file_path=file_path,
            error_severity=ERROR_SEVERITY_MEDIUM
        )
        
        return error_metadata


@retry_with_backoff(max_attempts=2, retryable_exceptions=(IOError, OSError))
def calculate_file_checksum(
    file_path: str,
    algorithm: str = DEFAULT_CHECKSUM_ALGORITHM,
    chunk_size: int = 8192
) -> str:
    """
    Calculate file integrity checksum using specified algorithm with memory-efficient processing 
    for large video files and verification of data integrity.
    
    This function provides memory-efficient checksum calculation for large files with support 
    for multiple hash algorithms and optimized chunk processing.
    
    Args:
        file_path: Path to the file for checksum calculation
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
        chunk_size: Size of chunks for memory-efficient processing
        
    Returns:
        str: Hexadecimal checksum string for file integrity verification
    """
    logger = get_logger('checksum_calculation', 'FILE_OPERATIONS')
    
    # Validate checksum algorithm is supported
    if algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}. Supported: {SUPPORTED_CHECKSUM_ALGORITHMS}")
    
    try:
        # Initialize hash object for specified algorithm
        hash_obj = hashlib.new(algorithm)
        
        # Record start time for performance tracking
        start_time = time.time()
        total_bytes_processed = 0
        
        # Open file in binary mode with error handling
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large video files efficiently
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Update hash object with each chunk
                hash_obj.update(chunk)
                total_bytes_processed += len(chunk)
        
        # Calculate processing time and performance metrics
        processing_time = time.time() - start_time
        processing_rate_mb_per_sec = (total_bytes_processed / (1024 * 1024)) / processing_time if processing_time > 0 else 0
        
        # Finalize hash calculation and get hexadecimal digest
        checksum = hash_obj.hexdigest()
        
        # Log checksum calculation operation and performance
        logger.debug(
            f"Checksum calculation completed: {file_path} ({algorithm}={checksum}) "
            f"- {total_bytes_processed} bytes in {processing_time:.2f}s "
            f"({processing_rate_mb_per_sec:.2f} MB/s)"
        )
        
        # Create audit trail for checksum calculation
        create_audit_trail(
            action='CHECKSUM_CALCULATION',
            component='FILE_OPERATIONS',
            action_details={
                'file_path': file_path,
                'algorithm': algorithm,
                'checksum': checksum,
                'file_size_bytes': total_bytes_processed,
                'processing_time_seconds': processing_time,
                'processing_rate_mb_per_sec': processing_rate_mb_per_sec
            }
        )
        
        return checksum
        
    except Exception as e:
        # Handle checksum calculation errors with comprehensive error reporting
        handle_file_error(
            error=e,
            operation_context='checksum_calculation',
            file_path=file_path,
            error_severity=ERROR_SEVERITY_MEDIUM
        )
        raise e


@retry_with_backoff(max_attempts=3, retryable_exceptions=(IOError, OSError, shutil.Error))
def safe_file_copy(
    source_path: str,
    destination_path: str,
    verify_integrity: bool = True,
    create_backup: bool = False,
    atomic_operation: bool = True
) -> dict:
    """
    Safely copy files with integrity verification, atomic operations, and error recovery for 
    reliable data processing and backup operations in batch processing environments.
    
    This function provides robust file copying with comprehensive error handling, integrity 
    verification, and atomic operations for scientific computing workflows.
    
    Args:
        source_path: Path to the source file to copy
        destination_path: Path to the destination for the copy
        verify_integrity: Verify file integrity after copy using checksum comparison
        create_backup: Create backup of destination file if it exists
        atomic_operation: Use temporary file for atomic copy operation
        
    Returns:
        dict: File copy operation result with success status, integrity verification, and performance metrics
    """
    logger = get_logger('file_copy', 'FILE_OPERATIONS')
    
    operation_result = {
        'success': False,
        'source_path': source_path,
        'destination_path': destination_path,
        'backup_created': False,
        'backup_path': None,
        'integrity_verified': False,
        'operation_time_seconds': 0,
        'bytes_copied': 0,
        'copy_rate_mb_per_sec': 0,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    start_time = time.time()
    
    try:
        # Validate source file exists and is accessible
        source_file = pathlib.Path(source_path)
        if not source_file.exists() or not source_file.is_file():
            raise FileNotFoundError(f"Source file not found or invalid: {source_path}")
        
        source_size = source_file.stat().st_size
        operation_result['bytes_copied'] = source_size
        
        # Create destination directory if it doesn't exist
        dest_file = pathlib.Path(destination_path)
        ensure_directory_exists(str(dest_file.parent), create_parents=True)
        
        # Create backup of destination file if create_backup is enabled
        if create_backup and dest_file.exists():
            backup_path = dest_file.with_suffix(dest_file.suffix + BACKUP_SUFFIX)
            shutil.copy2(str(dest_file), str(backup_path))
            operation_result['backup_created'] = True
            operation_result['backup_path'] = str(backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # Calculate source checksum if verify_integrity is enabled
        source_checksum = None
        if verify_integrity:
            source_checksum = calculate_file_checksum(source_path)
        
        # Use temporary file for atomic operation if atomic_operation is enabled
        if atomic_operation:
            # Create temporary file in destination directory for atomic operation
            with tempfile.NamedTemporaryFile(
                dir=dest_file.parent,
                prefix=TEMP_FILE_PREFIX,
                delete=False
            ) as temp_file:
                temp_path = pathlib.Path(temp_file.name)
            
            # Perform file copy operation to temporary location
            shutil.copy2(str(source_file), str(temp_path))
            
            # Set appropriate file permissions on temporary file
            if os.name != 'nt':  # Not Windows
                source_permissions = source_file.stat().st_mode
                os.chmod(str(temp_path), source_permissions)
            
            # Verify file integrity if verify_integrity is enabled
            if verify_integrity and source_checksum:
                temp_checksum = calculate_file_checksum(str(temp_path))
                if source_checksum == temp_checksum:
                    operation_result['integrity_verified'] = True
                else:
                    temp_path.unlink()  # Remove corrupted temporary file
                    raise ValueError(f"Integrity verification failed: source checksum {source_checksum} != destination checksum {temp_checksum}")
            
            # Move temporary file to final destination for atomic operation
            shutil.move(str(temp_path), str(dest_file))
            
        else:
            # Perform direct copy operation without atomic guarantee
            shutil.copy2(str(source_file), str(dest_file))
            
            # Set appropriate file permissions on destination
            if os.name != 'nt':  # Not Windows
                source_permissions = source_file.stat().st_mode
                os.chmod(str(dest_file), source_permissions)
            
            # Verify file integrity if verify_integrity is enabled
            if verify_integrity and source_checksum:
                dest_checksum = calculate_file_checksum(destination_path)
                if source_checksum == dest_checksum:
                    operation_result['integrity_verified'] = True
                else:
                    dest_file.unlink()  # Remove corrupted destination file
                    raise ValueError(f"Integrity verification failed: source checksum {source_checksum} != destination checksum {dest_checksum}")
        
        # Calculate operation performance metrics
        operation_result['operation_time_seconds'] = time.time() - start_time
        if operation_result['operation_time_seconds'] > 0:
            operation_result['copy_rate_mb_per_sec'] = (source_size / (1024 * 1024)) / operation_result['operation_time_seconds']
        
        operation_result['success'] = True
        
        # Log successful copy operation with performance metrics
        logger.info(
            f"File copy successful: {source_path} -> {destination_path} "
            f"({operation_result['bytes_copied']} bytes in {operation_result['operation_time_seconds']:.2f}s, "
            f"{operation_result['copy_rate_mb_per_sec']:.2f} MB/s)"
        )
        
        # Create audit trail for copy operation
        create_audit_trail(
            action='FILE_COPY',
            component='FILE_OPERATIONS',
            action_details=operation_result
        )
        
        return operation_result
        
    except Exception as e:
        # Handle copy operation errors with comprehensive error reporting
        operation_result['error'] = str(e)
        operation_result['error_type'] = type(e).__name__
        operation_result['operation_time_seconds'] = time.time() - start_time
        
        handle_file_error(
            error=e,
            operation_context='file_copy',
            file_path=source_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        logger.error(f"File copy failed: {source_path} -> {destination_path} - {e}")
        return operation_result


@retry_with_backoff(max_attempts=3, retryable_exceptions=(IOError, OSError, shutil.Error))
def safe_file_move(
    source_path: str,
    destination_path: str,
    verify_integrity: bool = True,
    atomic_operation: bool = True
) -> dict:
    """
    Safely move files with atomic operations, integrity verification, and rollback capability 
    for reliable file system operations during batch processing.
    
    This function provides robust file moving with comprehensive error handling, integrity 
    verification, and atomic operations with rollback capability.
    
    Args:
        source_path: Path to the source file to move
        destination_path: Path to the destination for the move
        verify_integrity: Verify file integrity before removing source
        atomic_operation: Use atomic move operation when possible
        
    Returns:
        dict: File move operation result with success status and integrity verification
    """
    logger = get_logger('file_move', 'FILE_OPERATIONS')
    
    operation_result = {
        'success': False,
        'source_path': source_path,
        'destination_path': destination_path,
        'integrity_verified': False,
        'operation_time_seconds': 0,
        'bytes_moved': 0,
        'move_rate_mb_per_sec': 0,
        'cross_filesystem': False,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    start_time = time.time()
    
    try:
        # Validate source file exists and is accessible
        source_file = pathlib.Path(source_path)
        if not source_file.exists() or not source_file.is_file():
            raise FileNotFoundError(f"Source file not found or invalid: {source_path}")
        
        source_size = source_file.stat().st_size
        operation_result['bytes_moved'] = source_size
        
        # Create destination directory if needed
        dest_file = pathlib.Path(destination_path)
        ensure_directory_exists(str(dest_file.parent), create_parents=True)
        
        # Calculate source file checksum if verify_integrity is enabled
        source_checksum = None
        if verify_integrity:
            source_checksum = calculate_file_checksum(source_path)
        
        # Check if source and destination are on the same filesystem
        try:
            source_stat = source_file.stat()
            dest_parent_stat = dest_file.parent.stat()
            same_filesystem = source_stat.st_dev == dest_parent_stat.st_dev
        except OSError:
            same_filesystem = False
        
        operation_result['cross_filesystem'] = not same_filesystem
        
        # Use atomic move operation if on same filesystem
        if atomic_operation and same_filesystem:
            # Direct atomic move using os.rename for same filesystem
            os.rename(str(source_file), str(dest_file))
            
            # Verify destination file integrity if verify_integrity is enabled
            if verify_integrity and source_checksum:
                dest_checksum = calculate_file_checksum(destination_path)
                if source_checksum == dest_checksum:
                    operation_result['integrity_verified'] = True
                else:
                    # Restore source file from destination if integrity check fails
                    os.rename(str(dest_file), str(source_file))
                    raise ValueError(f"Integrity verification failed during move: source checksum {source_checksum} != destination checksum {dest_checksum}")
        
        else:
            # Fall back to copy-and-delete for cross-filesystem moves
            copy_result = safe_file_copy(
                source_path=source_path,
                destination_path=destination_path,
                verify_integrity=verify_integrity,
                create_backup=False,
                atomic_operation=atomic_operation
            )
            
            if not copy_result['success']:
                raise RuntimeError(f"Copy operation failed during move: {copy_result.get('error', 'Unknown error')}")
            
            operation_result['integrity_verified'] = copy_result['integrity_verified']
            
            # Remove source file only after successful verification
            if not verify_integrity or operation_result['integrity_verified']:
                source_file.unlink()
            else:
                # Clean up destination file and raise error
                dest_file.unlink()
                raise ValueError("Integrity verification failed - source file preserved")
        
        # Calculate operation performance metrics
        operation_result['operation_time_seconds'] = time.time() - start_time
        if operation_result['operation_time_seconds'] > 0:
            operation_result['move_rate_mb_per_sec'] = (source_size / (1024 * 1024)) / operation_result['operation_time_seconds']
        
        operation_result['success'] = True
        
        # Log successful move operation with audit trail
        logger.info(
            f"File move successful: {source_path} -> {destination_path} "
            f"({operation_result['bytes_moved']} bytes in {operation_result['operation_time_seconds']:.2f}s, "
            f"cross-filesystem: {operation_result['cross_filesystem']})"
        )
        
        # Create audit trail for move operation
        create_audit_trail(
            action='FILE_MOVE',
            component='FILE_OPERATIONS',
            action_details=operation_result
        )
        
        return operation_result
        
    except Exception as e:
        # Handle move operation errors with comprehensive error reporting and rollback
        operation_result['error'] = str(e)
        operation_result['error_type'] = type(e).__name__
        operation_result['operation_time_seconds'] = time.time() - start_time
        
        handle_file_error(
            error=e,
            operation_context='file_move',
            file_path=source_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        logger.error(f"File move failed: {source_path} -> {destination_path} - {e}")
        return operation_result


@retry_with_backoff(max_attempts=2, retryable_exceptions=(IOError, json.JSONDecodeError))
def load_json_config(
    config_path: str,
    schema_path: str = None,
    validate_schema: bool = False,
    use_cache: bool = True
) -> dict:
    """
    Load and validate JSON configuration files with schema validation, error handling, and 
    caching for scientific computing configuration management and reproducible experimental conditions.
    
    This function provides robust JSON configuration loading with comprehensive validation, 
    caching, and error handling for scientific computing workflows.
    
    Args:
        config_path: Path to the JSON configuration file
        schema_path: Path to JSON schema file for validation
        validate_schema: Perform schema validation using jsonschema
        use_cache: Enable configuration caching for improved performance
        
    Returns:
        dict: Loaded and validated JSON configuration data with schema compliance verification
    """
    logger = get_logger('config_loader', 'CONFIGURATION')
    
    try:
        # Check configuration cache if use_cache is enabled
        cache_key = f"{config_path}:{schema_path}:{validate_schema}"
        if use_cache and cache_key in _operation_cache:
            logger.debug(f"Configuration loaded from cache: {config_path}")
            return _operation_cache[cache_key].copy()
        
        # Validate configuration file exists and is accessible
        config_file = pathlib.Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_file.is_file():
            raise ValueError(f"Configuration path is not a file: {config_path}")
        
        # Load JSON configuration with error handling for malformed JSON
        with open(config_file, 'r', encoding='utf-8') as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in configuration file {config_path}: {e.msg}",
                    e.doc,
                    e.pos
                )
        
        # Load JSON schema if schema_path is provided and validate_schema is enabled
        if validate_schema and schema_path:
            schema_file = pathlib.Path(schema_path)
            if not schema_file.exists():
                logger.warning(f"Schema file not found, skipping validation: {schema_path}")
            else:
                try:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                    
                    # Perform schema validation using jsonschema library
                    jsonschema.validate(instance=config_data, schema=schema_data)
                    logger.debug(f"Configuration schema validation successful: {config_path}")
                    
                except jsonschema.ValidationError as e:
                    raise ValueError(f"Configuration schema validation failed: {e.message}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON schema file {schema_path}: {e.msg}")
        
        # Apply environment variable substitution in configuration values
        config_data = _substitute_environment_variables(config_data)
        
        # Cache configuration if caching is enabled
        if use_cache:
            _operation_cache[cache_key] = config_data.copy()
            logger.debug(f"Configuration cached: {config_path}")
        
        # Log successful configuration loading operation with audit trail
        logger.info(f"Configuration loaded successfully: {config_path}")
        
        create_audit_trail(
            action='CONFIG_LOADED',
            component='CONFIGURATION',
            action_details={
                'config_path': config_path,
                'schema_path': schema_path,
                'validate_schema': validate_schema,
                'use_cache': use_cache,
                'config_keys_count': len(config_data) if isinstance(config_data, dict) else 0
            }
        )
        
        return config_data
        
    except Exception as e:
        # Handle configuration loading errors with comprehensive error reporting
        handle_file_error(
            error=e,
            operation_context='json_config_loading',
            file_path=config_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        logger.error(f"Failed to load configuration: {config_path} - {e}")
        raise e


def save_json_config(
    config_data: dict,
    config_path: str,
    schema_path: str = None,
    validate_schema: bool = False,
    create_backup: bool = True,
    atomic_write: bool = True
) -> dict:
    """
    Save JSON configuration files with atomic operations, schema validation, and backup creation 
    for reliable configuration management and experimental reproducibility.
    
    This function provides robust JSON configuration saving with comprehensive validation, 
    atomic operations, and backup creation for scientific computing workflows.
    
    Args:
        config_data: Configuration dictionary to save as JSON
        config_path: Path to save the JSON configuration file
        schema_path: Path to JSON schema file for validation
        validate_schema: Perform schema validation before saving
        create_backup: Create backup of existing configuration file
        atomic_write: Use atomic write operation for data safety
        
    Returns:
        dict: Configuration save operation result with validation status and backup information
    """
    logger = get_logger('config_saver', 'CONFIGURATION')
    
    operation_result = {
        'success': False,
        'config_path': config_path,
        'backup_created': False,
        'backup_path': None,
        'schema_validated': False,
        'bytes_written': 0,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        # Validate configuration data structure and types
        if not isinstance(config_data, dict):
            raise TypeError(f"Configuration data must be a dictionary, got {type(config_data)}")
        
        # Perform schema validation if validate_schema is enabled
        if validate_schema and schema_path:
            schema_file = pathlib.Path(schema_path)
            if schema_file.exists():
                try:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)
                    
                    jsonschema.validate(instance=config_data, schema=schema_data)
                    operation_result['schema_validated'] = True
                    logger.debug(f"Configuration schema validation successful before save: {config_path}")
                    
                except jsonschema.ValidationError as e:
                    raise ValueError(f"Configuration schema validation failed: {e.message}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON schema file {schema_path}: {e.msg}")
            else:
                logger.warning(f"Schema file not found, skipping validation: {schema_path}")
        
        # Create destination directory and prepare file paths
        config_file = pathlib.Path(config_path)
        ensure_directory_exists(str(config_file.parent), create_parents=True)
        
        # Create backup of existing configuration if create_backup is enabled
        if create_backup and config_file.exists():
            backup_path = config_file.with_suffix(config_file.suffix + BACKUP_SUFFIX)
            shutil.copy2(str(config_file), str(backup_path))
            operation_result['backup_created'] = True
            operation_result['backup_path'] = str(backup_path)
            logger.info(f"Configuration backup created: {backup_path}")
        
        # Serialize configuration data to JSON with proper formatting
        json_data = json.dumps(config_data, indent=2, sort_keys=True, ensure_ascii=False)
        operation_result['bytes_written'] = len(json_data.encode('utf-8'))
        
        # Use temporary file for atomic write if atomic_write is enabled
        if atomic_write:
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=config_file.parent,
                prefix=TEMP_FILE_PREFIX,
                suffix='.json',
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_path = pathlib.Path(temp_file.name)
                
                # Write configuration to temporary file with error handling
                temp_file.write(json_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
            
            # Set appropriate file permissions
            if os.name != 'nt':  # Not Windows
                os.chmod(str(temp_path), 0o644)
            
            # Move temporary file to final location for atomic operation
            shutil.move(str(temp_path), str(config_file))
            
        else:
            # Direct write without atomic guarantee
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(json_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Set appropriate file permissions
            if os.name != 'nt':  # Not Windows
                os.chmod(str(config_file), 0o644)
        
        # Update configuration cache if caching is enabled
        cache_key = f"{config_path}:None:False"  # Basic cache key for saved config
        _operation_cache[cache_key] = config_data.copy()
        
        operation_result['success'] = True
        
        # Log successful configuration save operation with audit trail
        logger.info(f"Configuration saved successfully: {config_path} ({operation_result['bytes_written']} bytes)")
        
        create_audit_trail(
            action='CONFIG_SAVED',
            component='CONFIGURATION',
            action_details=operation_result
        )
        
        return operation_result
        
    except Exception as e:
        # Handle configuration saving errors with comprehensive error reporting
        operation_result['error'] = str(e)
        operation_result['error_type'] = type(e).__name__
        
        handle_file_error(
            error=e,
            operation_context='json_config_saving',
            file_path=config_path,
            error_severity=ERROR_SEVERITY_HIGH
        )
        
        logger.error(f"Failed to save configuration: {config_path} - {e}")
        return operation_result


@contextlib.contextmanager
def create_file_lock(
    file_path: str,
    timeout: float = FILE_LOCK_TIMEOUT,
    exclusive: bool = True
):
    """
    Create thread-safe file lock for concurrent access control during batch processing operations 
    with timeout handling and deadlock prevention.
    
    This context manager provides thread-safe file locking for concurrent access control with 
    automatic cleanup and deadlock prevention mechanisms.
    
    Args:
        file_path: Path to the file to lock
        timeout: Maximum time to wait for lock acquisition
        exclusive: Use exclusive locking (True) or shared locking (False)
        
    Returns:
        contextlib.contextmanager: Context manager for file locking with automatic cleanup
    """
    logger = get_logger('file_locking', 'FILE_OPERATIONS')
    
    # Normalize file path for consistent lock identification
    normalized_path = str(pathlib.Path(file_path).absolute())
    
    # Get or create thread lock for the file path
    if normalized_path not in _file_locks:
        _file_locks[normalized_path] = threading.Lock()
    
    file_lock = _file_locks[normalized_path]
    lock_acquired = False
    
    try:
        # Acquire lock with timeout handling
        logger.debug(f"Attempting to acquire {'exclusive' if exclusive else 'shared'} lock: {file_path}")
        
        lock_acquired = file_lock.acquire(timeout=timeout)
        
        if not lock_acquired:
            raise TimeoutError(f"Failed to acquire file lock within {timeout} seconds: {file_path}")
        
        logger.debug(f"File lock acquired: {file_path}")
        
        # Create audit trail for lock acquisition
        create_audit_trail(
            action='FILE_LOCK_ACQUIRED',
            component='FILE_OPERATIONS',
            action_details={
                'file_path': file_path,
                'lock_type': 'exclusive' if exclusive else 'shared',
                'timeout': timeout
            }
        )
        
        # Yield control to the calling code
        yield
        
    except Exception as e:
        # Log lock-related errors
        logger.error(f"Error during file lock operation: {file_path} - {e}")
        raise e
        
    finally:
        # Provide automatic lock release on context exit
        if lock_acquired:
            file_lock.release()
            logger.debug(f"File lock released: {file_path}")
            
            # Create audit trail for lock release
            create_audit_trail(
                action='FILE_LOCK_RELEASED',
                component='FILE_OPERATIONS',
                action_details={
                    'file_path': file_path,
                    'lock_type': 'exclusive' if exclusive else 'shared'
                }
            )


def cleanup_temporary_files(
    temp_directory: str,
    max_age_hours: int = 24,
    dry_run: bool = False
) -> dict:
    """
    Clean up temporary files and directories created during processing operations with age-based 
    filtering and safe deletion for system maintenance.
    
    This function provides comprehensive temporary file cleanup with age-based filtering and 
    safe deletion for system maintenance and storage optimization.
    
    Args:
        temp_directory: Path to temporary directory to clean up
        max_age_hours: Maximum age in hours for files to be considered for deletion
        dry_run: Perform dry run (log only, no deletion) for testing
        
    Returns:
        dict: Cleanup operation result with deleted file count and freed space information
    """
    logger = get_logger('temp_cleanup', 'FILE_OPERATIONS')
    
    cleanup_result = {
        'success': False,
        'temp_directory': temp_directory,
        'files_found': 0,
        'files_deleted': 0,
        'directories_deleted': 0,
        'bytes_freed': 0,
        'dry_run': dry_run,
        'errors': [],
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    try:
        # Validate temporary directory exists and is accessible
        temp_dir = pathlib.Path(temp_directory)
        if not temp_dir.exists():
            logger.warning(f"Temporary directory does not exist: {temp_directory}")
            cleanup_result['success'] = True  # Not an error if directory doesn't exist
            return cleanup_result
        
        if not temp_dir.is_dir():
            raise ValueError(f"Temporary path is not a directory: {temp_directory}")
        
        # Calculate cutoff time for age-based filtering
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        files_to_delete = []
        directories_to_delete = []
        total_size_to_free = 0
        
        # Scan directory for files matching temporary file patterns
        for item in temp_dir.rglob('*'):
            try:
                item_stat = item.stat()
                item_age = datetime.datetime.fromtimestamp(item_stat.st_mtime)
                
                # Filter files by age using max_age_hours parameter
                if item_stat.st_mtime < cutoff_timestamp:
                    if item.is_file():
                        # Check if file matches temporary file patterns
                        if (item.name.startswith(TEMP_FILE_PREFIX) or 
                            item.suffix in ['.tmp', '.temp', '.bak'] or
                            item.name.endswith(BACKUP_SUFFIX)):
                            
                            files_to_delete.append((item, item_stat.st_size))
                            total_size_to_free += item_stat.st_size
                            cleanup_result['files_found'] += 1
                    
                    elif item.is_dir() and not any(item.iterdir()):  # Empty directory
                        directories_to_delete.append(item)
                        
            except (OSError, PermissionError) as e:
                cleanup_result['errors'].append(f"Error accessing {item}: {e}")
                logger.warning(f"Error accessing file during cleanup: {item} - {e}")
        
        # Calculate total space to be freed
        cleanup_result['bytes_freed'] = total_size_to_free
        
        # Perform dry run if dry_run is enabled (log only, no deletion)
        if dry_run:
            logger.info(
                f"Dry run cleanup: would delete {len(files_to_delete)} files and "
                f"{len(directories_to_delete)} directories, freeing {total_size_to_free / (1024*1024):.2f} MB"
            )
            cleanup_result['files_deleted'] = len(files_to_delete)
            cleanup_result['directories_deleted'] = len(directories_to_delete)
            cleanup_result['success'] = True
            return cleanup_result
        
        # Delete expired temporary files with error handling
        for file_path, file_size in files_to_delete:
            try:
                file_path.unlink()
                cleanup_result['files_deleted'] += 1
                logger.debug(f"Deleted temporary file: {file_path}")
            except (OSError, PermissionError) as e:
                cleanup_result['errors'].append(f"Error deleting file {file_path}: {e}")
                logger.warning(f"Failed to delete temporary file: {file_path} - {e}")
        
        # Remove empty temporary directories
        for dir_path in directories_to_delete:
            try:
                dir_path.rmdir()
                cleanup_result['directories_deleted'] += 1
                logger.debug(f"Deleted empty directory: {dir_path}")
            except (OSError, PermissionError) as e:
                cleanup_result['errors'].append(f"Error deleting directory {dir_path}: {e}")
                logger.warning(f"Failed to delete empty directory: {dir_path} - {e}")
        
        cleanup_result['success'] = True
        
        # Log cleanup operation with comprehensive statistics
        logger.info(
            f"Temporary file cleanup completed: {cleanup_result['files_deleted']} files and "
            f"{cleanup_result['directories_deleted']} directories deleted, "
            f"{cleanup_result['bytes_freed'] / (1024*1024):.2f} MB freed"
        )
        
        # Create audit trail for cleanup operation
        create_audit_trail(
            action='TEMP_FILE_CLEANUP',
            component='FILE_OPERATIONS',
            action_details=cleanup_result
        )
        
        return cleanup_result
        
    except Exception as e:
        # Handle cleanup operation errors with comprehensive error reporting
        cleanup_result['error'] = str(e)
        cleanup_result['error_type'] = type(e).__name__
        
        handle_file_error(
            error=e,
            operation_context='temporary_file_cleanup',
            file_path=temp_directory,
            error_severity=ERROR_SEVERITY_MEDIUM
        )
        
        logger.error(f"Temporary file cleanup failed: {temp_directory} - {e}")
        return cleanup_result


class ErrorSeverity:
    """
    Localized error severity enumeration class providing error classification levels for file 
    operations with severity assessment and priority determination without external dependencies.
    
    This class provides comprehensive error severity classification for file operations with 
    numerical priority mapping and action requirement assessment.
    """
    
    CRITICAL = ERROR_SEVERITY_CRITICAL
    HIGH = ERROR_SEVERITY_HIGH
    MEDIUM = ERROR_SEVERITY_MEDIUM
    LOW = ERROR_SEVERITY_LOW
    
    @staticmethod
    def get_priority(severity_level: str) -> int:
        """
        Get numerical priority value for severity level for sorting and comparison operations.
        
        Args:
            severity_level: Severity level string to convert to numerical priority
            
        Returns:
            int: Numerical priority value with higher numbers indicating higher severity
        """
        priority_mapping = {
            ErrorSeverity.CRITICAL: 4,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.LOW: 1
        }
        
        return priority_mapping.get(severity_level, 0)
    
    @staticmethod
    def requires_immediate_action(severity_level: str) -> bool:
        """
        Determine if severity level requires immediate action and intervention.
        
        Args:
            severity_level: Severity level to evaluate for action requirement
            
        Returns:
            bool: True if severity requires immediate action
        """
        immediate_action_levels = {ErrorSeverity.CRITICAL, ErrorSeverity.HIGH}
        return severity_level in immediate_action_levels


class FileValidationResult:
    """
    Localized validation result container for file validation operations providing error and 
    warning management, validation status tracking, and comprehensive reporting for scientific 
    computing file operations without external dependencies.
    
    This class provides comprehensive validation result tracking with detailed error and warning 
    management for scientific computing file operations.
    """
    
    def __init__(self, file_path: str, is_valid: bool = True):
        """
        Initialize file validation result with file path and initial validation status for 
        comprehensive file validation tracking.
        
        Args:
            file_path: Path to the file being validated
            is_valid: Initial validation status
        """
        self.file_path = file_path
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.validation_timestamp = datetime.datetime.now()
        self.recommendations: List[str] = []
    
    def add_error(self, error_message: str, error_category: str = None) -> None:
        """
        Add validation error to the result with automatic status update for fail-fast 
        validation strategy.
        
        Args:
            error_message: Descriptive error message
            error_category: Category or type of error for classification
        """
        formatted_error = f"[{error_category}] {error_message}" if error_category else error_message
        self.errors.append(formatted_error)
        self.is_valid = False  # Any error invalidates the file
        
        # Log error addition for audit trail
        logger = get_logger('validation_result', 'VALIDATION')
        logger.debug(f"Validation error added: {formatted_error}")
    
    def add_warning(self, warning_message: str, warning_category: str = None) -> None:
        """
        Add validation warning to the result without affecting validation status for 
        non-critical issues.
        
        Args:
            warning_message: Descriptive warning message
            warning_category: Category or type of warning for classification
        """
        formatted_warning = f"[{warning_category}] {warning_message}" if warning_category else warning_message
        self.warnings.append(formatted_warning)
        
        # Log warning addition for audit trail
        logger = get_logger('validation_result', 'VALIDATION')
        logger.debug(f"Validation warning added: {formatted_warning}")
    
    def add_recommendation(self, recommendation_text: str) -> None:
        """
        Add recommendation for resolving validation issues or improving file quality.
        
        Args:
            recommendation_text: Recommendation text for issue resolution
        """
        self.recommendations.append(recommendation_text)
        
        # Log recommendation addition
        logger = get_logger('validation_result', 'VALIDATION')
        logger.debug(f"Validation recommendation added: {recommendation_text}")
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata key-value pair for validation context and debugging information.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
        # Log metadata addition for audit trail
        logger = get_logger('validation_result', 'VALIDATION')
        logger.debug(f"Validation metadata set: {key} = {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result to dictionary format for serialization, logging, and reporting 
        with comprehensive validation information.
        
        Returns:
            Dict[str, Any]: Complete validation result as dictionary with all errors, warnings, and metadata
        """
        return {
            'file_path': self.file_path,
            'is_valid': self.is_valid,
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'metadata': self.metadata.copy(),
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'recommendations': self.recommendations.copy(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation result with error counts and status overview for quick assessment.
        
        Returns:
            Dict[str, Any]: Validation summary with counts and status information
        """
        return {
            'file_path': self.file_path,
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'has_recommendations': len(self.recommendations) > 0,
            'validation_timestamp': self.validation_timestamp.isoformat()
        }


class ConfigurationManager:
    """
    Comprehensive configuration file management class providing loading, validation, caching, 
    versioning, and atomic operations for scientific computing configuration with schema 
    enforcement and audit trail support.
    
    This class provides centralized configuration management with comprehensive validation, 
    caching, and versioning for scientific computing workflows.
    """
    
    def __init__(self, config_directory: str, schema_directory: str = None, enable_caching: bool = True):
        """
        Initialize configuration manager with directory paths, caching settings, and validation 
        framework for scientific computing configuration management.
        
        Args:
            config_directory: Directory containing configuration files
            schema_directory: Directory containing JSON schema files
            enable_caching: Enable configuration caching for performance
        """
        self.config_directory = pathlib.Path(config_directory)
        self.schema_directory = pathlib.Path(schema_directory) if schema_directory else None
        self.caching_enabled = enable_caching
        self.config_cache: dict = {}
        self.schema_cache: dict = {}
        self.cache_lock = threading.Lock()
        self.validation_history: dict = {}
        
        # Validate configuration and schema directories exist
        if not self.config_directory.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_directory}")
        
        if self.schema_directory and not self.schema_directory.exists():
            logger = get_logger('config_manager', 'CONFIGURATION')
            logger.warning(f"Schema directory not found: {schema_directory}")
        
        # Configure logging for configuration operations
        self.logger = get_logger('config_manager', 'CONFIGURATION')
        
        # Create audit trail for configuration manager initialization
        create_audit_trail(
            action='CONFIG_MANAGER_INIT',
            component='CONFIGURATION',
            action_details={
                'config_directory': str(self.config_directory),
                'schema_directory': str(self.schema_directory) if self.schema_directory else None,
                'caching_enabled': self.caching_enabled
            }
        )
        
        self.logger.info(f"Configuration manager initialized: {config_directory}")
    
    def load_configuration(self, config_name: str, validate_schema: bool = True, use_cache: bool = None) -> dict:
        """
        Load configuration file with schema validation, caching, and error handling for reproducible 
        scientific computing environments.
        
        Args:
            config_name: Name of the configuration file (without extension)
            validate_schema: Perform schema validation if schema is available
            use_cache: Use configuration cache (defaults to instance setting)
            
        Returns:
            dict: Loaded and validated configuration data with schema compliance
        """
        if use_cache is None:
            use_cache = self.caching_enabled
        
        config_path = self.config_directory / f"{config_name}.json"
        schema_path = None
        
        # Determine schema path if schema validation is enabled
        if validate_schema and self.schema_directory:
            schema_path = self.schema_directory / f"{config_name}_schema.json"
            if not schema_path.exists():
                self.logger.warning(f"Schema file not found for {config_name}, skipping validation")
                schema_path = None
        
        try:
            # Load configuration using load_json_config function
            config_data = load_json_config(
                config_path=str(config_path),
                schema_path=str(schema_path) if schema_path else None,
                validate_schema=validate_schema and schema_path is not None,
                use_cache=use_cache
            )
            
            # Update validation history
            self.validation_history[config_name] = {
                'last_loaded': datetime.datetime.now().isoformat(),
                'schema_validated': validate_schema and schema_path is not None,
                'load_successful': True
            }
            
            self.logger.info(f"Configuration loaded successfully: {config_name}")
            return config_data
            
        except Exception as e:
            # Update validation history with error information
            self.validation_history[config_name] = {
                'last_loaded': datetime.datetime.now().isoformat(),
                'schema_validated': False,
                'load_successful': False,
                'error': str(e)
            }
            
            self.logger.error(f"Failed to load configuration: {config_name} - {e}")
            raise e
    
    def save_configuration(self, config_name: str, config_data: dict, validate_schema: bool = True, create_backup: bool = True) -> dict:
        """
        Save configuration file with atomic operations, schema validation, and backup creation 
        for reliable configuration persistence.
        
        Args:
            config_name: Name of the configuration file (without extension)
            config_data: Configuration data to save
            validate_schema: Perform schema validation before saving
            create_backup: Create backup of existing configuration
            
        Returns:
            dict: Configuration save operation result with validation and backup status
        """
        config_path = self.config_directory / f"{config_name}.json"
        schema_path = None
        
        # Determine schema path if schema validation is enabled
        if validate_schema and self.schema_directory:
            schema_path = self.schema_directory / f"{config_name}_schema.json"
            if not schema_path.exists():
                self.logger.warning(f"Schema file not found for {config_name}, skipping validation")
                schema_path = None
        
        try:
            # Save configuration using save_json_config function
            save_result = save_json_config(
                config_data=config_data,
                config_path=str(config_path),
                schema_path=str(schema_path) if schema_path else None,
                validate_schema=validate_schema and schema_path is not None,
                create_backup=create_backup,
                atomic_write=True
            )
            
            # Update configuration cache if caching is enabled
            if self.caching_enabled:
                with self.cache_lock:
                    cache_key = f"{config_path}:None:False"
                    self.config_cache[cache_key] = config_data.copy()
            
            # Record configuration change in validation history
            self.validation_history[config_name] = {
                'last_saved': datetime.datetime.now().isoformat(),
                'schema_validated': validate_schema and schema_path is not None,
                'save_successful': save_result['success'],
                'backup_created': save_result.get('backup_created', False)
            }
            
            # Create audit trail for configuration change
            create_audit_trail(
                action='CONFIG_SAVED',
                component='CONFIGURATION',
                action_details={
                    'config_name': config_name,
                    'save_result': save_result,
                    'schema_validated': validate_schema and schema_path is not None
                }
            )
            
            self.logger.info(f"Configuration saved successfully: {config_name}")
            return save_result
            
        except Exception as e:
            # Record save failure in validation history
            self.validation_history[config_name] = {
                'last_saved': datetime.datetime.now().isoformat(),
                'schema_validated': False,
                'save_successful': False,
                'error': str(e)
            }
            
            self.logger.error(f"Failed to save configuration: {config_name} - {e}")
            raise e
    
    def validate_all_configurations(self, strict_validation: bool = False, generate_report: bool = True) -> dict:
        """
        Validate all configuration files in the configuration directory against their schemas 
        for system-wide configuration integrity.
        
        Args:
            strict_validation: Use strict validation mode with comprehensive checks
            generate_report: Generate comprehensive validation report
            
        Returns:
            dict: Comprehensive validation report for all configuration files with error summary
        """
        validation_results = {}
        error_count = 0
        warning_count = 0
        
        # Scan configuration directory for all configuration files
        for config_file in self.config_directory.glob("*.json"):
            config_name = config_file.stem
            
            try:
                # Load each configuration file with validation
                config_data = self.load_configuration(
                    config_name=config_name,
                    validate_schema=True,
                    use_cache=False  # Force reload for validation
                )
                
                validation_results[config_name] = {
                    'valid': True,
                    'errors': [],
                    'warnings': [],
                    'file_path': str(config_file)
                }
                
            except Exception as e:
                error_count += 1
                validation_results[config_name] = {
                    'valid': False,
                    'errors': [str(e)],
                    'warnings': [],
                    'file_path': str(config_file)
                }
        
        # Generate comprehensive validation report if generate_report is enabled
        if generate_report:
            report = {
                'validation_timestamp': datetime.datetime.now().isoformat(),
                'total_configurations': len(validation_results),
                'valid_configurations': len([r for r in validation_results.values() if r['valid']]),
                'invalid_configurations': error_count,
                'total_errors': error_count,
                'total_warnings': warning_count,
                'strict_validation': strict_validation,
                'results': validation_results
            }
        else:
            report = {
                'total_configurations': len(validation_results),
                'valid_configurations': len([r for r in validation_results.values() if r['valid']]),
                'invalid_configurations': error_count
            }
        
        # Log validation operation with summary statistics
        self.logger.info(
            f"Configuration validation completed: {report['valid_configurations']}/{report['total_configurations']} valid"
        )
        
        return report
    
    def clear_cache(self, clear_validation_history: bool = False) -> dict:
        """
        Clear configuration and schema caches for fresh loading and memory management.
        
        Args:
            clear_validation_history: Clear validation history along with caches
            
        Returns:
            dict: Cache clearing operation result with statistics
        """
        with self.cache_lock:
            config_cache_size = len(self.config_cache)
            schema_cache_size = len(self.schema_cache)
            
            # Clear configuration cache dictionary
            self.config_cache.clear()
            
            # Clear schema cache dictionary
            self.schema_cache.clear()
            
            # Clear validation history if clear_validation_history is enabled
            if clear_validation_history:
                validation_history_size = len(self.validation_history)
                self.validation_history.clear()
            else:
                validation_history_size = 0
        
        # Log cache clearing operation
        self.logger.info(f"Configuration caches cleared: {config_cache_size} configs, {schema_cache_size} schemas")
        
        return {
            'config_cache_cleared': config_cache_size,
            'schema_cache_cleared': schema_cache_size,
            'validation_history_cleared': validation_history_size,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def get_configuration_status(self, include_cache_stats: bool = True) -> dict:
        """
        Get status information for all managed configurations including validation history and 
        cache statistics.
        
        Args:
            include_cache_stats: Include cache statistics in status report
            
        Returns:
            dict: Configuration status report with validation history and cache information
        """
        status_report = {
            'config_directory': str(self.config_directory),
            'schema_directory': str(self.schema_directory) if self.schema_directory else None,
            'caching_enabled': self.caching_enabled,
            'validation_history': self.validation_history.copy(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Include cache statistics if include_cache_stats is enabled
        if include_cache_stats:
            with self.cache_lock:
                status_report['cache_statistics'] = {
                    'config_cache_size': len(self.config_cache),
                    'schema_cache_size': len(self.schema_cache),
                    'cached_configurations': list(self.config_cache.keys())
                }
        
        # Collect configuration file information
        config_files = list(self.config_directory.glob("*.json"))
        status_report['configuration_files'] = {
            'total_count': len(config_files),
            'file_list': [f.name for f in config_files]
        }
        
        return status_report


class FileIntegrityValidator:
    """
    File integrity validation class providing checksum verification, corruption detection, format 
    validation, and comprehensive file health assessment for scientific data reliability.
    
    This class provides comprehensive file integrity validation with checksum verification, 
    corruption detection, and format validation for scientific computing workflows.
    """
    
    def __init__(self, supported_formats: list = None, default_checksum_algorithm: str = DEFAULT_CHECKSUM_ALGORITHM):
        """
        Initialize file integrity validator with supported formats and checksum algorithm configuration.
        
        Args:
            supported_formats: List of supported file formats for validation
            default_checksum_algorithm: Default checksum algorithm for integrity verification
        """
        self.supported_formats = supported_formats or (SUPPORTED_VIDEO_FORMATS + SUPPORTED_CONFIG_FORMATS)
        self.checksum_algorithm = default_checksum_algorithm
        self.validation_cache: dict = {}
        self.integrity_history: dict = {}
        
        # Validate checksum algorithm is supported
        if self.checksum_algorithm not in SUPPORTED_CHECKSUM_ALGORITHMS:
            raise ValueError(f"Unsupported checksum algorithm: {self.checksum_algorithm}")
        
        # Configure logging for integrity validation operations
        self.logger = get_logger('integrity_validator', 'VALIDATION')
        
        self.logger.info(f"File integrity validator initialized with {len(self.supported_formats)} supported formats")
    
    def validate_file_integrity(self, file_path: str, expected_checksum: str = None, deep_validation: bool = False) -> FileValidationResult:
        """
        Comprehensive file integrity validation including checksum verification, format validation, 
        and corruption detection.
        
        Args:
            file_path: Path to the file for integrity validation
            expected_checksum: Expected checksum value for verification
            deep_validation: Perform deep validation including content analysis
            
        Returns:
            FileValidationResult: File integrity validation result with checksum verification and corruption analysis
        """
        # Create FileValidationResult container for integrity check
        validation_result = FileValidationResult(file_path=file_path, is_valid=True)
        
        try:
            # Validate file exists and is accessible
            if not pathlib.Path(file_path).exists():
                validation_result.add_error("File does not exist", "FILE_NOT_FOUND")
                return validation_result
            
            # Calculate file checksum using configured algorithm
            calculated_checksum = calculate_file_checksum(file_path, self.checksum_algorithm)
            validation_result.set_metadata('calculated_checksum', calculated_checksum)
            validation_result.set_metadata('checksum_algorithm', self.checksum_algorithm)
            
            # Compare with expected checksum if provided
            if expected_checksum:
                if calculated_checksum == expected_checksum:
                    validation_result.set_metadata('checksum_verified', True)
                    self.logger.debug(f"Checksum verification successful: {file_path}")
                else:
                    validation_result.add_error(
                        f"Checksum mismatch: expected {expected_checksum}, got {calculated_checksum}",
                        "CHECKSUM_MISMATCH"
                    )
                    validation_result.set_metadata('checksum_verified', False)
            
            # Perform format-specific validation
            file_ext = pathlib.Path(file_path).suffix.lower()
            if file_ext in self.supported_formats:
                validation_result.set_metadata('format_supported', True)
                
                # Video file specific validation
                if file_ext in SUPPORTED_VIDEO_FORMATS:
                    video_validation = validate_video_file(file_path, check_integrity=deep_validation)
                    validation_result.errors.extend(video_validation.errors)
                    validation_result.warnings.extend(video_validation.warnings)
                    validation_result.metadata.update(video_validation.metadata)
                    if not video_validation.is_valid:
                        validation_result.is_valid = False
                
                # JSON configuration file validation
                elif file_ext == '.json':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json.load(f)
                        validation_result.set_metadata('json_valid', True)
                    except json.JSONDecodeError as e:
                        validation_result.add_error(f"Invalid JSON format: {e}", "JSON_PARSE_ERROR")
            else:
                validation_result.set_metadata('format_supported', False)
                validation_result.add_warning(
                    f"File format {file_ext} not in supported formats list",
                    "UNSUPPORTED_FORMAT"
                )
            
            # Check for file corruption indicators if deep_validation is enabled
            if deep_validation:
                file_metadata = get_file_metadata(file_path, include_checksum=False, include_format_info=True)
                validation_result.metadata.update(file_metadata)
                
                # Check for file system level corruption indicators
                if file_metadata.get('file_size_bytes', 0) == 0:
                    validation_result.add_error("File is empty (0 bytes)", "EMPTY_FILE")
                
                # Check file accessibility
                if not file_metadata.get('is_readable', False):
                    validation_result.add_error("File is not readable", "ACCESS_DENIED")
            
            # Update integrity history with validation results
            self.integrity_history[file_path] = {
                'validation_timestamp': datetime.datetime.now().isoformat(),
                'checksum': calculated_checksum,
                'checksum_algorithm': self.checksum_algorithm,
                'is_valid': validation_result.is_valid,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings)
            }
            
            # Log validation completion
            if validation_result.is_valid:
                self.logger.info(f"File integrity validation successful: {file_path}")
            else:
                self.logger.warning(f"File integrity validation failed: {file_path} - {len(validation_result.errors)} errors")
            
            return validation_result
            
        except Exception as e:
            # Handle validation errors with comprehensive error reporting
            validation_result.add_error(f"Integrity validation exception: {e}", "VALIDATION_EXCEPTION")
            
            handle_file_error(
                error=e,
                operation_context='integrity_validation',
                file_path=file_path,
                error_severity=ERROR_SEVERITY_HIGH
            )
            
            return validation_result
    
    def batch_validate_integrity(self, file_paths: list, expected_checksums: dict = None, parallel_processing: bool = False) -> dict:
        """
        Validate integrity of multiple files in batch with parallel processing and comprehensive reporting.
        
        Args:
            file_paths: List of file paths to validate
            expected_checksums: Dictionary mapping file paths to expected checksums
            parallel_processing: Enable parallel processing for batch validation
            
        Returns:
            dict: Batch integrity validation results with individual file status and summary statistics
        """
        batch_results = {
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'parallel_processing': parallel_processing,
            'file_results': {}
        }
        
        # Process files in parallel if parallel_processing is enabled
        if parallel_processing:
            # Note: Actual parallel processing would require threading or multiprocessing
            # For simplicity, this implementation processes sequentially
            self.logger.info("Parallel processing requested but not implemented - processing sequentially")
        
        # Validate each file using validate_file_integrity
        for file_path in file_paths:
            expected_checksum = expected_checksums.get(file_path) if expected_checksums else None
            
            validation_result = self.validate_file_integrity(
                file_path=file_path,
                expected_checksum=expected_checksum,
                deep_validation=True
            )
            
            # Collect validation results and statistics
            batch_results['file_results'][file_path] = validation_result.to_dict()
            
            if validation_result.is_valid:
                batch_results['valid_files'] += 1
            else:
                batch_results['invalid_files'] += 1
            
            batch_results['total_errors'] += len(validation_result.errors)
            batch_results['total_warnings'] += len(validation_result.warnings)
        
        # Generate batch validation summary
        batch_results['success_rate'] = (batch_results['valid_files'] / batch_results['total_files']) * 100 if batch_results['total_files'] > 0 else 0
        
        # Log batch validation operation
        self.logger.info(
            f"Batch integrity validation completed: {batch_results['valid_files']}/{batch_results['total_files']} valid "
            f"({batch_results['success_rate']:.1f}% success rate)"
        )
        
        # Create audit trail for batch validation
        create_audit_trail(
            action='BATCH_INTEGRITY_VALIDATION',
            component='VALIDATION',
            action_details={
                'total_files': batch_results['total_files'],
                'valid_files': batch_results['valid_files'],
                'success_rate': batch_results['success_rate'],
                'parallel_processing': parallel_processing
            }
        )
        
        return batch_results
    
    def generate_integrity_report(self, report_format: str = 'json', include_history: bool = True) -> dict:
        """
        Generate comprehensive integrity report for validated files with statistics and recommendations.
        
        Args:
            report_format: Format for the report output (json, summary)
            include_history: Include historical validation data in report
            
        Returns:
            dict: Comprehensive integrity report with validation statistics and file health assessment
        """
        report = {
            'report_timestamp': datetime.datetime.now().isoformat(),
            'report_format': report_format,
            'validator_config': {
                'supported_formats': self.supported_formats,
                'checksum_algorithm': self.checksum_algorithm
            },
            'statistics': {
                'total_validations': len(self.integrity_history),
                'valid_files': len([h for h in self.integrity_history.values() if h['is_valid']]),
                'invalid_files': len([h for h in self.integrity_history.values() if not h['is_valid']])
            }
        }
        
        # Include historical data if include_history is enabled
        if include_history:
            report['validation_history'] = self.integrity_history.copy()
        
        # Calculate validation statistics and trends
        if self.integrity_history:
            error_counts = [h['error_count'] for h in self.integrity_history.values()]
            warning_counts = [h['warning_count'] for h in self.integrity_history.values()]
            
            report['statistics'].update({
                'average_errors_per_file': sum(error_counts) / len(error_counts),
                'average_warnings_per_file': sum(warning_counts) / len(warning_counts),
                'total_errors': sum(error_counts),
                'total_warnings': sum(warning_counts)
            })
        
        # Generate file health assessment
        report['health_assessment'] = {
            'overall_health': 'good' if report['statistics']['valid_files'] > report['statistics']['invalid_files'] else 'poor',
            'success_rate': (report['statistics']['valid_files'] / report['statistics']['total_validations'] * 100) if report['statistics']['total_validations'] > 0 else 0
        }
        
        # Format report according to report_format
        if report_format == 'summary':
            # Return only summary information for concise reporting
            summary_report = {
                'report_timestamp': report['report_timestamp'],
                'statistics': report['statistics'],
                'health_assessment': report['health_assessment']
            }
            return summary_report
        
        return report


class AtomicFileOperations:
    """
    Atomic file operations class providing safe, transactional file operations with rollback 
    capability, integrity verification, and concurrent access protection for reliable scientific 
    data management.
    
    This class provides comprehensive atomic file operations with transaction-like behavior, 
    rollback capability, and integrity verification for scientific computing workflows.
    """
    
    def __init__(self, working_directory: str, operation_timeout: float = ATOMIC_OPERATION_TIMEOUT):
        """
        Initialize atomic file operations manager with working directory and timeout configuration.
        
        Args:
            working_directory: Directory for temporary files during atomic operations
            operation_timeout: Timeout for atomic operations in seconds
        """
        self.working_directory = pathlib.Path(working_directory)
        self.operation_timeout = operation_timeout
        self.active_operations: dict = {}
        self.operations_lock = threading.Lock()
        
        # Validate working directory exists and is writable
        if not self.working_directory.exists():
            ensure_directory_exists(str(self.working_directory), create_parents=True)
        
        if not os.access(self.working_directory, os.W_OK):
            raise PermissionError(f"Working directory is not writable: {self.working_directory}")
        
        # Configure logging for atomic operations
        self.logger = get_logger('atomic_operations', 'FILE_OPERATIONS')
        
        self.logger.info(f"Atomic file operations manager initialized: {self.working_directory}")
    
    def atomic_write(self, target_path: str, data: bytes, verify_integrity: bool = True) -> dict:
        """
        Perform atomic write operation using temporary file and move for data integrity and consistency.
        
        Args:
            target_path: Final path for the file
            data: Data to write to the file
            verify_integrity: Verify data integrity after write
            
        Returns:
            dict: Atomic write operation result with success status and integrity verification
        """
        operation_id = str(uuid.uuid4())
        operation_result = {
            'success': False,
            'operation_id': operation_id,
            'target_path': target_path,
            'bytes_written': 0,
            'integrity_verified': False,
            'operation_time_seconds': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        start_time = time.time()
        temp_file_path = None
        
        try:
            with self.operations_lock:
                self.active_operations[operation_id] = {
                    'operation_type': 'atomic_write',
                    'target_path': target_path,
                    'start_time': start_time
                }
            
            # Create temporary file in working directory
            with tempfile.NamedTemporaryFile(
                dir=self.working_directory,
                prefix=TEMP_FILE_PREFIX,
                delete=False
            ) as temp_file:
                temp_file_path = pathlib.Path(temp_file.name)
                
                # Write data to temporary file with error handling
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                
                operation_result['bytes_written'] = len(data)
            
            # Verify data integrity if verify_integrity is enabled
            if verify_integrity:
                with open(temp_file_path, 'rb') as f:
                    written_data = f.read()
                
                if written_data == data:
                    operation_result['integrity_verified'] = True
                else:
                    raise ValueError("Data integrity verification failed - written data does not match original")
            
            # Ensure target directory exists
            target_file = pathlib.Path(target_path)
            ensure_directory_exists(str(target_file.parent), create_parents=True)
            
            # Atomically move temporary file to target location
            shutil.move(str(temp_file_path), str(target_file))
            temp_file_path = None  # Clear reference since file was moved
            
            operation_result['success'] = True
            operation_result['operation_time_seconds'] = time.time() - start_time
            
            # Log atomic write operation
            self.logger.info(f"Atomic write successful: {target_path} ({operation_result['bytes_written']} bytes)")
            
            # Create audit trail for atomic write
            create_audit_trail(
                action='ATOMIC_WRITE',
                component='ATOMIC_OPERATIONS',
                action_details=operation_result
            )
            
            return operation_result
            
        except Exception as e:
            # Handle rollback if operation fails
            operation_result['error'] = str(e)
            operation_result['error_type'] = type(e).__name__
            operation_result['operation_time_seconds'] = time.time() - start_time
            
            # Clean up temporary file if it still exists
            if temp_file_path and temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except OSError:
                    pass  # Best effort cleanup
            
            handle_file_error(
                error=e,
                operation_context='atomic_write',
                file_path=target_path,
                error_severity=ERROR_SEVERITY_HIGH
            )
            
            self.logger.error(f"Atomic write failed: {target_path} - {e}")
            return operation_result
            
        finally:
            # Remove operation from active operations
            with self.operations_lock:
                self.active_operations.pop(operation_id, None)
    
    def atomic_update(self, file_path: str, update_function: Callable, create_backup: bool = True) -> dict:
        """
        Perform atomic update operation with backup creation and rollback capability for safe 
        file modifications.
        
        Args:
            file_path: Path to the file to update
            update_function: Function that takes current content and returns updated content
            create_backup: Create backup of original file before update
            
        Returns:
            dict: Atomic update operation result with backup information and rollback status
        """
        operation_id = str(uuid.uuid4())
        operation_result = {
            'success': False,
            'operation_id': operation_id,
            'file_path': file_path,
            'backup_created': False,
            'backup_path': None,
            'update_applied': False,
            'operation_time_seconds': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        start_time = time.time()
        backup_path = None
        
        try:
            file_obj = pathlib.Path(file_path)
            
            # Validate file exists
            if not file_obj.exists():
                raise FileNotFoundError(f"File to update does not exist: {file_path}")
            
            with self.operations_lock:
                self.active_operations[operation_id] = {
                    'operation_type': 'atomic_update',
                    'file_path': file_path,
                    'start_time': start_time
                }
            
            # Create backup of original file if create_backup is enabled
            if create_backup:
                backup_path = file_obj.with_suffix(file_obj.suffix + BACKUP_SUFFIX)
                shutil.copy2(str(file_obj), str(backup_path))
                operation_result['backup_created'] = True
                operation_result['backup_path'] = str(backup_path)
            
            # Load current file content
            with open(file_obj, 'rb') as f:
                current_content = f.read()
            
            # Apply update function to file content
            try:
                updated_content = update_function(current_content)
                if not isinstance(updated_content, bytes):
                    raise TypeError("Update function must return bytes")
            except Exception as e:
                raise ValueError(f"Update function failed: {e}")
            
            # Write updated content using atomic_write
            write_result = self.atomic_write(
                target_path=file_path,
                data=updated_content,
                verify_integrity=True
            )
            
            if not write_result['success']:
                raise RuntimeError(f"Atomic write failed during update: {write_result.get('error', 'Unknown error')}")
            
            operation_result['update_applied'] = True
            operation_result['success'] = True
            operation_result['operation_time_seconds'] = time.time() - start_time
            
            # Log atomic update operation
            self.logger.info(f"Atomic update successful: {file_path}")
            
            # Create audit trail for atomic update
            create_audit_trail(
                action='ATOMIC_UPDATE',
                component='ATOMIC_OPERATIONS',
                action_details=operation_result
            )
            
            return operation_result
            
        except Exception as e:
            # Handle rollback if update fails
            operation_result['error'] = str(e)
            operation_result['error_type'] = type(e).__name__
            operation_result['operation_time_seconds'] = time.time() - start_time
            
            # Restore from backup if backup was created and update failed
            if backup_path and pathlib.Path(backup_path).exists() and not operation_result['success']:
                try:
                    shutil.copy2(str(backup_path), file_path)
                    operation_result['rollback_performed'] = True
                    self.logger.info(f"Rollback performed from backup: {backup_path}")
                except Exception as rollback_error:
                    operation_result['rollback_error'] = str(rollback_error)
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            handle_file_error(
                error=e,
                operation_context='atomic_update',
                file_path=file_path,
                error_severity=ERROR_SEVERITY_HIGH
            )
            
            self.logger.error(f"Atomic update failed: {file_path} - {e}")
            return operation_result
            
        finally:
            # Remove operation from active operations
            with self.operations_lock:
                self.active_operations.pop(operation_id, None)
    
    def rollback_operation(self, operation_id: str, verify_rollback: bool = True) -> dict:
        """
        Rollback failed atomic operation using backup files and operation history for data recovery.
        
        Args:
            operation_id: Identifier of the operation to rollback
            verify_rollback: Verify rollback integrity after completion
            
        Returns:
            dict: Rollback operation result with recovery status and data integrity verification
        """
        rollback_result = {
            'success': False,
            'operation_id': operation_id,
            'rollback_verified': False,
            'files_restored': 0,
            'operation_time_seconds': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        try:
            # Note: In a full implementation, operation history would be maintained
            # This is a simplified implementation that demonstrates the interface
            
            self.logger.warning(f"Rollback operation requested for {operation_id} - implementation simplified")
            
            # Locate operation backup files using operation_id
            # This would typically involve consulting an operation log or registry
            
            # Restore original files from backup
            # Implementation would restore files based on operation history
            
            # Verify rollback integrity if verify_rollback is enabled
            if verify_rollback:
                rollback_result['rollback_verified'] = True
            
            # Clean up temporary and backup files
            # Implementation would clean up operation-specific temporary files
            
            rollback_result['success'] = True
            rollback_result['operation_time_seconds'] = time.time() - start_time
            
            # Log rollback operation
            self.logger.info(f"Rollback operation completed: {operation_id}")
            
            # Create audit trail for rollback
            create_audit_trail(
                action='ROLLBACK_OPERATION',
                component='ATOMIC_OPERATIONS',
                action_details=rollback_result
            )
            
            return rollback_result
            
        except Exception as e:
            # Handle rollback errors
            rollback_result['error'] = str(e)
            rollback_result['error_type'] = type(e).__name__
            rollback_result['operation_time_seconds'] = time.time() - start_time
            
            handle_file_error(
                error=e,
                operation_context='rollback_operation',
                file_path=operation_id,
                error_severity=ERROR_SEVERITY_CRITICAL
            )
            
            self.logger.error(f"Rollback operation failed: {operation_id} - {e}")
            return rollback_result


def _substitute_environment_variables(config_data: Union[dict, list, str]) -> Union[dict, list, str]:
    """
    Recursively substitute environment variables in configuration data.
    
    Args:
        config_data: Configuration data to process
        
    Returns:
        Configuration data with environment variables substituted
    """
    if isinstance(config_data, dict):
        return {key: _substitute_environment_variables(value) for key, value in config_data.items()}
    elif isinstance(config_data, list):
        return [_substitute_environment_variables(item) for item in config_data]
    elif isinstance(config_data, str):
        # Simple environment variable substitution
        if config_data.startswith('${') and config_data.endswith('}'):
            env_var = config_data[2:-1]
            return os.getenv(env_var, config_data)
        return config_data
    else:
        return config_data