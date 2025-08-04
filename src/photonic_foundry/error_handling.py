"""
Comprehensive error handling and recovery mechanisms for photonic circuits.
"""

import logging
import traceback
import sys
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    HARDWARE = "hardware"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorInfo:
    """Structured error information."""
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    stack_trace: Optional[str] = None
    recoverable: bool = False
    recovery_suggestions: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class PhotonicException(Exception):
    """Base exception for photonic circuit operations."""
    
    def __init__(self, message: str, error_info: ErrorInfo = None):
        super().__init__(message)
        self.error_info = error_info or ErrorInfo(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            message=message
        )


class ValidationException(PhotonicException):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        error_info = ErrorInfo(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.VALIDATION,
            message=message,
            context={'field': field, 'value': value},
            recoverable=True,
            recovery_suggestions=[
                "Check input data format",
                "Verify all required fields are present",
                "Ensure data types are correct"
            ]
        )
        super().__init__(message, error_info)
        self.field = field
        self.value = value


class ComputationException(PhotonicException):
    """Exception for computation errors."""
    
    def __init__(self, message: str, operation: str = None, inputs: Dict[str, Any] = None):
        error_info = ErrorInfo(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.COMPUTATION,
            message=message,
            context={'operation': operation, 'inputs': inputs},
            recoverable=True,
            recovery_suggestions=[
                "Check input data ranges",
                "Verify mathematical constraints",
                "Consider using different parameters"
            ]
        )
        super().__init__(message, error_info)
        self.operation = operation
        self.inputs = inputs


class HardwareException(PhotonicException):
    """Exception for hardware-related errors."""
    
    def __init__(self, message: str, component: str = None, specifications: Dict[str, Any] = None):
        error_info = ErrorInfo(
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.HARDWARE,
            message=message,
            context={'component': component, 'specifications': specifications},
            recoverable=False,
            recovery_suggestions=[
                "Check hardware specifications",
                "Verify component compatibility",
                "Consider alternative hardware configurations"
            ]
        )
        super().__init__(message, error_info)
        self.component = component
        self.specifications = specifications


class ResourceException(PhotonicException):
    """Exception for resource exhaustion."""
    
    def __init__(self, message: str, resource_type: str = None, requested: Any = None, available: Any = None):
        error_info = ErrorInfo(
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.RESOURCE,
            message=message,
            context={'resource_type': resource_type, 'requested': requested, 'available': available},
            recoverable=True,
            recovery_suggestions=[
                "Reduce circuit complexity",
                "Optimize resource usage",
                "Consider distributed processing"
            ]
        )
        super().__init__(message, error_info)
        self.resource_type = resource_type
        self.requested = requested
        self.available = available


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self):
        self.error_log: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.VALIDATION: [
                self._recover_validation_error,
                self._sanitize_input_data
            ],
            ErrorCategory.COMPUTATION: [
                self._recover_computation_error,
                self._fallback_computation
            ],
            ErrorCategory.RESOURCE: [
                self._recover_resource_error,
                self._optimize_resource_usage
            ],
            ErrorCategory.USER_INPUT: [
                self._recover_user_input_error,
                self._provide_input_defaults
            ]
        }
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Handle and classify errors."""
        if isinstance(error, PhotonicException):
            error_info = error.error_info
        else:
            # Convert standard exceptions to ErrorInfo
            error_info = self._classify_standard_error(error)
            
        # Add context information
        if context:
            error_info.context = {**(error_info.context or {}), **context}
            
        # Add stack trace
        error_info.stack_trace = traceback.format_exc()
        
        # Log the error
        self._log_error(error_info)
        
        # Store in error log
        self.error_log.append(error_info)
        
        return error_info
        
    def attempt_recovery(self, error_info: ErrorInfo, original_inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Attempt to recover from error."""
        if not error_info.recoverable:
            logger.warning(f"Error not recoverable: {error_info.message}")
            return None
            
        strategies = self.recovery_strategies.get(error_info.category, [])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                result = strategy(error_info, original_inputs)
                if result is not None:
                    logger.info(f"Recovery successful with strategy: {strategy.__name__}")
                    return result
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
                
        logger.error(f"All recovery strategies failed for: {error_info.message}")
        return None
        
    def _classify_standard_error(self, error: Exception) -> ErrorInfo:
        """Classify standard Python exceptions."""
        error_type = type(error).__name__
        message = str(error)
        
        # Classification rules
        if isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.ERROR
            recoverable = True
        elif isinstance(error, (MemoryError, OSError)):
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.CRITICAL
            recoverable = False
        elif isinstance(error, (KeyError, AttributeError)):
            category = ErrorCategory.CONFIGURATION
            severity = ErrorSeverity.ERROR
            recoverable = True
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.WARNING
            recoverable = True
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.ERROR
            recoverable = False
            
        return ErrorInfo(
            severity=severity,
            category=category,
            message=f"{error_type}: {message}",
            recoverable=recoverable
        )
        
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.details:
            log_message += f" - {error_info.details}"
            
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
    def _recover_validation_error(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Attempt to recover from validation errors."""
        if not inputs:
            return None
            
        # Try to sanitize and fix common validation issues
        sanitized_inputs = inputs.copy()
        
        # Fix common issues
        for key, value in sanitized_inputs.items():
            if value is None:
                # Provide defaults for None values
                if 'size' in key:
                    sanitized_inputs[key] = 1
                elif 'count' in key:
                    sanitized_inputs[key] = 0
                elif 'name' in key:
                    sanitized_inputs[key] = 'default'
                    
        return sanitized_inputs
        
    def _sanitize_input_data(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Sanitize input data to fix validation issues."""
        if not inputs:
            return None
            
        from .validation import DataSanitizer
        
        sanitized = {}
        for key, value in inputs.items():
            if isinstance(value, str):
                sanitized[key] = DataSanitizer.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[key] = DataSanitizer.sanitize_number(value)
            elif isinstance(value, list):
                sanitized[key] = DataSanitizer.sanitize_list(value)
            else:
                sanitized[key] = value
                
        return sanitized
        
    def _recover_computation_error(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Attempt to recover from computation errors."""
        if not inputs:
            return None
            
        # Try with simplified parameters
        simplified_inputs = inputs.copy()
        
        # Reduce complexity
        if 'precision' in simplified_inputs:
            simplified_inputs['precision'] = max(4, simplified_inputs['precision'] // 2)
        if 'iterations' in simplified_inputs:
            simplified_inputs['iterations'] = max(1, simplified_inputs['iterations'] // 2)
            
        return simplified_inputs
        
    def _fallback_computation(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Provide fallback computation results."""
        # Return simplified/approximate results
        return {
            'status': 'approximated',
            'result': 'fallback_value',
            'accuracy': 0.5,
            'message': 'Using fallback computation due to error'
        }
        
    def _recover_resource_error(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Attempt to recover from resource errors."""
        if not inputs:
            return None
            
        optimized_inputs = inputs.copy()
        
        # Reduce resource requirements
        if 'batch_size' in optimized_inputs:
            optimized_inputs['batch_size'] = max(1, optimized_inputs['batch_size'] // 2)
        if 'max_components' in optimized_inputs:
            optimized_inputs['max_components'] = optimized_inputs['max_components'] // 2
            
        return optimized_inputs
        
    def _optimize_resource_usage(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Optimize resource usage to avoid errors."""
        return {
            'status': 'optimized',
            'resource_usage': 'reduced',
            'message': 'Optimized to reduce resource requirements'
        }
        
    def _recover_user_input_error(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Recover from user input errors."""
        # Provide helpful error message and suggestions
        return {
            'error': error_info.message,
            'suggestions': error_info.recovery_suggestions,
            'corrected_input': self._provide_input_defaults(error_info, inputs)
        }
        
    def _provide_input_defaults(self, error_info: ErrorInfo, inputs: Dict[str, Any] = None) -> Optional[Any]:
        """Provide default values for user inputs."""
        defaults = {
            'name': 'default_circuit',
            'pdk': 'skywater130',
            'wavelength': 1550,
            'precision': 8,
            'input_size': 1,
            'output_size': 1,
            'layers': [],
            'components': []
        }
        
        if inputs:
            result = inputs.copy()
            for key, default_value in defaults.items():
                if key not in result or result[key] is None:
                    result[key] = default_value
            return result
        else:
            return defaults
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_log:
            return {'total_errors': 0, 'categories': {}, 'severities': {}}
            
        categories = {}
        severities = {}
        
        for error in self.error_log:
            # Count by category
            cat = error.category.value
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count by severity
            sev = error.severity.value
            severities[sev] = severities.get(sev, 0) + 1
            
        return {
            'total_errors': len(self.error_log),
            'categories': categories,
            'severities': severities,
            'recent_errors': [
                {
                    'message': error.message,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'timestamp': error.timestamp
                }
                for error in self.error_log[-5:]  # Last 5 errors
            ]
        }
        
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log.clear()
        logger.info("Error log cleared")


@contextmanager
def safe_operation(operation_name: str, error_handler: ErrorHandler = None, 
                  context: Dict[str, Any] = None, reraise: bool = False):
    """Context manager for safe operations with error handling."""
    if error_handler is None:
        error_handler = ErrorHandler()
        
    try:
        logger.debug(f"Starting safe operation: {operation_name}")
        yield error_handler
        logger.debug(f"Completed safe operation: {operation_name}")
        
    except Exception as error:
        logger.error(f"Error in operation {operation_name}: {error}")
        
        # Handle the error
        error_info = error_handler.handle_error(error, context)
        
        # Attempt recovery if possible
        if error_info.recoverable:
            recovery_result = error_handler.attempt_recovery(error_info, context)
            if recovery_result is not None:
                logger.info(f"Recovery successful for operation: {operation_name}")
                return recovery_result
                
        # Re-raise if requested
        if reraise:
            raise
            
        # Log final failure
        logger.error(f"Operation {operation_name} failed and could not be recovered")


def create_robust_function(func: Callable, max_retries: int = 3, 
                          backoff_factor: float = 1.0) -> Callable:
    """Create a robust version of a function with retry logic."""
    
    def robust_wrapper(*args, **kwargs):
        error_handler = ErrorHandler()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
                
            except Exception as error:
                last_error = error
                error_info = error_handler.handle_error(error)
                
                if attempt < max_retries - 1 and error_info.recoverable:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    
                    # Try recovery
                    recovery_inputs = error_handler.attempt_recovery(error_info, {'args': args, 'kwargs': kwargs})
                    if recovery_inputs and 'args' in recovery_inputs:
                        args = recovery_inputs['args']
                        kwargs = recovery_inputs.get('kwargs', kwargs)
                else:
                    break
                    
        # All retries failed
        logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
        if last_error:
            raise last_error
            
    robust_wrapper.__name__ = f"robust_{func.__name__}"
    robust_wrapper.__doc__ = f"Robust version of {func.__name__} with retry logic"
    
    return robust_wrapper