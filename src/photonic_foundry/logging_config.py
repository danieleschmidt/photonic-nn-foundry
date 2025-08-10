"""
Comprehensive structured logging configuration for photonic foundry systems.
"""

import os
import sys
import json
import logging
import logging.config
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
import uuid
import threading
from contextlib import contextmanager
import time
import functools


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for consistent log output.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in log_entry and not key.startswith('_'):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
                        
        return json.dumps(log_entry, default=str)


class SecurityFilter(logging.Filter):
    """
    Filter to prevent logging of sensitive information.
    """
    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            'password', 'token', 'key', 'secret', 'credential',
            'authorization', 'auth', 'bearer', 'api_key'
        ]
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from log records."""
        message = record.getMessage().lower()
        
        # Check for sensitive patterns in message
        for pattern in self.sensitive_patterns:
            if pattern in message:
                # Replace sensitive data with placeholder
                record.msg = record.msg.replace(
                    record.args[0] if record.args else str(record.msg),
                    '[REDACTED]'
                )
                break
                
        return True


class PhotonicLogger:
    """
    Enhanced logger for photonic foundry operations.
    """
    
    def __init__(self, name: str, component: str = None):
        self.name = name
        self.component = component or name
        self.logger = logging.getLogger(name)
        self._request_id = None
        
    def set_request_id(self, request_id: str):
        """Set request ID for tracing."""
        self._request_id = request_id
        
    def _add_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add context information to log entries."""
        context = {
            'component': self.component,
        }
        
        if self._request_id:
            context['request_id'] = self._request_id
            
        if extra:
            context.update(extra)
            
        return context
        
    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Log debug message with context."""
        self.logger.debug(message, extra=self._add_context(extra))
        
    def info(self, message: str, extra: Dict[str, Any] = None):
        """Log info message with context."""
        self.logger.info(message, extra=self._add_context(extra))
        
    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Log warning message with context."""
        self.logger.warning(message, extra=self._add_context(extra))
        
    def error(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log error message with context."""
        self.logger.error(message, extra=self._add_context(extra), exc_info=exc_info)
        
    def critical(self, message: str, extra: Dict[str, Any] = None, exc_info: bool = False):
        """Log critical message with context."""
        self.logger.critical(message, extra=self._add_context(extra), exc_info=exc_info)
        
    def log_operation(self, operation: str, duration: float = None, 
                     success: bool = True, extra: Dict[str, Any] = None):
        """Log operation completion with metrics."""
        context = {
            'operation': operation,
            'duration_ms': duration * 1000 if duration else None,
            'success': success,
            'operation_id': str(uuid.uuid4())
        }
        
        if extra:
            context.update(extra)
            
        level = logging.INFO if success else logging.ERROR
        message = f"Operation '{operation}' {'completed successfully' if success else 'failed'}"
        
        if duration:
            message += f" in {duration:.3f}s"
            
        self.logger.log(level, message, extra=self._add_context(context))
        
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        context = {
            'metrics_type': 'performance',
            'metrics': metrics
        }
        
        self.info("Performance metrics collected", extra=context)


def setup_logging(
    log_level: str = None,
    log_format: str = "structured",  # structured, simple, detailed
    log_file: str = None,
    enable_console: bool = True,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    enable_security_filter: bool = True
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format type
        log_file: Log file path
        enable_console: Enable console logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        enable_security_filter: Enable security filtering
    """
    
    # Determine log level
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
    # Create log directory if needed
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
    # Configure formatters
    formatters = {
        'structured': StructuredFormatter(),
        'simple': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        'detailed': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
    }
    
    formatter = formatters.get(log_format, formatters['structured'])
    
    # Configure handlers
    handlers = {}
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level))
        
        if enable_security_filter:
            console_handler.addFilter(SecurityFilter())
            
        handlers['console'] = console_handler
        
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level))
        
        if enable_security_filter:
            file_handler.addFilter(SecurityFilter())
            
        handlers['file'] = file_handler
        
    # Error file handler (errors and above only)
    if log_file:
        error_log_file = str(Path(log_file).with_suffix('.error.log'))
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        if enable_security_filter:
            error_handler.addFilter(SecurityFilter())
            
        handlers['error'] = error_handler
        
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        handlers=list(handlers.values()),
        force=True
    )
    
    # Set specific logger levels
    logger_configs = {
        'photonic_foundry': log_level,
        'uvicorn': 'INFO',
        'fastapi': 'INFO',
        'sqlalchemy.engine': 'WARNING',  # Reduce SQL query noise
        'urllib3.connectionpool': 'WARNING',
    }
    
    for logger_name, level in logger_configs.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))
        
    # Log configuration
    root_logger = logging.getLogger()
    root_logger.info(
        f"Logging configured - Level: {log_level}, Format: {log_format}, "
        f"Console: {enable_console}, File: {log_file is not None}"
    )


def get_logger(name: str, component: str = None) -> PhotonicLogger:
    """
    Get a PhotonicLogger instance.
    
    Args:
        name: Logger name
        component: Component name for context
        
    Returns:
        PhotonicLogger instance
    """
    return PhotonicLogger(name, component)


def configure_request_logging():
    """Configure request-specific logging middleware."""
    import contextvars
    
    # Context variable for request ID
    request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id')
    
    class RequestContextFilter(logging.Filter):
        def filter(self, record):
            try:
                record.request_id = request_id_var.get()
            except LookupError:
                record.request_id = None
            return True
    
    # Add filter to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestContextFilter())
        
    return request_id_var


class LoggingContext:
    """Context manager for enhanced logging with additional context."""
    
    def __init__(self, logger: PhotonicLogger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
        self.operation_id = str(uuid.uuid4())
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        context = {**self.context, 'operation_id': self.operation_id}
        self.logger.info(f"Starting operation: {self.operation}", extra=context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        success = exc_type is None
        
        context = {
            **self.context, 
            'duration_seconds': duration,
            'operation_id': self.operation_id,
            'success': success
        }
        
        if success:
            self.logger.info(f"Completed operation: {self.operation}", extra=context)
        else:
            context['exception_type'] = exc_type.__name__ if exc_type else None
            context['exception_message'] = str(exc_val) if exc_val else None
            self.logger.error(f"Failed operation: {self.operation}", extra=context, exc_info=True)


class MetricsLogger:
    """Logger for performance and business metrics."""
    
    def __init__(self, name: str = "photonic_foundry.metrics"):
        self.logger = logging.getLogger(name)
        
    def log_timing(self, operation: str, duration_ms: float, **kwargs):
        """Log timing metrics."""
        self.logger.info("TIMING", extra={
            'metric_type': 'timing',
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
        
    def log_counter(self, metric_name: str, value: int = 1, **kwargs):
        """Log counter metrics."""
        self.logger.info("COUNTER", extra={
            'metric_type': 'counter',
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
        
    def log_gauge(self, metric_name: str, value: float, **kwargs):
        """Log gauge metrics."""
        self.logger.info("GAUGE", extra={
            'metric_type': 'gauge',
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
        
    def log_error_rate(self, operation: str, error_count: int, total_count: int, **kwargs):
        """Log error rate metrics."""
        error_rate = error_count / total_count if total_count > 0 else 0
        self.logger.info("ERROR_RATE", extra={
            'metric_type': 'error_rate',
            'operation': operation,
            'error_count': error_count,
            'total_count': total_count,
            'error_rate': error_rate,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })


class PerformanceLogger:
    """Logger for detailed performance tracking."""
    
    def __init__(self):
        self.metrics_logger = MetricsLogger()
        self._thread_local = threading.local()
        
    @contextmanager
    def measure_time(self, operation: str, **kwargs):
        """Context manager to measure and log operation time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics_logger.log_timing(operation, duration_ms, **kwargs)
            
    def measure_function(self, operation_name: str = None, log_args: bool = False):
        """Decorator to measure function execution time."""
        def decorator(func):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                # Log function entry
                extra = {'operation': op_name}
                if log_args:
                    extra['args_count'] = len(args)
                    extra['kwargs_keys'] = list(kwargs.keys())
                    
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Log success
                    self.metrics_logger.log_timing(op_name, duration_ms, success=True, **extra)
                    return result
                    
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Log failure
                    extra.update({
                        'success': False,
                        'exception_type': type(e).__name__,
                        'exception_message': str(e)
                    })
                    self.metrics_logger.log_timing(op_name, duration_ms, **extra)
                    raise
                    
            return wrapper
        return decorator


class AuditLogger:
    """Logger for security and audit events."""
    
    def __init__(self, name: str = "photonic_foundry.audit"):
        self.logger = logging.getLogger(name)
        
    def log_access(self, user_id: str, resource: str, action: str, success: bool = True, **kwargs):
        """Log access attempts."""
        self.logger.info("ACCESS", extra={
            'event_type': 'access',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
        
    def log_security_event(self, event_type: str, severity: str, description: str, **kwargs):
        """Log security-related events."""
        self.logger.warning("SECURITY", extra={
            'event_type': 'security',
            'security_event_type': event_type,
            'severity': severity,
            'description': description,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })
        
    def log_data_operation(self, operation: str, data_type: str, record_count: int = None, **kwargs):
        """Log data operations for compliance."""
        self.logger.info("DATA_OP", extra={
            'event_type': 'data_operation',
            'operation': operation,
            'data_type': data_type,
            'record_count': record_count,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        })


class AlertHandler(logging.Handler):
    """Custom handler for critical alerts."""
    
    def __init__(self, alert_callback: callable = None):
        super().__init__()
        self.alert_callback = alert_callback or self._default_alert
        self.setLevel(logging.ERROR)
        
    def emit(self, record):
        """Emit alert for critical log records."""
        if record.levelno >= logging.ERROR:
            try:
                alert_data = {
                    'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line_number': record.lineno
                }
                
                if record.exc_info:
                    alert_data['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                    
                self.alert_callback(alert_data)
            except Exception as e:
                # Don't let alert failures break logging
                print(f"Alert handler error: {e}", file=sys.stderr)
                
    def _default_alert(self, alert_data):
        """Default alert implementation - just print to stderr."""
        print(f"ALERT: {alert_data['level']} - {alert_data['message']}", file=sys.stderr)


# Global instances
_performance_logger = None
_metrics_logger = None
_audit_logger = None


def get_performance_logger() -> PerformanceLogger:
    """Get global performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def get_metrics_logger() -> MetricsLogger:
    """Get global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def log_startup_info():
    """Log system startup information."""
    logger = get_logger("photonic_foundry.startup")
    logger.info("Photonic Foundry system starting up", extra={
        'python_version': sys.version,
        'platform': sys.platform,
        'working_directory': os.getcwd(),
        'process_id': os.getpid()
    })


def log_shutdown_info():
    """Log system shutdown information."""
    logger = get_logger("photonic_foundry.shutdown")
    logger.info("Photonic Foundry system shutting down", extra={
        'process_id': os.getpid()
    })


# Decorator functions for common use cases
def log_function_calls(logger_name: str = None, log_args: bool = False):
    """Decorator to log function calls."""
    def decorator(func):
        logger = get_logger(logger_name or f"{func.__module__}.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extra = {
                'function': func.__name__,
                'module': func.__module__
            }
            
            if log_args:
                extra['args_count'] = len(args)
                extra['kwargs_keys'] = list(kwargs.keys())
                
            logger.debug(f"Calling function: {func.__name__}", extra=extra)
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function completed: {func.__name__}", extra=extra)
                return result
            except Exception as e:
                extra.update({
                    'exception_type': type(e).__name__,
                    'exception_message': str(e)
                })
                logger.error(f"Function failed: {func.__name__}", extra=extra, exc_info=True)
                raise
                
        return wrapper
    return decorator


def measure_performance(operation_name: str = None):
    """Decorator to measure and log performance."""
    perf_logger = get_performance_logger()
    return perf_logger.measure_function(operation_name)


# Initialize logging on module import
if not logging.getLogger().hasHandlers():
    setup_logging()
    log_startup_info()