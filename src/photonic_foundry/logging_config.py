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
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting operation: {self.operation}", extra=self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        success = exc_type is None
        
        context = {**self.context, 'duration_seconds': duration}
        
        if success:
            self.logger.info(f"Completed operation: {self.operation}", extra=context)
        else:
            context['exception_type'] = exc_type.__name__ if exc_type else None
            context['exception_message'] = str(exc_val) if exc_val else None
            self.logger.error(f"Failed operation: {self.operation}", extra=context, exc_info=True)


# Initialize logging on module import
if not logging.getLogger().hasHandlers():
    setup_logging()