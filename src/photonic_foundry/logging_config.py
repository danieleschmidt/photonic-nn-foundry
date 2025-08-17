"""
Comprehensive logging configuration for PhotonicFoundry
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_file_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        enable_file_logging: Whether to enable file logging
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Clear existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "photonic_foundry.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogFilter(logging.Filter):
    """Filter for adding structured data to log records."""
    
    def __init__(self, extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.extra_fields = extra_fields or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add extra fields to log record."""
        for key, value in self.extra_fields.items():
            setattr(record, key, value)
        return True


def configure_component_logger(
    component_name: str,
    level: str = "INFO",
    extra_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Configure a logger for a specific component.
    
    Args:
        component_name: Name of the component
        level: Logging level
        extra_fields: Additional fields to include in logs
        
    Returns:
        Configured component logger
    """
    logger = logging.getLogger(f"photonic_foundry.{component_name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    if extra_fields:
        logger.addFilter(StructuredLogFilter(extra_fields))
    
    return logger


# Pre-configured component loggers
quantum_logger = configure_component_logger("quantum", extra_fields={"component": "quantum"})
security_logger = configure_component_logger("security", extra_fields={"component": "security"})
performance_logger = configure_component_logger("performance", extra_fields={"component": "performance"})
resilience_logger = configure_component_logger("resilience", extra_fields={"component": "resilience"})