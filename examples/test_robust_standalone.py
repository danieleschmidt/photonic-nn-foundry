#!/usr/bin/env python3
"""
Standalone test of robust functionality without external dependencies.
"""

import sys
import os
import random
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inline validation and error handling classes for testing

class ValidationLevel(Enum):
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"

class ValidationError(Exception):
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)

class CircuitValidator:
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.supported_components = {
            'mach_zehnder_interferometer',
            'ring_resonator',
            'waveguide',
            'photodetector',
            'electro_optic_modulator'
        }
        self.supported_layer_types = {
            'linear', 'activation', 'conv2d', 'pooling'
        }
        
    def validate_circuit(self, circuit_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Validate required fields
        required_fields = ['name', 'layers', 'total_components']
        for field in required_fields:
            if field not in circuit_data:
                result.add_error(f"Missing required field: {field}")
            elif circuit_data[field] is None:
                result.add_error(f"Field '{field}' cannot be None")
        
        # Validate layers
        layers = circuit_data.get('layers', [])
        if not isinstance(layers, list):
            result.add_error("Layers must be a list")
        elif len(layers) == 0:
            result.add_error("Circuit must have at least one layer")
        else:
            for i, layer in enumerate(layers):
                layer_type = layer.get('type', '')
                if layer_type not in self.supported_layer_types:
                    result.add_error(f"Layer {i}: Unsupported layer type '{layer_type}'")
                
                input_size = layer.get('input_size')
                output_size = layer.get('output_size')
                if not isinstance(input_size, int) or input_size <= 0:
                    result.add_error(f"Layer {i}: input_size must be positive integer")
                if not isinstance(output_size, int) or output_size <= 0:
                    result.add_error(f"Layer {i}: output_size must be positive integer")
        
        # Validate wavelength
        wavelength = circuit_data.get('wavelength', 1550)
        if not isinstance(wavelength, (int, float)):
            result.add_error("Wavelength must be a number")
        elif wavelength < 1260 or wavelength > 1625:
            result.add_warning(f"Wavelength {wavelength}nm outside typical range (1260-1625nm)")
        
        return result

class DataSanitizer:
    @staticmethod
    def sanitize_string(value: Any, max_length: int = 255) -> str:
        if value is None:
            return ""
        str_value = str(value).strip()
        str_value = ''.join(char for char in str_value if ord(char) >= 32)
        if len(str_value) > max_length:
            str_value = str_value[:max_length]
        return str_value
        
    @staticmethod
    def sanitize_number(value: Any, min_val: float = None, max_val: float = None) -> Union[int, float]:
        if value is None:
            return 0
        try:
            if isinstance(value, str):
                clean_value = ''.join(c for c in value if c.isdigit() or c in '.-')
                if '.' in clean_value:
                    num_value = float(clean_value)
                else:
                    num_value = int(clean_value) if clean_value else 0
            else:
                num_value = float(value)
            
            if min_val is not None and num_value < min_val:
                num_value = min_val
            if max_val is not None and num_value > max_val:
                num_value = max_val
            return num_value
        except (ValueError, TypeError):
            return 0

class SecurityValidator:
    def __init__(self):
        self.dangerous_patterns = [
            'script', 'javascript', 'eval', 'exec', 'import',
            '__', 'os.', 'sys.', 'subprocess', 'open(',
        ]
        
    def validate_safe_input(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        self._check_dangerous_patterns(data, result)
        return result
        
    def _check_dangerous_patterns(self, data: Any, result: ValidationResult, path: str = ""):
        if isinstance(data, str):
            for pattern in self.dangerous_patterns:
                if pattern.lower() in data.lower():
                    result.add_error(f"Dangerous pattern '{pattern}' found in {path}")
        elif isinstance(data, dict):
            for key, value in data.items():
                self._check_dangerous_patterns(value, result, f"{path}.{key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_dangerous_patterns(item, result, f"{path}[{i}]")

# Error handling classes
class ErrorSeverity(Enum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorCategory(Enum):
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
    def __init__(self, message: str, error_info: ErrorInfo = None):
        super().__init__(message)
        self.error_info = error_info or ErrorInfo(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            message=message
        )

class ValidationException(PhotonicException):
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

class ResourceException(PhotonicException):
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

class ErrorHandler:
    def __init__(self):
        self.error_log: List[ErrorInfo] = []
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        if isinstance(error, PhotonicException):
            error_info = error.error_info
        else:
            error_info = self._classify_standard_error(error)
            
        if context:
            error_info.context = {**(error_info.context or {}), **context}
            
        self._log_error(error_info)
        self.error_log.append(error_info)
        return error_info
        
    def _classify_standard_error(self, error: Exception) -> ErrorInfo:
        error_type = type(error).__name__
        message = str(error)
        
        if isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.ERROR
            recoverable = True
        elif isinstance(error, (MemoryError, OSError)):
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.CRITICAL
            recoverable = False
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
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
    def attempt_recovery(self, error_info: ErrorInfo, original_inputs: Dict[str, Any] = None) -> Optional[Any]:
        if not error_info.recoverable:
            logger.warning(f"Error not recoverable: {error_info.message}")
            return None
            
        # Simple recovery strategy for demonstration
        if error_info.category == ErrorCategory.VALIDATION and original_inputs:
            sanitized_inputs = original_inputs.copy()
            
            # Fix common issues
            for key, value in sanitized_inputs.items():
                if value is None:
                    if 'size' in key:
                        sanitized_inputs[key] = 1
                    elif 'count' in key:
                        sanitized_inputs[key] = 0
                    elif 'name' in key:
                        sanitized_inputs[key] = 'default'
                        
            return sanitized_inputs
        
        return None
        
    def get_error_summary(self) -> Dict[str, Any]:
        if not self.error_log:
            return {'total_errors': 0, 'categories': {}, 'severities': {}}
            
        categories = {}
        severities = {}
        
        for error in self.error_log:
            cat = error.category.value
            categories[cat] = categories.get(cat, 0) + 1
            
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
                for error in self.error_log[-5:]
            ]
        }

# Test functions

def create_invalid_circuit() -> Dict[str, Any]:
    return {
        'layers': [
            {
                'type': 'unsupported_layer',
                'input_size': -5,
                'output_size': 'not_a_number',
                'components': 'not_a_list'
            },
            {
                'type': 'linear',
                'input_size': 10,
                'output_size': 5,
                'components': []
            }
        ],
        'connections': [(0, 2), (-1, 1)],
        'total_components': 'wrong_type',
        'wavelength': 5000,
        'dangerous_field': 'javascript:alert("hack")'
    }

def demonstrate_validation():
    print("=== Validation Demonstration ===")
    
    validator = CircuitValidator(validation_level=ValidationLevel.NORMAL)
    
    print("\n1. Testing with invalid circuit...")
    invalid_circuit = create_invalid_circuit()
    result = validator.validate_circuit(invalid_circuit)
    
    print(f"   Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"   Errors: {len(result.errors)}")
    for error in result.errors:
        print(f"     - {error}")
    print(f"   Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"     - {warning}")
    
    print("\n2. Testing data sanitization...")
    sanitizer = DataSanitizer()
    
    dirty_data = {
        'name': '  Circuit<script>alert("hack")</script>  ',
        'size': 'not_a_number',
        'large_number': 999999999999999
    }
    
    cleaned_data = {
        'name': sanitizer.sanitize_string(dirty_data['name'], max_length=32),
        'size': sanitizer.sanitize_number(dirty_data['size'], min_val=1, max_val=1024),
        'large_number': sanitizer.sanitize_number(dirty_data['large_number'], max_val=1000000)
    }
    
    print(f"   Original: {dirty_data}")
    print(f"   Cleaned:  {cleaned_data}")

def demonstrate_error_handling():
    print("\n=== Error Handling Demonstration ===")
    
    error_handler = ErrorHandler()
    
    print("\n1. Testing error classification...")
    
    test_errors = [
        ValueError("Invalid input value"),
        ValidationException("Circuit validation failed", field="name"),
        ComputationException("Matrix computation failed", operation="multiply"),
        ResourceException("Memory exhausted", resource_type="RAM", requested=1000, available=500)
    ]
    
    for error in test_errors:
        error_info = error_handler.handle_error(error)
        print(f"   {type(error).__name__}: {error_info.category.value}, recoverable: {error_info.recoverable}")

def demonstrate_security_validation():
    print("\n=== Security Validation Demonstration ===")
    
    security_validator = SecurityValidator()
    
    dangerous_data = {
        'circuit_name': 'test_circuit',
        'description': 'A circuit with <script>alert("xss")</script> embedded',
        'config': {
            'eval_string': 'eval("malicious_code()")',
            'import_statement': 'import os; os.system("rm -rf /")',
            'nested': {
                'deep': {
                    'very_deep': '__import__("subprocess").call(["ls"])'
                }
            }
        }
    }
    
    security_result = security_validator.validate_safe_input(dangerous_data)
    
    print(f"   Security validation: {'PASS' if security_result.is_valid else 'FAIL'}")
    print(f"   Security errors: {len(security_result.errors)}")
    for error in security_result.errors:
        print(f"     - {error}")

def demonstrate_error_recovery():
    print("\n=== Error Recovery Demonstration ===")
    
    error_handler = ErrorHandler()
    
    recoverable_circuit = {
        'name': None,
        'layers': [
            {
                'type': 'linear',
                'input_size': None,
                'output_size': 10,
                'components': None
            }
        ],
        'total_components': None,
        'connections': [],
        'wavelength': 1550
    }
    
    print("\n1. Original circuit with issues:")
    print(f"   Name: {recoverable_circuit['name']}")
    print(f"   Layer 0 input_size: {recoverable_circuit['layers'][0]['input_size']}")
    
    try:
        validator = CircuitValidator()
        result = validator.validate_circuit(recoverable_circuit)
        if not result.is_valid:
            raise ValidationException("Circuit has validation errors")
            
    except Exception as error:
        error_info = error_handler.handle_error(error)
        
        recovered_data = error_handler.attempt_recovery(error_info, recoverable_circuit)
        
        if recovered_data:
            print("\n2. After recovery:")
            print(f"   Name: {recovered_data.get('name', 'N/A')}")
            print(f"   Layer 0 input_size: {recovered_data['layers'][0].get('input_size', 'N/A')}")
            
            recovery_result = validator.validate_circuit(recovered_data)
            print(f"   Validation after recovery: {'PASS' if recovery_result.is_valid else 'FAIL'}")
        else:
            print("   Recovery failed")

def main():
    print("=== PhotonicFoundry Robust Usage Demo (Standalone) ===")
    
    demonstrate_validation()
    demonstrate_error_handling()
    demonstrate_security_validation()
    demonstrate_error_recovery()
    
    print("\n=== Error Summary ===")
    error_handler = ErrorHandler()
    
    test_errors = [
        ValidationException("Test validation error", field="test_field"),
        ComputationException("Test computation error", operation="test_op"),
        ResourceException("Test resource error", resource_type="memory")
    ]
    
    for error in test_errors:
        error_handler.handle_error(error)
    
    summary = error_handler.get_error_summary()
    print(f"Total errors logged: {summary['total_errors']}")
    print(f"Error categories: {summary['categories']}")
    print(f"Error severities: {summary['severities']}")
    
    print("\nRecent errors:")
    for error in summary['recent_errors']:
        print(f"  - [{error['severity'].upper()}] {error['message']}")
    
    print("\n=== Robust Demo Complete ===")
    print("PhotonicFoundry demonstrated comprehensive error handling!")

if __name__ == "__main__":
    main()