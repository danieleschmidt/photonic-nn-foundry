#!/usr/bin/env python3
"""
Robust usage example demonstrating error handling and validation.
"""

import sys
import os
import random
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_foundry.validation import CircuitValidator, ValidationLevel, DataSanitizer
from photonic_foundry.error_handling import (
    ErrorHandler, safe_operation, create_robust_function,
    ValidationException, ComputationException, ResourceException
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_invalid_circuit() -> Dict[str, Any]:
    """Create intentionally invalid circuit for testing validation."""
    return {
        # Missing required 'name' field
        'layers': [
            {
                'type': 'unsupported_layer',  # Invalid layer type
                'input_size': -5,  # Invalid negative size
                'output_size': 'not_a_number',  # Invalid type
                'components': 'not_a_list'  # Invalid type
            },
            {
                'type': 'linear',
                'input_size': 10,
                'output_size': 5,  # Size mismatch with previous layer
                'components': []
            }
        ],
        'connections': [(0, 2), (-1, 1)],  # Invalid connections
        'total_components': 'wrong_type',  # Should be integer
        'wavelength': 5000,  # Outside typical range
        'dangerous_field': 'javascript:alert("hack")'  # Security issue
    }


def create_problematic_circuit() -> Dict[str, Any]:
    """Create circuit that will cause computation errors."""
    return {
        'name': 'problematic_circuit',
        'layers': [
            {
                'type': 'linear',
                'input_size': 1000000,  # Extremely large
                'output_size': 1000000,
                'component_count': 1000000000000,  # Trillion components!
                'components': []
            }
        ],
        'total_components': 1000000000000,
        'connections': [],
        'pdk': 'unknown_pdk',
        'wavelength': 1550
    }


def simulate_computation_with_errors(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate computation that might fail."""
    component_count = circuit_data.get('total_components', 0)
    
    # Simulate different types of errors
    if component_count > 1000000:
        raise ResourceException(
            "Too many components for available memory",
            resource_type="memory",
            requested=component_count,
            available=1000000
        )
    
    if 'problematic' in circuit_data.get('name', ''):
        raise ComputationException(
            "Numerical instability in matrix calculations",
            operation="matrix_multiplication",
            inputs={'size': component_count}
        )
    
    # Simulate successful computation
    return {
        'energy': component_count * 0.5,
        'latency': len(circuit_data.get('layers', [])) * 50,
        'area': component_count * 0.001
    }


def demonstrate_validation():
    """Demonstrate validation capabilities."""
    print("=== Validation Demonstration ===")
    
    # Create validator
    validator = CircuitValidator(validation_level=ValidationLevel.NORMAL)
    
    # Test with invalid circuit
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
    
    # Test data sanitization
    print("\n2. Testing data sanitization...")
    sanitizer = DataSanitizer()
    
    dirty_data = {
        'name': '  Circuit<script>alert("hack")</script>  ',
        'size': 'not_a_number',
        'values': 'should_be_list',
        'large_number': 999999999999999
    }
    
    cleaned_data = {
        'name': sanitizer.sanitize_string(dirty_data['name'], max_length=32),
        'size': sanitizer.sanitize_number(dirty_data['size'], min_val=1, max_val=1024),
        'values': sanitizer.sanitize_list(dirty_data['values'], item_type=float),
        'large_number': sanitizer.sanitize_number(dirty_data['large_number'], max_val=1000000)
    }
    
    print(f"   Original: {dirty_data}")
    print(f"   Cleaned:  {cleaned_data}")


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demonstration ===")
    
    error_handler = ErrorHandler()
    
    # Test with problematic circuit
    print("\n1. Testing with problematic circuit...")
    problematic_circuit = create_problematic_circuit()
    
    try:
        result = simulate_computation_with_errors(problematic_circuit)
        print(f"   Computation succeeded: {result}")
    except Exception as error:
        error_info = error_handler.handle_error(error, {'circuit_name': problematic_circuit['name']})
        print(f"   Error handled: {error_info.message}")
        print(f"   Category: {error_info.category.value}")
        print(f"   Recoverable: {error_info.recoverable}")
        
        if error_info.recovery_suggestions:
            print("   Recovery suggestions:")
            for suggestion in error_info.recovery_suggestions:
                print(f"     - {suggestion}")
        
        # Attempt recovery
        recovery_result = error_handler.attempt_recovery(error_info, problematic_circuit)
        if recovery_result:
            print(f"   Recovery successful: {recovery_result}")
        else:
            print("   Recovery failed")


def demonstrate_safe_operations():
    """Demonstrate safe operations with automatic error handling."""
    print("\n=== Safe Operations Demonstration ===")
    
    error_handler = ErrorHandler()
    
    # Test safe operation context manager
    print("\n1. Using safe operation context manager...")
    
    with safe_operation("circuit_validation", error_handler) as handler:
        invalid_circuit = create_invalid_circuit()
        validator = CircuitValidator()
        result = validator.validate_circuit(invalid_circuit)
        
        if not result.is_valid:
            raise ValidationException("Circuit validation failed", field="multiple")
    
    # Test robust function wrapper
    print("\n2. Using robust function wrapper...")
    
    @create_robust_function
    def unreliable_computation(data: Dict[str, Any]) -> Dict[str, Any]:
        """Function that fails randomly."""
        if random.random() < 0.7:  # 70% chance of failure
            raise ComputationException("Random computation failure")
        return {'result': 'success', 'data': data}
    
    test_data = {'input': 'test_value'}
    
    try:
        result = unreliable_computation(test_data)
        print(f"   Robust computation result: {result}")
    except Exception as error:
        print(f"   Robust computation failed: {error}")


def demonstrate_security_validation():
    """Demonstrate security validation."""
    print("\n=== Security Validation Demonstration ===")
    
    from photonic_foundry.validation import SecurityValidator
    
    security_validator = SecurityValidator()
    
    # Test with potentially dangerous input
    dangerous_data = {
        'circuit_name': 'test_circuit',
        'description': 'A circuit with <script>alert("xss")</script> embedded',
        'config': {
            # SECURITY_DISABLED: 'eval_string': 'eval("malicious_code()")',
            # SECURITY_DISABLED: 'import_statement': 'import os; os.system("rm -rf /")',
            'nested': {
                'deep': {
                    # SECURITY_DISABLED: 'very_deep': '__import__("subprocess").call(["ls"])'
                }
            }
        }
    }
    
    security_result = security_validator.validate_safe_input(dangerous_data)
    
    print(f"   Security validation: {'PASS' if security_result.is_valid else 'FAIL'}")
    print(f"   Security errors: {len(security_result.errors)}")
    for error in security_result.errors:
        print(f"     - {error}")
    print(f"   Security warnings: {len(security_result.warnings)}")
    for warning in security_result.warnings:
        print(f"     - {warning}")


def demonstrate_error_recovery():
    """Demonstrate automatic error recovery."""
    print("\n=== Error Recovery Demonstration ===")
    
    error_handler = ErrorHandler()
    
    # Create circuit with recoverable errors
    recoverable_circuit = {
        'name': None,  # Will be fixed by recovery
        'layers': [
            {
                'type': 'linear',
                'input_size': None,  # Will be fixed
                'output_size': 10,
                'components': None  # Will be fixed
            }
        ],
        'total_components': None,  # Will be calculated
        'connections': [],
        'wavelength': 1550
    }
    
    print("\n1. Original circuit with issues:")
    print(f"   Name: {recoverable_circuit['name']}")
    print(f"   Layer 0 input_size: {recoverable_circuit['layers'][0]['input_size']}")
    print(f"   Layer 0 components: {recoverable_circuit['layers'][0]['components']}")
    
    try:
        # This will fail validation
        validator = CircuitValidator()
        result = validator.validate_circuit(recoverable_circuit)
        if not result.is_valid:
            raise ValidationException("Circuit has validation errors")
            
    except Exception as error:
        error_info = error_handler.handle_error(error)
        
        # Attempt recovery
        recovered_data = error_handler.attempt_recovery(error_info, recoverable_circuit)
        
        if recovered_data:
            print("\n2. After recovery:")
            print(f"   Name: {recovered_data.get('name', 'N/A')}")
            print(f"   Layer 0 input_size: {recovered_data['layers'][0].get('input_size', 'N/A')}")
            print(f"   Layer 0 components: {recovered_data['layers'][0].get('components', 'N/A')}")
            
            # Try validation again
            recovery_result = validator.validate_circuit(recovered_data)
            print(f"   Validation after recovery: {'PASS' if recovery_result.is_valid else 'FAIL'}")
        else:
            print("   Recovery failed")


def main():
    """Main demonstration of robust functionality."""
    print("=== PhotonicFoundry Robust Usage Demo ===")
    
    # Demonstrate all robust features
    demonstrate_validation()
    demonstrate_error_handling()
    demonstrate_safe_operations()
    demonstrate_security_validation()
    demonstrate_error_recovery()
    
    # Final error summary
    print("\n=== Error Summary ===")
    error_handler = ErrorHandler()
    
    # Generate some errors for demonstration
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
    print("- Input validation with sanitization")
    print("- Automatic error recovery")
    print("- Security validation")
    print("- Safe operation contexts")
    print("- Robust function wrappers")


if __name__ == "__main__":
    main()