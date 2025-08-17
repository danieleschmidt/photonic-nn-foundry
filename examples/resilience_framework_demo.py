#!/usr/bin/env python3
"""
Demonstration of the Photonic Foundry Resilience Framework

This example shows how to use the comprehensive resilience framework
for error handling, validation, logging, monitoring, and security.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

# Import the resilience framework
from photonic_foundry import (
    ResilienceFramework, ResilienceConfig,
    initialize_resilience_framework, resilience_context,
    with_resilience, PhotonicAccelerator
)


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.layers(x)


def demonstrate_basic_usage():
    """Demonstrate basic resilience framework usage."""
    print("=== Basic Resilience Framework Usage ===")
    
    # Create configuration
    config = ResilienceConfig(
        log_level="INFO",
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        enable_console_logging=True,
        enable_performance_monitoring=True,
        enable_health_checks=True,
        security_level="MEDIUM"
    )
    
    # Initialize framework
    framework = initialize_resilience_framework(config)
    
    try:
        # Get system status
        status = framework.get_system_status()
        print(f"System Status: {status['resilience_framework']['initialized']}")
        print(f"Components: {len(status['resilience_framework']['components'])}")
        
        # Test validation and processing
        test_data = {
            'name': 'test_circuit',
            'wavelength': 1550,
            'precision': 8,
            'layers': [
                {'type': 'linear', 'input_size': 4, 'output_size': 8},
                {'type': 'activation', 'input_size': 8, 'output_size': 8},
                {'type': 'linear', 'input_size': 8, 'output_size': 2}
            ],
            'total_components': 3
        }
        
        result = framework.validate_and_process(test_data, "circuit")
        print(f"Validation Result: {result['success']}")
        print(f"Processing Time: {result['metrics']['processing_time_seconds']:.4f}s")
        
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")
            
    finally:
        # Shutdown framework
        framework.shutdown()


def demonstrate_context_manager():
    """Demonstrate using the framework with context manager."""
    print("\n=== Context Manager Usage ===")
    
    config = ResilienceConfig(
        log_level="DEBUG",
        enable_circuit_breakers=True,
        enable_strict_validation=True
    )
    
    with resilience_context(config) as framework:
        print(f"Framework initialized: {framework._initialized}")
        
        # Simulate processing some data
        model = SimpleModel()
        
        # This would normally go through validation, security checks, etc.
        test_input = torch.randn(1, 4)
        output = model(test_input)
        
        print(f"Model output shape: {output.shape}")
        print(f"System uptime: {framework.get_system_status()['resilience_framework']['uptime_seconds']:.2f}s")


@with_resilience(context="model_processing")
def process_model_with_resilience(model_data):
    """Demonstrate function decorator with resilience protection."""
    print("\n=== Decorator Usage ===")
    
    # This function is automatically protected with:
    # - Input validation
    # - Security checks  
    # - Error handling with recovery
    # - Performance monitoring
    # - Circuit breaker protection
    
    print(f"Processing model: {model_data.get('name', 'unnamed')}")
    
    # Simulate some processing
    import time
    time.sleep(0.1)  # Simulate work
    
    # Return processed result
    return {
        'status': 'processed',
        'name': model_data.get('name'),
        'layers_processed': len(model_data.get('layers', [])),
        'processing_time': 0.1
    }


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n=== Error Handling Demo ===")
    
    config = ResilienceConfig(
        enable_error_recovery=True,
        max_retry_attempts=3
    )
    
    with resilience_context(config) as framework:
        
        # Test with invalid data that should trigger recovery
        invalid_data = {
            'name': None,  # This will trigger validation error
            'wavelength': -1,  # Invalid wavelength
            'layers': []  # Empty layers
        }
        
        try:
            result = framework.validate_and_process(invalid_data, "circuit")
            print(f"Error handling result: {result['success']}")
            print(f"Errors: {result['errors']}")
            print(f"Warnings: {result['warnings']}")
            
            if result['data']:
                print(f"Recovered data: {result['data'].get('name', 'N/A')}")
                
        except Exception as e:
            print(f"Unrecoverable error: {e}")


def demonstrate_security_features():
    """Demonstrate security features."""
    print("\n=== Security Features Demo ===")
    
    config = ResilienceConfig(
        security_level="HIGH",
        enable_malware_scanning=True,
        rate_limit_requests=5,  # Low limit for demo
        rate_limit_window=10
    )
    
    with resilience_context(config) as framework:
        
        # Get security monitor
        security_monitor = framework.get_component('security_monitor')
        
        if security_monitor:
            # Simulate some security events
            security_monitor.record_security_event(
                'suspicious_activity',
                'medium', 
                'Unusual model structure detected',
                source_ip='192.168.1.100'
            )
            
            # Get security summary
            summary = security_monitor.get_security_summary()
            print(f"Security events: {summary['total_events']}")
            print(f"Threat types: {list(summary['threat_counts'].keys())}")


def demonstrate_monitoring():
    """Demonstrate monitoring and health checks."""
    print("\n=== Monitoring Demo ===")
    
    config = ResilienceConfig(
        enable_performance_monitoring=True,
        enable_health_checks=True,
        metrics_collection_interval=5  # Quick collection for demo
    )
    
    with resilience_context(config) as framework:
        
        # Get performance monitor
        perf_monitor = framework.get_component('performance_monitor')
        
        if perf_monitor:
            # Measure some operations
            with perf_monitor.measure_time("demo_operation"):
                import time
                time.sleep(0.05)  # Simulate work
                
            # Log some metrics
            metrics_logger = framework.get_component('metrics_logger')
            if metrics_logger:
                metrics_logger.log_counter('demo_counter', 1)
                metrics_logger.log_gauge('demo_gauge', 42.0)
        
        # Get health check manager
        health_manager = framework.get_component('health_manager')
        
        if health_manager:
            health_summary = health_manager.get_health_summary()
            print(f"Overall health: {health_summary['overall_health']}")
            print(f"Health checks: {health_summary['total_checks']}")
            print(f"Success rate: {health_summary['success_rate']:.1%}")


def demonstrate_circuit_breakers():
    """Demonstrate circuit breaker functionality."""
    print("\n=== Circuit Breakers Demo ===")
    
    config = ResilienceConfig(
        enable_circuit_breakers=True,
        default_failure_threshold=2,  # Low threshold for demo
        default_recovery_timeout=5.0   # Quick recovery for demo
    )
    
    with resilience_context(config) as framework:
        
        # Get circuit breaker middleware
        cb_middleware = framework.get_component('circuit_breaker_middleware')
        
        if cb_middleware:
            
            # Create a test function that sometimes fails
            failure_count = 0
            
            @cb_middleware.protect_endpoint("demo_endpoint")
            def unreliable_operation():
                nonlocal failure_count
                failure_count += 1
                
                # Fail the first 3 attempts
                if failure_count <= 3:
                    raise Exception(f"Simulated failure #{failure_count}")
                
                return f"Success after {failure_count} attempts"
            
            # Try calling the protected function multiple times
            for i in range(6):
                try:
                    result = unreliable_operation()
                    print(f"Attempt {i+1}: {result}")
                except Exception as e:
                    print(f"Attempt {i+1}: Failed - {e}")
                    
                # Small delay between attempts
                import time
                time.sleep(0.1)


def main():
    """Main demonstration function."""
    print("Photonic Foundry Resilience Framework Demonstration")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage()
        demonstrate_context_manager()
        
        # Test decorator (need to initialize framework first for auto_init)
        config = ResilienceConfig(log_level="INFO")
        initialize_resilience_framework(config)
        
        try:
            model_data = {
                'name': 'demo_model',
                'layers': [
                    {'type': 'linear', 'input_size': 4, 'output_size': 8},
                    {'type': 'linear', 'input_size': 8, 'output_size': 2}
                ]
            }
            
            result = process_model_with_resilience(model_data)
            print(f"Decorated function result: {result['status']}")
            
        except Exception as e:
            print(f"Decorator demo error: {e}")
        
        # Continue with other demonstrations
        demonstrate_error_handling()
        demonstrate_security_features()
        demonstrate_monitoring()
        demonstrate_circuit_breakers()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()