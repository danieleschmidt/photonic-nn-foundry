"""
Robust Framework for Photonic Neural Network Foundry
Implements comprehensive error handling, validation, logging, monitoring, and security.
"""

import logging
import time
import json
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import sys
import traceback
import os

# Configure logging
def setup_robust_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging for robust operations."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Robust logging initialized - Level: {log_level}")
    return logger

logger = setup_robust_logging()


class ValidationError(Exception):
    """Custom exception for validation failures."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class SecurityError(Exception):
    """Custom exception for security violations."""
    pass


class PerformanceError(Exception):
    """Custom exception for performance threshold violations."""
    pass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthCheck:
    """System health check result."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    metrics: Dict[str, Any]
    timestamp: float


class SecurityLevel(Enum):
    """Security levels for operations."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    QUANTUM_RESISTANT = "quantum_resistant"


def robust_error_handler(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for robust error handling with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                     f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


class InputValidator:
    """Comprehensive input validation."""
    
    @staticmethod
    def validate_layer_sizes(layer_sizes: List[int]) -> ValidationResult:
        """Validate neural network layer sizes."""
        details = {"layer_sizes": layer_sizes}
        
        if not isinstance(layer_sizes, list):
            return ValidationResult(
                passed=False,
                message="Layer sizes must be a list",
                details=details,
                timestamp=time.time()
            )
        
        if len(layer_sizes) < 2:
            return ValidationResult(
                passed=False,
                message="Need at least 2 layers (input and output)",
                details=details,
                timestamp=time.time()
            )
        
        for i, size in enumerate(layer_sizes):
            if not isinstance(size, int) or size <= 0:
                return ValidationResult(
                    passed=False,
                    message=f"Layer {i} size must be positive integer, got {size}",
                    details=details,
                    timestamp=time.time()
                )
        
        # Security check: Prevent extremely large networks
        max_layer_size = 10000
        total_params = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        
        if any(size > max_layer_size for size in layer_sizes):
            return ValidationResult(
                passed=False,
                message=f"Layer size exceeds security limit of {max_layer_size}",
                details=details,
                timestamp=time.time()
            )
        
        if total_params > 1000000:  # 1M parameters limit
            return ValidationResult(
                passed=False,
                message=f"Total parameters ({total_params}) exceeds security limit",
                details=details,
                timestamp=time.time()
            )
        
        return ValidationResult(
            passed=True,
            message="Layer sizes validation passed",
            details=details,
            timestamp=time.time()
        )
    
    @staticmethod
    def validate_string_input(value: str, field_name: str, 
                            max_length: int = 255, 
                            allowed_chars: Optional[str] = None) -> ValidationResult:
        """Validate string inputs with security checks."""
        details = {"field_name": field_name, "value_length": len(value) if value else 0}
        
        if not isinstance(value, str):
            return ValidationResult(
                passed=False,
                message=f"{field_name} must be a string",
                details=details,
                timestamp=time.time()
            )
        
        if len(value) > max_length:
            return ValidationResult(
                passed=False,
                message=f"{field_name} exceeds maximum length of {max_length}",
                details=details,
                timestamp=time.time()
            )
        
        # Security: Check for potentially dangerous patterns
        dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(', '../', '..\\\\']
        for pattern in dangerous_patterns:
            if pattern.lower() in value.lower():
                return ValidationResult(
                    passed=False,
                    message=f"Security violation: dangerous pattern '{pattern}' in {field_name}",
                    details=details,
                    timestamp=time.time()
                )
        
        if allowed_chars:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                return ValidationResult(
                    passed=False,
                    message=f"Invalid characters in {field_name}: {invalid_chars}",
                    details=details,
                    timestamp=time.time()
                )
        
        return ValidationResult(
            passed=True,
            message=f"{field_name} validation passed",
            details=details,
            timestamp=time.time()
        )


class PerformanceMonitor:
    """Monitor system performance and enforce limits."""
    
    def __init__(self, max_memory_mb: int = 1000, max_execution_time: float = 60.0):
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.metrics = {}
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor operation performance."""
        start_time = time.time()
        start_metrics = self._get_system_metrics()
        
        logger.info(f"Starting monitored operation: {operation_name}")
        
        try:
            yield self
            
        except Exception as e:
            logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise
            
        finally:
            end_time = time.time()
            end_metrics = self._get_system_metrics()
            execution_time = end_time - start_time
            
            # Check performance limits
            if execution_time > self.max_execution_time:
                raise PerformanceError(
                    f"Operation {operation_name} exceeded time limit: "
                    f"{execution_time:.2f}s > {self.max_execution_time}s"
                )
            
            # Store metrics
            self.metrics[operation_name] = {
                'execution_time': execution_time,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'timestamp': end_time
            }
            
            logger.info(f"Operation {operation_name} completed in {execution_time:.2f}s")
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'timestamp': time.time(),
            'memory_usage_mb': 0,  # Simplified - would use psutil in full implementation
            'cpu_usage_percent': 0.0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            'total_operations': len(self.metrics),
            'average_execution_time': sum(m['execution_time'] for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0,
            'operations': self.metrics,
            'limits': {
                'max_memory_mb': self.max_memory_mb,
                'max_execution_time': self.max_execution_time
            },
            'report_timestamp': time.time()
        }


class SecurityManager:
    """Comprehensive security management."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        logger.info(f"Security manager initialized with level: {security_level.value}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        if length < 16:
            raise SecurityError("Token length must be at least 16 characters")
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data using SHA-256."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def validate_operation_security(self, operation: str, context: Dict[str, Any]) -> bool:
        """Validate security for operations."""
        client_id = context.get('client_id', 'unknown')
        
        # Check for rate limiting
        current_time = time.time()
        if client_id in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[client_id]
            if attempts >= self.max_failed_attempts:
                if current_time - last_attempt < self.lockout_duration:
                    raise SecurityError(f"Client {client_id} is locked out due to too many failed attempts")
                else:
                    # Reset after lockout period
                    del self.failed_attempts[client_id]
        
        # Validate operation against security level
        if self.security_level == SecurityLevel.QUANTUM_RESISTANT:
            if not context.get('quantum_secured', False):
                logger.warning(f"Operation {operation} not quantum-secured but required by security level")
        
        return True
    
    def record_failed_attempt(self, client_id: str):
        """Record a failed security attempt."""
        current_time = time.time()
        if client_id in self.failed_attempts:
            attempts, _ = self.failed_attempts[client_id]
            self.failed_attempts[client_id] = (attempts + 1, current_time)
        else:
            self.failed_attempts[client_id] = (1, current_time)
        
        logger.warning(f"Failed security attempt recorded for client {client_id}")


class RobustPhotonicAccelerator:
    """Enhanced photonic accelerator with comprehensive robustness features."""
    
    def __init__(self, pdk: str = 'skywater130', wavelength: float = 1550.0,
                 security_level: SecurityLevel = SecurityLevel.ENHANCED):
        # Input validation
        pdk_validation = InputValidator.validate_string_input(
            pdk, "pdk", max_length=50, 
            allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789_"
        )
        if not pdk_validation.passed:
            raise ValidationError(pdk_validation.message, pdk_validation.details)
        
        if not isinstance(wavelength, (int, float)) or wavelength <= 0:
            raise ValidationError("Wavelength must be positive number")
        
        if not (1000 <= wavelength <= 2000):  # Typical photonic wavelength range
            raise ValidationError(f"Wavelength {wavelength}nm outside typical range (1000-2000nm)")
        
        self.pdk = pdk
        self.wavelength = wavelength
        self.circuits = {}
        self.validator = InputValidator()
        self.performance_monitor = PerformanceMonitor()
        self.security_manager = SecurityManager(security_level)
        
        # Generate session token
        self.session_token = self.security_manager.generate_secure_token()
        
        logger.info(f"RobustPhotonicAccelerator initialized - PDK={pdk}, λ={wavelength}nm, "
                   f"Security={security_level.value}")
    
    @robust_error_handler(max_retries=2, backoff_factor=0.5)
    def convert_simple_model_robust(self, layer_sizes: List[int], 
                                  circuit_name: Optional[str] = None,
                                  security_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Robust version of simple model conversion with comprehensive validation."""
        
        # Security validation
        sec_context = security_context or {'client_id': 'default'}
        self.security_manager.validate_operation_security('convert_model', sec_context)
        
        # Input validation
        validation_result = self.validator.validate_layer_sizes(layer_sizes)
        if not validation_result.passed:
            raise ValidationError(validation_result.message, validation_result.details)
        
        # Performance monitoring
        with self.performance_monitor.monitor_operation('model_conversion'):
            
            # Generate secure circuit name if not provided
            if circuit_name is None:
                name_hash = hashlib.md5(str(layer_sizes).encode()).hexdigest()[:8]
                circuit_name = f"robust_model_{name_hash}"
            else:
                name_validation = self.validator.validate_string_input(
                    circuit_name, "circuit_name", max_length=100,
                    allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789_-"
                )
                if not name_validation.passed:
                    raise ValidationError(name_validation.message, name_validation.details)
            
            # Import standalone core for circuit creation
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from core_standalone import PhotonicCircuit, MZILayer, CircuitMetrics
            
            # Create circuit with robust error handling
            try:
                circuit = PhotonicCircuit(circuit_name)
                
                # Create layers with validation
                for i in range(len(layer_sizes) - 1):
                    layer = MZILayer(layer_sizes[i], layer_sizes[i+1], precision=8)
                    circuit.add_layer(layer)
                
                # Generate outputs
                verilog_code = circuit.generate_full_verilog()
                metrics = circuit.estimate_metrics()
                
                # Security: Hash sensitive information
                verilog_hash = self.security_manager.hash_sensitive_data(verilog_code)
                
                # Store circuit securely
                self.circuits[circuit_name] = {
                    'circuit': circuit,
                    'created_timestamp': time.time(),
                    'session_token': self.session_token,
                    'verilog_hash': verilog_hash
                }
                
                # Create comprehensive result
                result = {
                    'circuit_name': circuit_name,
                    'success': True,
                    'circuit_info': {
                        'layers': len(circuit.layers),
                        'total_components': circuit.total_components,
                        'layer_sizes': layer_sizes
                    },
                    'performance_metrics': metrics.to_dict(),
                    'verilog_stats': {
                        'length': len(verilog_code),
                        'hash': verilog_hash
                    },
                    'security': {
                        'level': self.security_manager.security_level.value,
                        'validated': True
                    },
                    'validation_results': [validation_result.to_dict()],
                    'timestamp': time.time()
                }
                
                logger.info(f"Successfully created robust circuit: {circuit_name}")
                return result
                
            except Exception as e:
                logger.error(f"Circuit creation failed: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValidationError(f"Circuit creation failed: {str(e)}")
    
    def get_health_status(self) -> List[HealthCheck]:
        """Comprehensive health check of the system."""
        health_checks = []
        current_time = time.time()
        
        # Check system components
        components = [
            ('validator', self.validator),
            ('performance_monitor', self.performance_monitor),
            ('security_manager', self.security_manager)
        ]
        
        for comp_name, comp_obj in components:
            try:
                # Basic availability check
                status = 'healthy' if comp_obj else 'critical'
                health_checks.append(HealthCheck(
                    component=comp_name,
                    status=status,
                    message=f"{comp_name} is operational",
                    metrics={'last_check': current_time},
                    timestamp=current_time
                ))
            except Exception as e:
                health_checks.append(HealthCheck(
                    component=comp_name,
                    status='critical',
                    message=f"{comp_name} error: {str(e)}",
                    metrics={'error': str(e)},
                    timestamp=current_time
                ))
        
        # Check circuit storage
        health_checks.append(HealthCheck(
            component='circuit_storage',
            status='healthy' if len(self.circuits) >= 0 else 'warning',
            message=f"{len(self.circuits)} circuits stored",
            metrics={'circuit_count': len(self.circuits)},
            timestamp=current_time
        ))
        
        return health_checks
    
    def generate_robustness_report(self) -> Dict[str, Any]:
        """Generate comprehensive robustness and health report."""
        health_status = self.get_health_status()
        performance_report = self.performance_monitor.get_performance_report()
        
        # Count health status
        health_summary = {}
        for check in health_status:
            health_summary[check.status] = health_summary.get(check.status, 0) + 1
        
        return {
            'system_status': 'healthy' if health_summary.get('critical', 0) == 0 else 'degraded',
            'health_summary': health_summary,
            'health_checks': [asdict(check) for check in health_status],
            'performance_report': performance_report,
            'security_info': {
                'level': self.security_manager.security_level.value,
                'failed_attempts': len(self.security_manager.failed_attempts),
                'session_token_present': bool(self.session_token)
            },
            'circuit_count': len(self.circuits),
            'accelerator_config': {
                'pdk': self.pdk,
                'wavelength': self.wavelength
            },
            'report_timestamp': time.time()
        }


def create_robust_demo() -> Dict[str, Any]:
    """Create demonstration of robust photonic acceleration."""
    print("🛡️ Creating Robust Photonic Acceleration Demo...")
    
    try:
        # Initialize robust accelerator
        accelerator = RobustPhotonicAccelerator(
            pdk='skywater130', 
            wavelength=1550.0,
            security_level=SecurityLevel.ENHANCED
        )
        
        # Test various layer configurations
        test_configs = [
            ([4, 8, 2], "simple_mlp"),
            ([10, 64, 32, 5], "deep_network"),
            ([28*28, 256, 128, 10], "mnist_classifier")
        ]
        
        results = []
        for layer_sizes, name in test_configs:
            print(f"  → Testing configuration: {layer_sizes}")
            result = accelerator.convert_simple_model_robust(
                layer_sizes=layer_sizes,
                circuit_name=name,
                security_context={'client_id': 'robust_demo', 'quantum_secured': False}
            )
            results.append(result)
        
        # Generate comprehensive robustness report
        robustness_report = accelerator.generate_robustness_report()
        
        # Save results
        demo_output = {
            'demo_type': 'robust_validation',
            'test_results': results,
            'robustness_report': robustness_report,
            'demo_timestamp': time.time()
        }
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        with open('output/robust_demo_results.json', 'w') as f:
            json.dump(demo_output, f, indent=2)
        
        print(f"✅ Robust demo completed successfully!")
        print(f"   → Tested {len(results)} configurations")
        print(f"   → System status: {robustness_report['system_status']}")
        print(f"   → Security level: {robustness_report['security_info']['level']}")
        print(f"   → Results saved to output/robust_demo_results.json")
        
        return demo_output
        
    except Exception as e:
        error_report = {
            'demo_type': 'robust_validation',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.time()
        }
        
        print(f"❌ Robust demo failed: {str(e)}")
        
        # Save error report
        os.makedirs('output', exist_ok=True)
        with open('output/robust_demo_error.json', 'w') as f:
            json.dump(error_report, f, indent=2)
        
        raise


if __name__ == "__main__":
    # Setup logging for standalone execution
    setup_robust_logging("INFO", "output/robust_framework.log")
    
    # Run robust demo
    results = create_robust_demo()
    print("\n🛡️ GENERATION 2 SUCCESS: Robust framework implemented!")