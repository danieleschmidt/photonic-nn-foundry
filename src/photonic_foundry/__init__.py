"""
Photonic Neural Network Foundry

A comprehensive software stack for silicon-photonic AI accelerators with 
quantum-inspired task planning and optimization capabilities.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@photonic-foundry.com"

from .core import PhotonicAccelerator, PhotonicCircuit, MZILayer, CircuitMetrics
from .transpiler import torch2verilog
from .quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumState, ResourceConstraint
from .quantum_security import QuantumSecurityManager, QuantumSecurityToken, SecurityLevel, SecurityConstraint
from .quantum_resilience import QuantumResilienceManager, CircuitHealthMonitor, QuantumErrorCorrector
from .quantum_optimizer import QuantumOptimizationEngine, DistributedQuantumProcessor, OptimizationConfig
from .scaling import ScalingConfig, get_auto_scaler, start_scaling_services, stop_scaling_services

# Import resilience framework components
from .resilience_framework import (
    ResilienceFramework, ResilienceConfig,
    initialize_resilience_framework, get_resilience_framework,
    resilience_context, with_resilience
)

# Import key resilience components for direct use
from .error_handling import ErrorHandler, safe_operation
from .validation import CircuitValidator, create_comprehensive_validator
from .monitoring import get_system_status, start_monitoring, stop_monitoring
from .security import SecurityScanner, scan_security
from .circuit_breaker import circuit_breaker, get_circuit_breaker
from .logging_config import setup_logging, get_logger

# Import advanced scaling and performance optimization components
from .advanced_scaling import (
    AdvancedScalingConfig, EnterpriseAutoScaler, LoadBalancingAlgorithm,
    get_enterprise_scaler, start_enterprise_scaling, stop_enterprise_scaling
)
from .concurrent_processing import (
    DistributedTaskExecutor, TaskPriority, get_distributed_executor,
    start_concurrent_processing, stop_concurrent_processing
)
# Optional advanced caching (requires zstandard)
try:
    from .intelligent_caching import create_intelligent_cache, CachePolicy
    ADVANCED_CACHING_AVAILABLE = True
except ImportError:
    ADVANCED_CACHING_AVAILABLE = False
    def create_intelligent_cache(*args, **kwargs):
        raise ImportError("Advanced caching requires 'zstandard' package. Install with: pip install zstandard")
    CachePolicy = None
from .performance_analytics import (
    get_performance_analyzer, start_performance_monitoring, 
    stop_performance_monitoring, measure_time, profile_performance
)
# Optional enterprise config (requires jsonschema) 
try:
    from .enterprise_config import (
        get_config_manager, Environment, set_config, get_config,
        load_config_from_file, export_config_report
    )
    ENTERPRISE_CONFIG_AVAILABLE = True
except ImportError:
    ENTERPRISE_CONFIG_AVAILABLE = False
    def get_config_manager(*args, **kwargs):
        raise ImportError("Enterprise config requires 'jsonschema' package. Install with: pip install jsonschema")
    Environment = None
    set_config = get_config = load_config_from_file = export_config_report = get_config_manager

__all__ = [
    # Core photonic components
    "PhotonicAccelerator",
    "PhotonicCircuit", 
    "MZILayer",
    "CircuitMetrics",
    "torch2verilog",
    
    # Quantum components
    "QuantumTaskPlanner",
    "QuantumTask", 
    "QuantumState",
    "ResourceConstraint",
    "QuantumSecurityManager",
    "QuantumSecurityToken",
    "SecurityLevel",
    "SecurityConstraint", 
    "QuantumResilienceManager",
    "CircuitHealthMonitor",
    "QuantumErrorCorrector",
    "QuantumOptimizationEngine",
    "DistributedQuantumProcessor", 
    "OptimizationConfig",
    
    # Resilience framework
    "ResilienceFramework",
    "ResilienceConfig", 
    "initialize_resilience_framework",
    "get_resilience_framework",
    "resilience_context",
    "with_resilience",
    
    # Error handling & validation
    "ErrorHandler",
    "safe_operation",
    "CircuitValidator",
    "create_comprehensive_validator",
    
    # Monitoring & observability
    "get_system_status",
    "start_monitoring", 
    "stop_monitoring",
    
    # Security
    "SecurityScanner",
    "scan_security",
    
    # Circuit breakers & fault tolerance
    "circuit_breaker",
    "get_circuit_breaker",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Advanced scaling and performance optimization
    "AdvancedScalingConfig",
    "EnterpriseAutoScaler",
    "LoadBalancingAlgorithm",
    "get_enterprise_scaler",
    "start_enterprise_scaling",
    "stop_enterprise_scaling",
    "DistributedTaskExecutor",
    "TaskPriority",
    "get_distributed_executor",
    "start_concurrent_processing",
    "stop_concurrent_processing",
    "create_intelligent_cache",
    "CachePolicy",
    "get_performance_analyzer",
    "start_performance_monitoring",
    "stop_performance_monitoring",
    "measure_time",
    "profile_performance",
    "get_config_manager",
    "Environment",
    "set_config",
    "get_config",
    "load_config_from_file",
    "export_config_report",
    "ScalingConfig",
    "get_auto_scaler",
    "start_scaling_services",
    "stop_scaling_services",
]