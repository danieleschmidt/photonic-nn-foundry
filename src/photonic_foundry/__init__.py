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
from .quantum_optimizer import QuantumOptimizationEngine, DistributedQuantumProcessor, OptimizationConfig, ScalingConfig

__all__ = [
    "PhotonicAccelerator",
    "PhotonicCircuit", 
    "MZILayer",
    "CircuitMetrics",
    "torch2verilog",
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
    "ScalingConfig",
]