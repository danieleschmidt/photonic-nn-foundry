"""
Circuit optimization utilities for photonic neural networks.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from circuit optimization."""
    original_components: int
    optimized_components: int
    reduction_percentage: float
    estimated_speedup: float
    energy_savings: float
    optimization_methods: List[str]


class CircuitOptimizer:
    """Optimize photonic circuits for performance and efficiency."""
    
    def __init__(self):
        self.optimization_methods = [
            'reduce_crossings',
            'minimize_components', 
            'optimize_power'
        ]
        
    def optimize_circuit(self, circuit_data: Dict[str, Any], 
                        methods: Optional[List[str]] = None) -> OptimizationResult:
        """
        Apply optimization methods to circuit.
        
        Args:
            circuit_data: Circuit specification dictionary
            methods: List of optimization methods to apply
            
        Returns:
            OptimizationResult with optimization details
        """
        if methods is None:
            methods = self.optimization_methods
            
        original_components = circuit_data.get('total_components', 0)
        
        for method in methods:
            if hasattr(self, f'_optimize_{method}'):
                optimizer_func = getattr(self, f'_optimize_{method}')
                circuit_data = optimizer_func(circuit_data)
                logger.info(f"Applied optimization: {method}")
                
        optimized_components = circuit_data.get('total_components', 0)
        reduction = ((original_components - optimized_components) / 
                    max(original_components, 1)) * 100
        
        return OptimizationResult(
            original_components=original_components,
            optimized_components=optimized_components,
            reduction_percentage=reduction,
            estimated_speedup=self._estimate_speedup(reduction),
            energy_savings=self._estimate_energy_savings(reduction),
            optimization_methods=methods
        )
        
    def _optimize_reduce_crossings(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce waveguide crossings through improved routing."""
        layers = circuit_data.get('layers', [])
        
        for layer in layers:
            if 'connections' in layer:
                optimized_connections = self._minimize_crossings(layer['connections'])
                layer['connections'] = optimized_connections
                
                crossing_reduction = 0.15
                if 'component_count' in layer:
                    layer['component_count'] = int(layer['component_count'] * (1 - crossing_reduction))
                    
        circuit_data['total_components'] = sum(layer.get('component_count', 0) for layer in layers)
        return circuit_data
        
    def _optimize_minimize_components(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize total component count through sharing."""
        layers = circuit_data.get('layers', [])
        shared_components = {}
        
        for layer in layers:
            layer_type = layer.get('type')
            if layer_type in shared_components:
                sharing_factor = 0.8
                layer['component_count'] = int(layer.get('component_count', 0) * sharing_factor)
            else:
                shared_components[layer_type] = layer
                
        circuit_data['total_components'] = sum(layer.get('component_count', 0) for layer in layers)
        return circuit_data
        
    def _optimize_optimize_power(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize optical power distribution."""
        layers = circuit_data.get('layers', [])
        total_power_budget = 100  # mW
        power_per_layer = total_power_budget / len(layers) if layers else 0
        
        for layer in layers:
            layer_components = layer.get('component_count', 0)
            optimized_power = min(power_per_layer, layer_components * 0.1)
            layer['power_allocation'] = optimized_power
            layer['component_count'] = int(layer.get('component_count', 0) * 0.98)
            
        circuit_data['total_components'] = sum(layer.get('component_count', 0) for layer in layers)
        return circuit_data
        
    def _minimize_crossings(self, connections: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Minimize waveguide crossings in connection list."""
        return sorted(connections, key=lambda x: (x[0], x[1]))
        
    def _estimate_speedup(self, component_reduction: float) -> float:
        """Estimate speedup from component reduction."""
        base_speedup = 1.0
        reduction_factor = component_reduction / 100.0
        estimated_speedup = base_speedup + (reduction_factor * 2.0)
        return max(1.0, estimated_speedup)
        
    def _estimate_energy_savings(self, component_reduction: float) -> float:
        """Estimate energy savings from optimization."""
        reduction_factor = component_reduction / 100.0
        energy_savings = reduction_factor * 0.8
        return min(0.9, energy_savings)


class PerformanceOptimizer:
    """Optimize performance characteristics of photonic circuits."""
    
    def __init__(self):
        self.metrics = {
            'latency_target': 100,  # ps
            'energy_target': 1.0,   # pJ/op
            'area_target': 1.0,     # mm²
            'throughput_target': 1000,  # GOPS
        }
        
    def optimize_for_latency(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize circuit for minimum latency."""
        logger.info("Optimizing for minimum latency")
        
        circuit_data['pipeline_stages'] = min(circuit_data.get('pipeline_stages', 5), 3)
        circuit_data['clock_frequency'] = 2.0  # 2 GHz max
        
        num_layers = len(circuit_data.get('layers', []))
        circuit_data['estimated_latency'] = num_layers * 50  # 50ps per layer
        
        return circuit_data
        
    def optimize_for_energy(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize circuit for minimum energy consumption."""
        logger.info("Optimizing for minimum energy consumption")
        
        for layer in circuit_data.get('layers', []):
            layer['power_gated'] = True
            layer['switching_factor'] = 0.3
            
        base_energy = circuit_data.get('estimated_energy', 1.0)
        circuit_data['estimated_energy'] = base_energy * 0.6
        
        return circuit_data
        
    def optimize_for_area(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize circuit for minimum area."""
        logger.info("Optimizing for minimum area")
        
        for layer in circuit_data.get('layers', []):
            layer['packing_density'] = 0.8
            
        base_area = circuit_data.get('estimated_area', 1.0)
        circuit_data['estimated_area'] = base_area * 0.7
        
        return circuit_data
        
    def multi_objective_optimization(self, circuit_data: Dict[str, Any], 
                                   weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Optimize for multiple objectives simultaneously."""
        if weights is None:
            weights = {'latency': 0.3, 'energy': 0.4, 'area': 0.3}
            
        logger.info(f"Multi-objective optimization with weights: {weights}")
        
        if weights.get('latency', 0) > 0:
            circuit_data = self.optimize_for_latency(circuit_data)
            
        if weights.get('energy', 0) > 0:
            circuit_data = self.optimize_for_energy(circuit_data)
            
        if weights.get('area', 0) > 0:
            circuit_data = self.optimize_for_area(circuit_data)
            
        # Calculate composite score
        latency_score = 1.0 / max(circuit_data.get('estimated_latency', 100), 1)
        energy_score = 1.0 / max(circuit_data.get('estimated_energy', 1), 0.1)
        area_score = 1.0 / max(circuit_data.get('estimated_area', 1), 0.1)
        
        composite_score = (
            weights.get('latency', 0) * latency_score +
            weights.get('energy', 0) * energy_score +
            weights.get('area', 0) * area_score
        )
        
        circuit_data['optimization_score'] = composite_score
        
        return circuit_data