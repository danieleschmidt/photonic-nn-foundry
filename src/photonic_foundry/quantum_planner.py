"""
Quantum-inspired task planning for photonic neural networks.

This module implements quantum-inspired algorithms for optimizing photonic circuit 
compilation, task scheduling, and resource allocation using quantum annealing 
principles and superposition-based search strategies.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math
import random
import time
import logging
from .core import PhotonicCircuit, PhotonicAccelerator, CircuitMetrics

logger = logging.getLogger(__name__)


class QuantumState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"
    COLLAPSED = "collapsed"
    ENTANGLED = "entangled"
    OPTIMIZED = "optimized"


@dataclass
class QuantumTask:
    """Quantum-inspired task representation."""
    id: str
    priority: float = 1.0
    complexity: int = 1
    dependencies: List[str] = field(default_factory=list)
    resources_required: Dict[str, float] = field(default_factory=dict)
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    probability_amplitude: complex = 1.0 + 0j
    entangled_tasks: Set[str] = field(default_factory=set)
    estimated_energy: float = 0.0
    estimated_latency: float = 0.0
    
    def collapse_state(self, execution_probability: float):
        """Collapse quantum superposition to classical state."""
        if random.random() < execution_probability:
            self.quantum_state = QuantumState.COLLAPSED
            self.probability_amplitude = 1.0 + 0j
        else:
            self.quantum_state = QuantumState.SUPERPOSITION
            self.probability_amplitude = 0.0 + 0j


@dataclass 
class ResourceConstraint:
    """Resource constraints for quantum planning."""
    max_energy: float = float('inf')
    max_latency: float = float('inf')
    max_area: float = float('inf')
    max_concurrent_tasks: int = 16
    thermal_limit: float = 85.0  # Celsius


class QuantumTaskPlanner:
    """Quantum-inspired task planner for photonic circuits."""
    
    def __init__(self, accelerator: PhotonicAccelerator, 
                 constraints: ResourceConstraint = None):
        """
        Initialize quantum task planner.
        
        Args:
            accelerator: PhotonicAccelerator instance
            constraints: Resource constraints for planning
        """
        self.accelerator = accelerator
        self.constraints = constraints or ResourceConstraint()
        self.task_registry = {}
        self.quantum_superposition = []
        self.entanglement_matrix = np.eye(0)
        self.optimization_history = []
        
        logger.info("Initialized QuantumTaskPlanner")
    
    def register_task(self, task: QuantumTask):
        """Register a quantum task in the planner."""
        self.task_registry[task.id] = task
        logger.debug(f"Registered quantum task: {task.id}")
    
    def create_circuit_compilation_plan(self, circuit: PhotonicCircuit) -> List[QuantumTask]:
        """
        Create quantum-inspired compilation plan for photonic circuit.
        
        Args:
            circuit: PhotonicCircuit to plan compilation for
            
        Returns:
            List of QuantumTask objects for compilation
        """
        tasks = []
        
        # Task 1: Circuit Analysis (quantum superposition of optimization paths)
        analysis_task = QuantumTask(
            id="circuit_analysis",
            priority=1.0,
            complexity=3,
            resources_required={"compute": 0.2, "memory": 0.1},
            estimated_energy=5.0,
            estimated_latency=10.0
        )
        tasks.append(analysis_task)
        
        # Task 2: Layer-by-layer Verilog generation (parallelizable with quantum interference)
        for i, layer in enumerate(circuit.layers):
            layer_task = QuantumTask(
                id=f"layer_verilog_{i}",
                priority=0.8,
                complexity=2,
                dependencies=["circuit_analysis"],
                resources_required={"compute": 0.3, "memory": 0.15},
                estimated_energy=layer.output_size * 0.1,
                estimated_latency=layer.output_size * 0.2
            )
            tasks.append(layer_task)
        
        # Task 3: Optimization using quantum annealing principles
        optimization_task = QuantumTask(
            id="quantum_optimization",
            priority=0.9,
            complexity=5,
            dependencies=[f"layer_verilog_{i}" for i in range(len(circuit.layers))],
            resources_required={"compute": 0.8, "memory": 0.4},
            estimated_energy=50.0,
            estimated_latency=100.0
        )
        tasks.append(optimization_task)
        
        # Task 4: Final compilation and metric calculation
        final_task = QuantumTask(
            id="final_compilation",
            priority=1.0,
            complexity=2,
            dependencies=["quantum_optimization"],
            resources_required={"compute": 0.4, "memory": 0.2},
            estimated_energy=20.0,
            estimated_latency=30.0
        )
        tasks.append(final_task)
        
        # Create quantum entanglement between related tasks
        self._create_task_entanglement(tasks)
        
        return tasks
    
    def _create_task_entanglement(self, tasks: List[QuantumTask]):
        """Create quantum entanglement between related tasks."""
        for i, task_a in enumerate(tasks):
            for j, task_b in enumerate(tasks):
                if i != j and self._are_tasks_related(task_a, task_b):
                    task_a.entangled_tasks.add(task_b.id)
                    task_b.entangled_tasks.add(task_a.id)
                    task_a.quantum_state = QuantumState.ENTANGLED
                    task_b.quantum_state = QuantumState.ENTANGLED
    
    def _are_tasks_related(self, task_a: QuantumTask, task_b: QuantumTask) -> bool:
        """Check if two tasks are quantum-related (share resources or dependencies)."""
        # Tasks are related if they share resource requirements
        shared_resources = set(task_a.resources_required.keys()) & set(task_b.resources_required.keys())
        if shared_resources:
            return True
            
        # Tasks are related if one depends on the other
        if task_b.id in task_a.dependencies or task_a.id in task_b.dependencies:
            return True
            
        return False
    
    def quantum_annealing_optimization(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """
        Apply quantum annealing to optimize task execution order.
        
        Args:
            tasks: List of QuantumTask objects to optimize
            
        Returns:
            Optimized list of tasks
        """
        logger.info("Starting quantum annealing optimization")
        
        # Initialize quantum state space
        n_tasks = len(tasks)
        temperature = 1000.0  # Initial temperature
        cooling_rate = 0.95
        min_temperature = 0.1
        
        # Current solution (task execution order)
        current_solution = list(range(n_tasks))
        current_energy = self._calculate_solution_energy(current_solution, tasks)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        iteration = 0
        max_iterations = 1000
        
        while temperature > min_temperature and iteration < max_iterations:
            # Generate neighbor solution using quantum tunnel effect
            neighbor_solution = self._quantum_neighbor(current_solution)
            neighbor_energy = self._calculate_solution_energy(neighbor_solution, tasks)
            
            # Calculate acceptance probability using Boltzmann distribution
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
                    logger.debug(f"New best energy: {best_energy:.2f} at iteration {iteration}")
            
            temperature *= cooling_rate
            iteration += 1
        
        # Reorder tasks according to optimized solution
        optimized_tasks = [tasks[i] for i in best_solution]
        
        # Mark tasks as optimized
        for task in optimized_tasks:
            task.quantum_state = QuantumState.OPTIMIZED
        
        logger.info(f"Quantum annealing completed. Best energy: {best_energy:.2f}")
        self.optimization_history.append({
            'iteration': iteration,
            'final_temperature': temperature,
            'best_energy': best_energy,
            'optimization_time': time.time()
        })
        
        return optimized_tasks
    
    def _quantum_neighbor(self, solution: List[int]) -> List[int]:
        """Generate neighbor solution using quantum tunneling effect."""
        neighbor = solution.copy()
        
        if len(solution) < 2:
            return neighbor
        
        # Quantum tunneling: allow non-local moves
        if random.random() < 0.3:  # 30% chance of quantum tunnel
            # Long-range swap (tunneling)
            i, j = random.sample(range(len(solution)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Local move (classical)
            i = random.randint(0, len(solution) - 2)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        
        return neighbor
    
    def _calculate_solution_energy(self, solution: List[int], tasks: List[QuantumTask]) -> float:
        """
        Calculate energy of a task execution solution.
        
        Combines resource usage, dependency violations, and quantum interference.
        """
        total_energy = 0.0
        resource_usage = {"compute": 0.0, "memory": 0.0}
        completed_tasks = set()
        
        for i, task_idx in enumerate(solution):
            task = tasks[task_idx]
            
            # Dependency violation penalty
            for dep_id in task.dependencies:
                dep_idx = next((j for j, t in enumerate(tasks) if t.id == dep_id), None)
                if dep_idx is not None and dep_idx not in [solution[k] for k in range(i)]:
                    total_energy += 1000.0  # Heavy penalty for dependency violations
            
            # Resource constraint penalties
            for resource, required in task.resources_required.items():
                resource_usage[resource] += required
                if resource == "compute" and resource_usage[resource] > 1.0:
                    total_energy += 100.0 * (resource_usage[resource] - 1.0) ** 2
                elif resource == "memory" and resource_usage[resource] > 1.0:
                    total_energy += 50.0 * (resource_usage[resource] - 1.0) ** 2
            
            # Quantum interference effects for entangled tasks
            for entangled_task_id in task.entangled_tasks:
                entangled_idx = next((j for j, t in enumerate(tasks) if t.id == entangled_task_id), None)
                if entangled_idx is not None:
                    position_diff = abs(solution.index(task_idx) - solution.index(entangled_idx))
                    # Reward for keeping entangled tasks close
                    total_energy += 10.0 * position_diff
            
            # Task complexity and estimated costs
            total_energy += task.complexity * 5.0
            total_energy += task.estimated_energy
            total_energy += task.estimated_latency * 0.1
            
            completed_tasks.add(task_idx)
        
        return total_energy
    
    def superposition_search(self, circuit: PhotonicCircuit, 
                           optimization_targets: List[str] = None) -> Dict[str, Any]:
        """
        Use quantum superposition to explore multiple optimization paths simultaneously.
        
        Args:
            circuit: PhotonicCircuit to optimize
            optimization_targets: List of metrics to optimize ['energy', 'latency', 'area']
            
        Returns:
            Dictionary containing optimization results for different targets
        """
        if optimization_targets is None:
            optimization_targets = ['energy', 'latency', 'area']
        
        logger.info(f"Starting superposition search for targets: {optimization_targets}")
        
        results = {}
        
        for target in optimization_targets:
            logger.info(f"Exploring optimization path for: {target}")
            
            # Create quantum superposition of possible optimizations
            optimization_tasks = self._create_optimization_superposition(circuit, target)
            
            # Collapse superposition through quantum measurement
            collapsed_tasks = []
            for task in optimization_tasks:
                measurement_probability = self._calculate_measurement_probability(task, target)
                task.collapse_state(measurement_probability)
                
                if task.quantum_state == QuantumState.COLLAPSED:
                    collapsed_tasks.append(task)
            
            # Apply quantum annealing to collapsed tasks
            optimized_tasks = self.quantum_annealing_optimization(collapsed_tasks)
            
            # Execute optimized tasks and measure results
            execution_result = self._execute_quantum_tasks(optimized_tasks, circuit)
            results[target] = execution_result
        
        logger.info(f"Superposition search completed for {len(results)} targets")
        return results
    
    def _create_optimization_superposition(self, circuit: PhotonicCircuit, 
                                         target: str) -> List[QuantumTask]:
        """Create quantum superposition of optimization tasks for specific target."""
        tasks = []
        
        if target == 'energy':
            # Energy-focused optimization tasks
            tasks.extend([
                QuantumTask(id="reduce_laser_power", priority=0.9, complexity=3),
                QuantumTask(id="optimize_thermal_tuning", priority=0.8, complexity=4),
                QuantumTask(id="minimize_crosstalk", priority=0.7, complexity=3),
                QuantumTask(id="power_gating", priority=0.6, complexity=2)
            ])
        
        elif target == 'latency':
            # Latency-focused optimization tasks
            tasks.extend([
                QuantumTask(id="pipeline_optimization", priority=0.9, complexity=4),
                QuantumTask(id="reduce_propagation_delay", priority=0.8, complexity=3),
                QuantumTask(id="parallel_processing", priority=0.7, complexity=5),
                QuantumTask(id="buffer_optimization", priority=0.6, complexity=2)
            ])
        
        elif target == 'area':
            # Area-focused optimization tasks
            tasks.extend([
                QuantumTask(id="component_sharing", priority=0.9, complexity=4),
                QuantumTask(id="layout_compaction", priority=0.8, complexity=5),
                QuantumTask(id="minimize_routing", priority=0.7, complexity=3),
                QuantumTask(id="resource_reuse", priority=0.6, complexity=3)
            ])
        
        # Initialize all tasks in superposition
        for task in tasks:
            task.quantum_state = QuantumState.SUPERPOSITION
            # Complex probability amplitude representing superposition
            phase = random.uniform(0, 2 * math.pi)
            task.probability_amplitude = complex(
                math.cos(phase) / math.sqrt(len(tasks)),
                math.sin(phase) / math.sqrt(len(tasks))
            )
        
        return tasks
    
    def _calculate_measurement_probability(self, task: QuantumTask, target: str) -> float:
        """Calculate probability of measuring a task as executable."""
        base_probability = abs(task.probability_amplitude) ** 2
        
        # Adjust based on target alignment
        target_alignment = {
            'energy': {'reduce_laser_power': 1.0, 'optimize_thermal_tuning': 0.9, 'power_gating': 0.8},
            'latency': {'pipeline_optimization': 1.0, 'parallel_processing': 0.9, 'buffer_optimization': 0.7},
            'area': {'component_sharing': 1.0, 'layout_compaction': 0.95, 'resource_reuse': 0.8}
        }
        
        alignment_factor = target_alignment.get(target, {}).get(task.id, 0.5)
        
        # Include task priority and complexity
        priority_factor = task.priority
        complexity_factor = 1.0 / (1.0 + task.complexity * 0.1)
        
        final_probability = base_probability * alignment_factor * priority_factor * complexity_factor
        return min(final_probability, 1.0)
    
    def _execute_quantum_tasks(self, tasks: List[QuantumTask], 
                              circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Execute quantum tasks and return performance metrics."""
        logger.info(f"Executing {len(tasks)} quantum tasks")
        
        # Simulate task execution with quantum effects
        total_improvement = 0.0
        executed_optimizations = []
        
        for task in tasks:
            if task.quantum_state in [QuantumState.COLLAPSED, QuantumState.OPTIMIZED]:
                # Simulate optimization effect
                improvement = self._simulate_task_effect(task, circuit)
                total_improvement += improvement
                executed_optimizations.append({
                    'task_id': task.id,
                    'improvement': improvement,
                    'complexity': task.complexity
                })
        
        # Calculate final metrics with quantum effects
        baseline_metrics = circuit.analyze_circuit()
        
        quantum_enhanced_metrics = CircuitMetrics(
            energy_per_op=baseline_metrics.energy_per_op * (1.0 - total_improvement * 0.1),
            latency=baseline_metrics.latency * (1.0 - total_improvement * 0.05),
            area=baseline_metrics.area * (1.0 - total_improvement * 0.08),
            power=baseline_metrics.power * (1.0 - total_improvement * 0.12),
            throughput=baseline_metrics.throughput * (1.0 + total_improvement * 0.15),
            accuracy=min(baseline_metrics.accuracy * (1.0 + total_improvement * 0.02), 1.0)
        )
        
        return {
            'baseline_metrics': baseline_metrics.to_dict(),
            'optimized_metrics': quantum_enhanced_metrics.to_dict(),
            'improvement_factor': total_improvement,
            'executed_optimizations': executed_optimizations,
            'quantum_efficiency': self._calculate_quantum_efficiency(tasks)
        }
    
    def _simulate_task_effect(self, task: QuantumTask, circuit: PhotonicCircuit) -> float:
        """Simulate the optimization effect of executing a quantum task."""
        # Base improvement based on task complexity and circuit size
        base_improvement = task.complexity * 0.02
        circuit_complexity = len(circuit.layers) * circuit.total_components
        
        # Quantum enhancement factor based on superposition and entanglement
        quantum_factor = 1.0
        if len(task.entangled_tasks) > 0:
            quantum_factor += len(task.entangled_tasks) * 0.1
        
        # Task-specific improvements
        task_improvements = {
            'reduce_laser_power': 0.15,
            'optimize_thermal_tuning': 0.12,
            'pipeline_optimization': 0.20,
            'parallel_processing': 0.25,
            'component_sharing': 0.18,
            'layout_compaction': 0.22
        }
        
        specific_improvement = task_improvements.get(task.id, 0.05)
        
        total_improvement = (base_improvement + specific_improvement) * quantum_factor
        return min(total_improvement, 0.8)  # Cap at 80% improvement
    
    def _calculate_quantum_efficiency(self, tasks: List[QuantumTask]) -> float:
        """Calculate quantum efficiency of the task execution."""
        if not tasks:
            return 0.0
        
        # Measure quantum coherence preservation
        entangled_pairs = 0
        superposition_maintained = 0
        
        for task in tasks:
            if task.quantum_state == QuantumState.ENTANGLED:
                entangled_pairs += len(task.entangled_tasks)
            if task.quantum_state in [QuantumState.SUPERPOSITION, QuantumState.OPTIMIZED]:
                superposition_maintained += 1
        
        coherence_factor = superposition_maintained / len(tasks)
        entanglement_factor = min(entangled_pairs / len(tasks), 1.0)
        
        quantum_efficiency = (coherence_factor + entanglement_factor) / 2.0
        return quantum_efficiency
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about quantum optimization performance."""
        return {
            'total_tasks_registered': len(self.task_registry),
            'optimization_runs': len(self.optimization_history),
            'average_optimization_energy': np.mean([h['best_energy'] for h in self.optimization_history]) if self.optimization_history else 0,
            'task_state_distribution': self._get_task_state_distribution(),
            'entanglement_statistics': self._get_entanglement_statistics(),
            'quantum_advantages_realized': self._calculate_quantum_advantages()
        }
    
    def _get_task_state_distribution(self) -> Dict[str, int]:
        """Get distribution of quantum states across all tasks."""
        distribution = {state.value: 0 for state in QuantumState}
        for task in self.task_registry.values():
            distribution[task.quantum_state.value] += 1
        return distribution
    
    def _get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get statistics about task entanglement."""
        if not self.task_registry:
            return {'total_entangled_pairs': 0, 'max_entanglement_degree': 0, 'average_entanglement': 0.0}
        
        entangled_pairs = 0
        max_entanglement = 0
        
        for task in self.task_registry.values():
            entangled_count = len(task.entangled_tasks)
            entangled_pairs += entangled_count
            max_entanglement = max(max_entanglement, entangled_count)
        
        return {
            'total_entangled_pairs': entangled_pairs // 2,  # Divide by 2 to avoid double counting
            'max_entanglement_degree': max_entanglement,
            'average_entanglement': entangled_pairs / len(self.task_registry)
        }
    
    def _calculate_quantum_advantages(self) -> Dict[str, float]:
        """Calculate quantum advantages over classical planning."""
        if len(self.optimization_history) < 2:
            return {'search_space_reduction': 0.0, 'convergence_acceleration': 0.0}
        
        # Estimate search space reduction through quantum parallelism
        classical_search_space = math.factorial(min(8, len(self.task_registry)))  # Classical permutation search
        quantum_search_space = len(self.task_registry) ** 2  # Quantum parallel search
        
        search_reduction = 1.0 - (quantum_search_space / classical_search_space) if classical_search_space > 0 else 0.0
        
        # Measure convergence acceleration
        recent_energies = [h['best_energy'] for h in self.optimization_history[-5:]]
        convergence_rate = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies) if len(recent_energies) > 1 else 0.0
        
        return {
            'search_space_reduction': max(0.0, min(1.0, search_reduction)),
            'convergence_acceleration': convergence_rate,
            'quantum_speedup_factor': math.sqrt(max(1.0, len(self.task_registry)))  # Theoretical quantum speedup
        }