"""
Unit tests for quantum task planner functionality.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import time

from photonic_foundry.quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumState,
    ResourceConstraint
)
from photonic_foundry.core import PhotonicAccelerator, PhotonicCircuit


class TestQuantumTask:
    """Test QuantumTask class."""
    
    def test_quantum_task_creation(self):
        """Test basic quantum task creation."""
        task = QuantumTask(
            id="test_task",
            priority=0.8,
            complexity=3,
            dependencies=["dep1", "dep2"],
            resources_required={"compute": 0.5, "memory": 0.3}
        )
        
        assert task.id == "test_task"
        assert task.priority == 0.8
        assert task.complexity == 3
        assert task.dependencies == ["dep1", "dep2"]
        assert task.resources_required == {"compute": 0.5, "memory": 0.3}
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert task.probability_amplitude == 1.0 + 0j
    
    def test_collapse_state(self):
        """Test quantum state collapse."""
        task = QuantumTask(id="collapse_test")
        
        # Test successful collapse
        task.collapse_state(1.0)  # 100% probability
        assert task.quantum_state == QuantumState.COLLAPSED
        assert task.probability_amplitude == 1.0 + 0j
        
        # Reset and test failed collapse
        task.quantum_state = QuantumState.SUPERPOSITION
        task.collapse_state(0.0)  # 0% probability
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert task.probability_amplitude == 0.0 + 0j


class TestResourceConstraint:
    """Test ResourceConstraint class."""
    
    def test_default_constraints(self):
        """Test default resource constraints."""
        constraints = ResourceConstraint()
        
        assert constraints.max_energy == float('inf')
        assert constraints.max_latency == float('inf')
        assert constraints.max_area == float('inf')
        assert constraints.max_concurrent_tasks == 16
        assert constraints.thermal_limit == 85.0
    
    def test_custom_constraints(self):
        """Test custom resource constraints."""
        constraints = ResourceConstraint(
            max_energy=100.0,
            max_latency=500.0,
            max_area=2.0,
            max_concurrent_tasks=8,
            thermal_limit=75.0
        )
        
        assert constraints.max_energy == 100.0
        assert constraints.max_latency == 500.0
        assert constraints.max_area == 2.0
        assert constraints.max_concurrent_tasks == 8
        assert constraints.thermal_limit == 75.0


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner class."""
    
    @pytest.fixture
    def accelerator(self):
        """Create PhotonicAccelerator fixture."""
        return PhotonicAccelerator()
    
    @pytest.fixture
    def planner(self, accelerator):
        """Create QuantumTaskPlanner fixture."""
        constraints = ResourceConstraint(
            max_energy=50.0,
            max_latency=200.0,
            max_concurrent_tasks=4
        )
        return QuantumTaskPlanner(accelerator, constraints)
    
    @pytest.fixture
    def sample_circuit(self, accelerator):
        """Create sample circuit fixture."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        return accelerator.convert_pytorch_model(model)
    
    def test_planner_initialization(self, planner):
        """Test quantum task planner initialization."""
        assert planner.accelerator is not None
        assert planner.constraints is not None
        assert isinstance(planner.task_registry, dict)
        assert isinstance(planner.quantum_superposition, list)
        assert isinstance(planner.optimization_history, list)
    
    def test_register_task(self, planner):
        """Test task registration."""
        task = QuantumTask(id="register_test", priority=0.9)
        
        planner.register_task(task)
        
        assert "register_test" in planner.task_registry
        assert planner.task_registry["register_test"] == task
    
    def test_create_circuit_compilation_plan(self, planner, sample_circuit):
        """Test circuit compilation plan creation."""
        tasks = planner.create_circuit_compilation_plan(sample_circuit)
        
        assert len(tasks) > 0
        assert any(task.id == "circuit_analysis" for task in tasks)
        assert any("layer_verilog" in task.id for task in tasks)
        assert any(task.id == "quantum_optimization" for task in tasks)
        assert any(task.id == "final_compilation" for task in tasks)
        
        # Check dependencies
        analysis_task = next(task for task in tasks if task.id == "circuit_analysis")
        assert len(analysis_task.dependencies) == 0  # No dependencies for analysis
        
        optimization_task = next(task for task in tasks if task.id == "quantum_optimization")
        assert len(optimization_task.dependencies) > 0  # Should depend on layer tasks
    
    def test_quantum_annealing_optimization(self, planner):
        """Test quantum annealing optimization."""
        # Create simple tasks
        tasks = [
            QuantumTask(id="task1", complexity=2, estimated_energy=10.0),
            QuantumTask(id="task2", complexity=1, estimated_energy=5.0, dependencies=["task1"]),
            QuantumTask(id="task3", complexity=3, estimated_energy=15.0)
        ]
        
        # Register tasks
        for task in tasks:
            planner.register_task(task)
        
        # Run optimization
        optimized_tasks = planner.quantum_annealing_optimization(tasks)
        
        assert len(optimized_tasks) == len(tasks)
        assert all(task.quantum_state == QuantumState.OPTIMIZED for task in optimized_tasks)
        
        # Check that dependency order is respected
        task_order = {task.id: i for i, task in enumerate(optimized_tasks)}
        assert task_order["task1"] < task_order["task2"]  # task1 should come before task2
    
    def test_superposition_search(self, planner, sample_circuit):
        """Test superposition search functionality."""
        optimization_targets = ['energy', 'latency']
        
        results = planner.superposition_search(sample_circuit, optimization_targets)
        
        assert 'energy' in results
        assert 'latency' in results
        
        for target, result in results.items():
            assert 'baseline_metrics' in result
            assert 'optimized_metrics' in result
            assert 'improvement_factor' in result
            assert 'executed_optimizations' in result
            assert 'quantum_efficiency' in result
            
            # Check metrics structure
            baseline = result['baseline_metrics']
            optimized = result['optimized_metrics']
            
            assert 'energy_per_op' in baseline and 'energy_per_op' in optimized
            assert 'latency' in baseline and 'latency' in optimized
            assert 'throughput' in baseline and 'throughput' in optimized
    
    def test_optimization_statistics(self, planner, sample_circuit):
        """Test optimization statistics generation."""
        # Create and run some optimization tasks
        tasks = planner.create_circuit_compilation_plan(sample_circuit)
        optimized_tasks = planner.quantum_annealing_optimization(tasks)
        
        stats = planner.get_optimization_statistics()
        
        assert 'total_tasks_registered' in stats
        assert 'optimization_runs' in stats
        assert 'task_state_distribution' in stats
        assert 'entanglement_statistics' in stats
        assert 'quantum_advantages_realized' in stats
        
        # Check task state distribution
        state_dist = stats['task_state_distribution']
        assert all(isinstance(count, int) for count in state_dist.values())
        
        # Check entanglement statistics
        entanglement_stats = stats['entanglement_statistics']
        assert 'total_entangled_pairs' in entanglement_stats
        assert 'max_entanglement_degree' in entanglement_stats
        assert 'average_entanglement' in entanglement_stats
        
        # Check quantum advantages
        advantages = stats['quantum_advantages_realized']
        assert 'search_space_reduction' in advantages
        assert 'convergence_acceleration' in advantages
        assert 'quantum_speedup_factor' in advantages
    
    def test_task_entanglement(self, planner):
        """Test quantum task entanglement creation."""
        tasks = [
            QuantumTask(id="entangled1", resources_required={"compute": 0.5}),
            QuantumTask(id="entangled2", resources_required={"compute": 0.3}),
            QuantumTask(id="entangled3", dependencies=["entangled1"])
        ]
        
        planner._create_task_entanglement(tasks)
        
        # Check that tasks with shared resources are entangled
        assert "entangled2" in tasks[0].entangled_tasks
        assert "entangled1" in tasks[1].entangled_tasks
        
        # Check that dependent tasks are entangled
        assert "entangled1" in tasks[2].entangled_tasks
        assert "entangled3" in tasks[0].entangled_tasks
        
        # Check quantum states
        assert tasks[0].quantum_state == QuantumState.ENTANGLED
        assert tasks[1].quantum_state == QuantumState.ENTANGLED
        assert tasks[2].quantum_state == QuantumState.ENTANGLED
    
    def test_solution_energy_calculation(self, planner):
        """Test solution energy calculation for optimization."""
        tasks = [
            QuantumTask(id="task1", complexity=2, estimated_energy=10.0, estimated_latency=50.0),
            QuantumTask(id="task2", complexity=1, estimated_energy=5.0, estimated_latency=30.0, dependencies=["task1"]),
            QuantumTask(id="task3", complexity=3, estimated_energy=15.0, estimated_latency=80.0)
        ]
        
        # Test valid solution (respects dependencies)
        valid_solution = [0, 1, 2]  # task1, task2, task3
        valid_energy = planner._calculate_solution_energy(valid_solution, tasks)
        
        # Test invalid solution (violates dependencies)
        invalid_solution = [1, 0, 2]  # task2, task1, task3 (task2 before task1)
        invalid_energy = planner._calculate_solution_energy(invalid_solution, tasks)
        
        # Invalid solution should have higher energy due to dependency penalty
        assert invalid_energy > valid_energy
        
        # Both should be positive
        assert valid_energy > 0
        assert invalid_energy > 0
    
    def test_measurement_probability_calculation(self, planner):
        """Test quantum measurement probability calculation."""
        task = QuantumTask(
            id="measurement_test",
            priority=0.8,
            complexity=2,
            probability_amplitude=0.5 + 0.3j
        )
        
        # Test for different targets
        energy_prob = planner._calculate_measurement_probability(task, 'energy')
        latency_prob = planner._calculate_measurement_probability(task, 'latency')
        area_prob = planner._calculate_measurement_probability(task, 'area')
        
        # All probabilities should be between 0 and 1
        assert 0 <= energy_prob <= 1
        assert 0 <= latency_prob <= 1
        assert 0 <= area_prob <= 1
        
        # Test with specific task ID that has alignment
        task.id = "reduce_laser_power"
        energy_prob_aligned = planner._calculate_measurement_probability(task, 'energy')
        
        # Should have higher probability for aligned target
        assert energy_prob_aligned >= energy_prob
    
    @patch('time.time')
    def test_optimization_history_tracking(self, mock_time, planner):
        """Test optimization history tracking."""
        mock_time.return_value = 1234567890.0
        
        tasks = [QuantumTask(id="history_test")]
        
        # Run optimization
        planner.quantum_annealing_optimization(tasks)
        
        # Check history was recorded
        assert len(planner.optimization_history) > 0
        
        history_entry = planner.optimization_history[0]
        assert 'iteration' in history_entry
        assert 'final_temperature' in history_entry
        assert 'best_energy' in history_entry
        assert 'optimization_time' in history_entry
    
    def test_quantum_neighbor_generation(self, planner):
        """Test quantum neighbor solution generation."""
        current_solution = [0, 1, 2, 3]
        
        # Generate multiple neighbors
        neighbors = []
        for _ in range(10):
            neighbor = planner._quantum_neighbor(current_solution)
            neighbors.append(neighbor)
        
        # All neighbors should be different from current (with high probability)
        different_neighbors = [n for n in neighbors if n != current_solution]
        assert len(different_neighbors) > 5  # At least half should be different
        
        # All neighbors should be valid permutations
        for neighbor in neighbors:
            assert len(neighbor) == len(current_solution)
            assert set(neighbor) == set(current_solution)
    
    def test_create_optimization_superposition(self, planner, sample_circuit):
        """Test optimization superposition creation."""
        # Test for different targets
        energy_tasks = planner._create_optimization_superposition(sample_circuit, 'energy')
        latency_tasks = planner._create_optimization_superposition(sample_circuit, 'latency')
        area_tasks = planner._create_optimization_superposition(sample_circuit, 'area')
        
        # Each should create tasks
        assert len(energy_tasks) > 0
        assert len(latency_tasks) > 0
        assert len(area_tasks) > 0
        
        # All tasks should be in superposition initially
        for tasks in [energy_tasks, latency_tasks, area_tasks]:
            for task in tasks:
                assert task.quantum_state == QuantumState.SUPERPOSITION
                assert abs(task.probability_amplitude) > 0
        
        # Should have target-specific tasks
        energy_task_ids = [task.id for task in energy_tasks]
        assert any("reduce_laser_power" in task_id for task_id in energy_task_ids)
        
        latency_task_ids = [task.id for task in latency_tasks]
        assert any("pipeline_optimization" in task_id for task_id in latency_task_ids)
        
        area_task_ids = [task.id for task in area_tasks]
        assert any("component_sharing" in task_id for task_id in area_task_ids)