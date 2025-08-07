#!/usr/bin/env python3
"""
Quantum-inspired task planning demonstration.

This example shows how to use the quantum task planner for optimizing
photonic neural network compilation and execution.
"""

import torch
import torch.nn as nn
import numpy as np
from photonic_foundry import (
    PhotonicAccelerator, 
    QuantumTaskPlanner,
    ResourceConstraint,
    QuantumState
)


def main():
    """Demonstrate quantum-inspired task planning for photonic circuits."""
    print("🔬 Quantum-Inspired Photonic Task Planning Demo")
    print("=" * 60)
    
    # Initialize photonic accelerator
    accelerator = PhotonicAccelerator(pdk="skywater130", wavelength=1550.0)
    
    # Set resource constraints for quantum planning
    constraints = ResourceConstraint(
        max_energy=100.0,  # pJ
        max_latency=500.0,  # ps
        max_area=5.0,      # mm²
        max_concurrent_tasks=8,
        thermal_limit=75.0  # Celsius
    )
    
    # Initialize quantum task planner
    quantum_planner = QuantumTaskPlanner(accelerator, constraints)
    print(f"✅ Initialized QuantumTaskPlanner with constraints")
    
    # Create a sample neural network
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(), 
        nn.Linear(128, 10)
    )
    
    print(f"📊 Neural Network Architecture:")
    print(f"   Layers: {len([m for m in model.modules() if isinstance(m, nn.Linear)])}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Convert to photonic circuit
    print("\n🔄 Converting PyTorch model to photonic circuit...")
    circuit = accelerator.convert_pytorch_model(model)
    
    print(f"✅ Photonic Circuit Created:")
    print(f"   Layers: {len(circuit.layers)}")
    print(f"   Total Components: {circuit.total_components}")
    
    # Create quantum compilation plan
    print("\n⚛️ Creating quantum compilation plan...")
    compilation_tasks = quantum_planner.create_circuit_compilation_plan(circuit)
    
    print(f"✅ Generated {len(compilation_tasks)} quantum tasks:")
    for task in compilation_tasks:
        print(f"   - {task.id}: Priority={task.priority:.1f}, Complexity={task.complexity}")
        print(f"     State: {task.quantum_state.value}, Dependencies: {task.dependencies}")
    
    # Apply quantum annealing optimization
    print("\n🌡️ Applying quantum annealing optimization...")
    optimized_tasks = quantum_planner.quantum_annealing_optimization(compilation_tasks)
    
    print("✅ Optimization completed. Task execution order:")
    for i, task in enumerate(optimized_tasks):
        print(f"   {i+1}. {task.id} (State: {task.quantum_state.value})")
    
    # Demonstrate superposition search
    print("\n🔀 Running superposition search for multi-objective optimization...")
    optimization_targets = ['energy', 'latency', 'area']
    
    superposition_results = quantum_planner.superposition_search(
        circuit, 
        optimization_targets
    )
    
    print("✅ Superposition search completed:")
    for target, result in superposition_results.items():
        print(f"\n📈 {target.upper()} Optimization:")
        baseline = result['baseline_metrics']
        optimized = result['optimized_metrics']
        
        print(f"   Energy: {baseline['energy_per_op']:.1f} → {optimized['energy_per_op']:.1f} pJ")
        print(f"   Latency: {baseline['latency']:.1f} → {optimized['latency']:.1f} ps")
        print(f"   Area: {baseline['area']:.3f} → {optimized['area']:.3f} mm²")
        print(f"   Improvement Factor: {result['improvement_factor']:.1%}")
        print(f"   Quantum Efficiency: {result['quantum_efficiency']:.1%}")
    
    # Show quantum statistics
    print("\n📊 Quantum Optimization Statistics:")
    stats = quantum_planner.get_optimization_statistics()
    
    print(f"   Total Tasks: {stats['total_tasks_registered']}")
    print(f"   Optimization Runs: {stats['optimization_runs']}")
    
    if stats['optimization_runs'] > 0:
        print(f"   Average Energy: {stats['average_optimization_energy']:.2f}")
    
    print(f"   Task States: {stats['task_state_distribution']}")
    print(f"   Entanglement: {stats['entanglement_statistics']}")
    
    quantum_advantages = stats['quantum_advantages_realized']
    print(f"   Quantum Advantages:")
    print(f"     Search Space Reduction: {quantum_advantages['search_space_reduction']:.1%}")
    print(f"     Convergence Acceleration: {quantum_advantages['convergence_acceleration']:.3f}")
    print(f"     Quantum Speedup Factor: {quantum_advantages['quantum_speedup_factor']:.1f}x")
    
    # Demonstrate inference with quantum-optimized circuit
    print("\n🎯 Running quantum-optimized inference...")
    
    # Generate sample input
    batch_size = 32
    input_data = np.random.randn(batch_size, 784) * 0.1
    
    # Simulate inference
    output_data, inference_time = accelerator.simulate_inference(circuit, input_data)
    
    print(f"✅ Inference completed:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Shape: {input_data.shape}")
    print(f"   Output Shape: {output_data.shape}")
    print(f"   Inference Time: {inference_time*1000:.2f} ms")
    print(f"   Throughput: {batch_size/inference_time:.0f} samples/sec")
    
    # Calculate quantum enhancement metrics
    baseline_metrics = circuit.analyze_circuit()
    quantum_enhanced = superposition_results['energy']
    
    energy_improvement = (baseline_metrics.energy_per_op - 
                         quantum_enhanced['optimized_metrics']['energy_per_op']) / baseline_metrics.energy_per_op
    latency_improvement = (baseline_metrics.latency - 
                          quantum_enhanced['optimized_metrics']['latency']) / baseline_metrics.latency
    
    print(f"\n🚀 Overall Quantum Enhancement:")
    print(f"   Energy Reduction: {energy_improvement:.1%}")
    print(f"   Latency Reduction: {latency_improvement:.1%}")
    print(f"   Quantum Advantage Realized: {'✅ Yes' if energy_improvement > 0.1 else '⚠️ Limited'}")
    
    print("\n" + "=" * 60)
    print("🎉 Quantum task planning demonstration completed!")
    print("🔬 The quantum planner successfully optimized photonic circuit compilation")
    print("⚛️ Quantum effects: superposition, entanglement, and annealing applied")


if __name__ == "__main__":
    main()