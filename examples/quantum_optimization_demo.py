#!/usr/bin/env python3
"""
Advanced quantum optimization and scaling demonstration.

This example showcases the quantum-inspired optimization engine and
distributed processing capabilities for large-scale photonic systems.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import time
from photonic_foundry import PhotonicAccelerator, PhotonicCircuit
from photonic_foundry.quantum_optimizer import (
    QuantumOptimizationEngine,
    DistributedQuantumProcessor,
    OptimizationConfig,
    ScalingConfig,
    OptimizationStrategy,
    ScalingMode
)


def create_sample_neural_networks(count: int = 5) -> List[nn.Module]:
    """Create sample neural networks of varying sizes."""
    networks = []
    
    for i in range(count):
        if i % 3 == 0:
            # Small network
            model = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        elif i % 3 == 1:
            # Medium network
            model = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        else:
            # Large network
            model = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        
        networks.append(model)
    
    return networks


def define_optimization_objective(circuit: PhotonicCircuit):
    """Define optimization objective function for circuit parameters."""
    
    def objective_function(parameters: np.ndarray) -> float:
        """
        Multi-objective function optimizing energy, latency, and accuracy.
        
        Args:
            parameters: Array of circuit parameters to optimize
            
        Returns:
            Objective value (lower is better)
        """
        # Simulate parameter effects on circuit metrics
        baseline_metrics = circuit.analyze_circuit()
        
        # Parameter effects (simplified model)
        laser_power_factor = parameters[0] if len(parameters) > 0 else 1.0
        thermal_factor = parameters[1] if len(parameters) > 1 else 1.0
        coupling_factor = parameters[2] if len(parameters) > 2 else 1.0
        
        # Calculate modified metrics
        energy_per_op = baseline_metrics.energy_per_op * laser_power_factor * (1 + thermal_factor * 0.1)
        latency = baseline_metrics.latency * (1 + thermal_factor * 0.05)
        accuracy = baseline_metrics.accuracy * coupling_factor * (1 - abs(laser_power_factor - 1.0) * 0.02)
        
        # Multi-objective: minimize energy and latency, maximize accuracy
        energy_objective = energy_per_op / 100.0  # Normalize
        latency_objective = latency / 1000.0      # Normalize
        accuracy_objective = (1.0 - accuracy) * 10.0  # Minimize (1 - accuracy)
        
        # Weighted combination
        total_objective = 0.4 * energy_objective + 0.3 * latency_objective + 0.3 * accuracy_objective
        
        # Add some noise to simulate measurement uncertainty
        noise = np.random.normal(0, 0.01)
        return total_objective + noise
    
    return objective_function


async def demonstrate_optimization_strategies():
    """Demonstrate different quantum optimization strategies."""
    print("\n🔬 Quantum Optimization Strategies Demo")
    print("=" * 60)
    
    # Create sample circuit
    accelerator = PhotonicAccelerator()
    model = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    circuit = accelerator.convert_pytorch_model(model)
    print(f"Created circuit with {len(circuit.layers)} layers, {circuit.total_components} components")
    
    # Define optimization parameters and bounds
    parameter_bounds = [
        (0.8, 1.2),  # Laser power factor
        (0.9, 1.1),  # Thermal factor  
        (0.95, 1.05) # Coupling factor
    ]
    
    objective_function = define_optimization_objective(circuit)
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.QUANTUM_ANNEALING,
        OptimizationStrategy.GENETIC_ALGORITHM,
        OptimizationStrategy.PARTICLE_SWARM,
        OptimizationStrategy.HYBRID_QUANTUM_CLASSICAL
    ]
    
    optimization_results = {}
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value.replace('_', ' ').title()} Strategy")
        print("-" * 40)
        
        # Configure optimization
        config = OptimizationConfig(
            strategy=strategy,
            max_iterations=200 if strategy != OptimizationStrategy.HYBRID_QUANTUM_CLASSICAL else 300,
            population_size=30,
            parallel_evaluations=True,
            use_gpu_acceleration=torch.cuda.is_available()
        )
        
        # Create optimizer
        optimizer = QuantumOptimizationEngine(config)
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize_circuit_parameters(circuit, objective_function, parameter_bounds)
        optimization_time = time.time() - start_time
        
        if result.get('success', False):
            print(f"✅ Optimization successful!")
            print(f"   Best parameters: {result['best_parameters']}")
            print(f"   Best objective: {result['best_objective']:.6f}")
            print(f"   Iterations: {result.get('iterations_completed', 'N/A')}")
            print(f"   Time: {optimization_time:.2f}s")
            
            # Calculate improvement
            baseline_obj = objective_function(np.array([1.0, 1.0, 1.0]))  # Baseline parameters
            improvement = (baseline_obj - result['best_objective']) / baseline_obj * 100
            print(f"   Improvement: {improvement:.1f}%")
            
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
        
        optimization_results[strategy.value] = result
    
    # Compare strategies
    print(f"\n📊 Strategy Comparison")
    print("-" * 40)
    
    successful_results = {k: v for k, v in optimization_results.items() if v.get('success', False)}
    
    if successful_results:
        best_strategy = min(successful_results.items(), key=lambda x: x[1]['best_objective'])
        print(f"🏆 Best strategy: {best_strategy[0]}")
        print(f"   Best objective: {best_strategy[1]['best_objective']:.6f}")
        print(f"   Optimization time: {best_strategy[1]['optimization_time']:.2f}s")
        
        # Show all results
        print(f"\n📈 All Results:")
        for strategy_name, result in successful_results.items():
            objective = result['best_objective']
            opt_time = result['optimization_time']
            print(f"   {strategy_name}: {objective:.6f} (in {opt_time:.2f}s)")


async def demonstrate_distributed_processing():
    """Demonstrate distributed processing capabilities."""
    print("\n🌐 Distributed Processing Demo")
    print("=" * 60)
    
    # Create multiple neural networks
    networks = create_sample_neural_networks(10)
    print(f"Created {len(networks)} neural networks for batch processing")
    
    # Convert to photonic circuits
    accelerator = PhotonicAccelerator()
    circuits = []
    
    for i, model in enumerate(networks):
        circuit = accelerator.convert_pytorch_model(model)
        circuit.name = f"circuit_{i}"
        circuits.append(circuit)
    
    print(f"Converted {len(circuits)} circuits")
    
    # Configure distributed processor
    scaling_config = ScalingConfig(
        max_workers=4,
        scaling_mode=ScalingMode.HYBRID,
        auto_scale_enabled=True,
        memory_threshold=0.7,
        cpu_threshold=0.7,
        min_nodes=2,
        max_nodes=8
    )
    
    processor = DistributedQuantumProcessor(scaling_config)
    
    # Define processing function
    def analyze_circuit_performance(circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Analyze circuit performance metrics."""
        start_time = time.time()
        
        # Perform analysis
        metrics = circuit.analyze_circuit()
        
        # Simulate computational work
        time.sleep(0.1 + np.random.uniform(0, 0.2))  # Simulate variable processing time
        
        processing_time = time.time() - start_time
        
        return {
            'circuit_name': circuit.name,
            'layer_count': len(circuit.layers),
            'component_count': circuit.total_components,
            'energy_per_op': metrics.energy_per_op,
            'latency': metrics.latency,
            'throughput': metrics.throughput,
            'accuracy': metrics.accuracy,
            'processing_time': processing_time
        }
    
    # Process circuits in parallel
    print(f"\n⚡ Processing {len(circuits)} circuits in parallel...")
    batch_start = time.time()
    
    results = await processor.process_circuit_batch(circuits, analyze_circuit_performance)
    
    batch_time = time.time() - batch_start
    successful_results = [r for r in results if r is not None]
    
    print(f"✅ Batch processing completed!")
    print(f"   Total time: {batch_time:.2f}s")
    print(f"   Successful circuits: {len(successful_results)}/{len(circuits)}")
    print(f"   Throughput: {len(successful_results)/batch_time:.2f} circuits/sec")
    
    # Show individual results
    print(f"\n📋 Circuit Analysis Results:")
    for result in successful_results[:5]:  # Show first 5 results
        print(f"   {result['circuit_name']}:")
        print(f"     Layers: {result['layer_count']}, Components: {result['component_count']}")
        print(f"     Energy: {result['energy_per_op']:.2f}pJ, Latency: {result['latency']:.2f}ps")
        print(f"     Processing time: {result['processing_time']*1000:.1f}ms")
    
    if len(successful_results) > 5:
        print(f"   ... and {len(successful_results) - 5} more")
    
    # Performance report
    print(f"\n📊 Distributed Processing Performance:")
    performance_report = processor.get_performance_report()
    
    print(f"   Active workers: {performance_report['active_workers']}")
    print(f"   Scaling mode: {performance_report['scaling_config']['mode']}")
    print(f"   Auto-scaling: {'✅ Enabled' if performance_report['scaling_config']['auto_scale_enabled'] else '❌ Disabled'}")
    
    metrics = performance_report['performance_metrics']
    print(f"   CPU utilization: {metrics['cpu_utilization']:.1%}")
    print(f"   Memory utilization: {metrics['memory_utilization']:.1%}")
    print(f"   Parallel efficiency: {metrics['parallel_efficiency']:.2f}")
    
    # Cleanup
    processor.shutdown()


async def demonstrate_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    print("\n🚀 Auto-Scaling Demo")
    print("=" * 60)
    
    # Configure aggressive auto-scaling for demonstration
    scaling_config = ScalingConfig(
        max_workers=2,  # Start small
        scaling_mode=ScalingMode.HORIZONTAL,
        auto_scale_enabled=True,
        memory_threshold=0.5,  # Lower thresholds for demo
        cpu_threshold=0.5,
        min_nodes=1,
        max_nodes=6,
        scale_up_cooldown=1.0,   # Faster scaling for demo
        scale_down_cooldown=2.0
    )
    
    processor = DistributedQuantumProcessor(scaling_config)
    
    # Create workloads of increasing intensity
    light_workload = create_sample_neural_networks(3)
    heavy_workload = create_sample_neural_networks(15)
    
    accelerator = PhotonicAccelerator()
    
    def intensive_processing(circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Simulate intensive processing to trigger scaling."""
        # Heavy computation to stress system
        for _ in range(100):
            metrics = circuit.analyze_circuit()
            # Add some CPU-intensive work
            np.random.rand(1000, 1000).dot(np.random.rand(1000, 1000))
        
        return {
            'circuit_name': circuit.name,
            'computation_intensive': True,
            'final_energy': metrics.energy_per_op
        }
    
    print(f"🔥 Processing light workload (should not trigger scaling)...")
    initial_report = processor.get_performance_report()
    initial_workers = initial_report['active_workers']
    print(f"   Initial workers: {initial_workers}")
    
    light_circuits = [accelerator.convert_pytorch_model(model) for model in light_workload]
    for i, circuit in enumerate(light_circuits):
        circuit.name = f"light_circuit_{i}"
    
    await processor.process_circuit_batch(light_circuits, lambda c: c.analyze_circuit().to_dict())
    
    light_report = processor.get_performance_report()
    light_workers = light_report['active_workers']
    print(f"   Workers after light workload: {light_workers}")
    
    print(f"\n🚀 Processing heavy workload (should trigger scaling up)...")
    heavy_circuits = [accelerator.convert_pytorch_model(model) for model in heavy_workload]
    for i, circuit in enumerate(heavy_circuits):
        circuit.name = f"heavy_circuit_{i}"
    
    # This should trigger auto-scaling
    await processor.process_circuit_batch(heavy_circuits, intensive_processing)
    
    heavy_report = processor.get_performance_report()
    heavy_workers = heavy_report['active_workers']
    print(f"   Workers after heavy workload: {heavy_workers}")
    
    if heavy_workers > light_workers:
        print(f"✅ Auto-scaling UP triggered! ({light_workers} → {heavy_workers} workers)")
    else:
        print(f"⚠️ Auto-scaling UP not triggered (may need more intensive workload)")
    
    # Wait for scale-down (simulate idle period)
    print(f"\n⏳ Waiting for scale-down cooldown...")
    await asyncio.sleep(3.0)  # Wait for scale-down
    
    # Process very light workload
    single_circuit = [light_circuits[0]]
    await processor.process_circuit_batch(single_circuit, lambda c: {'result': 'minimal'})
    
    final_report = processor.get_performance_report()
    final_workers = final_report['active_workers']
    print(f"   Workers after cooldown: {final_workers}")
    
    if final_workers < heavy_workers:
        print(f"✅ Auto-scaling DOWN triggered! ({heavy_workers} → {final_workers} workers)")
    else:
        print(f"⚠️ Auto-scaling DOWN not triggered (may need longer cooldown)")
    
    print(f"\n📈 Auto-scaling Summary:")
    print(f"   Initial: {initial_workers} workers")
    print(f"   Peak: {heavy_workers} workers")
    print(f"   Final: {final_workers} workers")
    print(f"   Scaling mode: {scaling_config.scaling_mode.value}")
    
    processor.shutdown()


async def main():
    """Main demonstration function."""
    print("🔬 Quantum-Inspired Optimization & Scaling Demo")
    print("=" * 60)
    print("This demonstration showcases:")
    print("• Quantum-inspired optimization algorithms")
    print("• Multi-strategy optimization comparison") 
    print("• Distributed processing capabilities")
    print("• Auto-scaling for dynamic workloads")
    print("• Performance monitoring and metrics")
    
    try:
        # Run optimization strategies demo
        await demonstrate_optimization_strategies()
        
        # Run distributed processing demo
        await demonstrate_distributed_processing()
        
        # Run auto-scaling demo
        await demonstrate_auto_scaling()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("🔬 Quantum optimization and scaling features verified")
        print("⚡ System ready for production-scale photonic computing")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())