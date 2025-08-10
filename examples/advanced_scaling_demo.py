#!/usr/bin/env python3
"""
Advanced Auto-Scaling and Performance Optimization Demo

This comprehensive demo showcases the enterprise-grade auto-scaling and performance
optimization capabilities of the photonic foundry system, including:

1. Advanced Resource Management and Optimization
2. Sophisticated Load Balancing and Distributed Processing  
3. Concurrent Processing Pools and Resource Pooling
4. Adaptive Caching with Machine Learning Optimization
5. Advanced Auto-Scaling Triggers and Predictive Scaling
6. Comprehensive Performance Monitoring and Metrics Collection
7. Enterprise-Grade Configuration Management

This demo simulates a realistic production workload and demonstrates how all
components work together to provide optimal performance and scalability.
"""

import sys
import time
import random
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import as_completed
import logging

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our advanced scaling components
from photonic_foundry.advanced_scaling import (
    AdvancedScalingConfig, EnterpriseAutoScaler, LoadBalancingAlgorithm,
    ResourceLimits, ScalingMode, get_enterprise_scaler, 
    start_enterprise_scaling, stop_enterprise_scaling
)

from photonic_foundry.concurrent_processing import (
    DistributedTaskExecutor, TaskPriority, get_distributed_executor,
    StreamProcessor, ActorSystem, Actor, start_concurrent_processing,
    stop_concurrent_processing
)

from photonic_foundry.intelligent_caching import (
    create_intelligent_cache, CachePolicy, CompressionType
)

from photonic_foundry.performance_analytics import (
    get_performance_analyzer, start_performance_monitoring,
    stop_performance_monitoring, measure_time, profile_performance
)

from photonic_foundry.enterprise_config import (
    get_config_manager, Environment, ConfigurationManager,
    load_config_from_file, set_config, get_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhotonicWorkloadSimulator:
    """Simulates realistic photonic processing workloads."""
    
    def __init__(self):
        self.workload_types = [
            "matrix_multiplication",
            "neural_network_inference", 
            "quantum_circuit_optimization",
            "photonic_routing",
            "signal_processing"
        ]
        
    def generate_workload(self, workload_type: str = None) -> dict:
        """Generate a simulated workload."""
        if not workload_type:
            workload_type = random.choice(self.workload_types)
            
        # Simulate different workload characteristics
        workload_profiles = {
            "matrix_multiplication": {
                "cpu_intensity": random.uniform(0.7, 0.95),
                "memory_usage_mb": random.randint(100, 500),
                "expected_duration": random.uniform(0.1, 2.0),
                "data_size": random.randint(1000, 10000)
            },
            "neural_network_inference": {
                "cpu_intensity": random.uniform(0.5, 0.8),
                "memory_usage_mb": random.randint(200, 800),
                "expected_duration": random.uniform(0.05, 1.0),
                "data_size": random.randint(500, 5000)
            },
            "quantum_circuit_optimization": {
                "cpu_intensity": random.uniform(0.8, 0.99),
                "memory_usage_mb": random.randint(50, 300),
                "expected_duration": random.uniform(0.2, 5.0),
                "data_size": random.randint(100, 2000)
            },
            "photonic_routing": {
                "cpu_intensity": random.uniform(0.3, 0.6),
                "memory_usage_mb": random.randint(50, 200),
                "expected_duration": random.uniform(0.01, 0.5),
                "data_size": random.randint(10, 1000)
            },
            "signal_processing": {
                "cpu_intensity": random.uniform(0.6, 0.85),
                "memory_usage_mb": random.randint(100, 400),
                "expected_duration": random.uniform(0.1, 1.5),
                "data_size": random.randint(1000, 8000)
            }
        }
        
        profile = workload_profiles[workload_type].copy()
        profile["type"] = workload_type
        profile["created_at"] = datetime.now()
        
        return profile
        
    def execute_workload(self, workload: dict) -> dict:
        """Execute a simulated workload."""
        start_time = time.time()
        
        # Simulate processing time with some variability
        base_duration = workload["expected_duration"]
        actual_duration = base_duration * random.uniform(0.8, 1.4)
        
        # Simulate CPU-intensive work
        iterations = int(workload["data_size"] * workload["cpu_intensity"])
        result = 0
        for i in range(iterations):
            result += i * 0.1
            if i % 1000 == 0:
                time.sleep(0.001)  # Simulate I/O or memory access
                
        execution_time = time.time() - start_time
        
        # Simulate occasional failures
        success = random.random() > 0.05  # 5% failure rate
        
        return {
            "success": success,
            "execution_time": execution_time,
            "result": result if success else None,
            "cpu_time": execution_time * workload["cpu_intensity"],
            "memory_peak_mb": workload["memory_usage_mb"] * random.uniform(0.8, 1.2),
            "error": None if success else f"Simulated failure in {workload['type']}"
        }


class PerformanceDashboard:
    """Real-time performance dashboard."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics_history = []
        self.is_running = False
        
    def start(self):
        """Start dashboard updates."""
        self.is_running = True
        dashboard_thread = threading.Thread(target=self._update_loop, daemon=True)
        dashboard_thread.start()
        
    def stop(self):
        """Stop dashboard updates."""
        self.is_running = False
        
    def _update_loop(self):
        """Dashboard update loop."""
        while self.is_running:
            try:
                # Get comprehensive status from all systems
                enterprise_scaler = get_enterprise_scaler()
                distributed_executor = get_distributed_executor()
                performance_analyzer = get_performance_analyzer()
                
                dashboard_data = {
                    "timestamp": datetime.now(),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "enterprise_scaling": enterprise_scaler.get_comprehensive_status(),
                    "distributed_processing": distributed_executor.get_comprehensive_stats(),
                    "performance_analytics": performance_analyzer.get_performance_dashboard()
                }
                
                self.metrics_history.append(dashboard_data)
                
                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-50:]
                    
                # Print dashboard update
                self._print_dashboard(dashboard_data)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                time.sleep(10)
                
    def _print_dashboard(self, data):
        """Print dashboard information."""
        timestamp = data["timestamp"].strftime("%H:%M:%S")
        uptime = int(data["uptime_seconds"])
        
        # Enterprise scaling info
        scaling_info = data["enterprise_scaling"]
        current_workers = scaling_info["scaling_config"]["current_workers"]
        
        # Performance info
        perf_info = data["performance_analytics"]
        system_health = perf_info.get("system_health", {})
        overall_score = system_health.get("overall_score", 0)
        
        print(f"\n=== Photonic Foundry Performance Dashboard [{timestamp}] ===")
        print(f"Uptime: {uptime}s | Workers: {current_workers} | Health Score: {overall_score:.1f}/100")
        print(f"Resource Utilization: CPU {system_health.get('component_scores', {}).get('cpu', 0):.1f}% | "
              f"Memory {system_health.get('component_scores', {}).get('memory', 0):.1f}%")


class LoadTester:
    """Generate realistic load patterns for testing scaling."""
    
    def __init__(self, simulator: PhotonicWorkloadSimulator):
        self.simulator = simulator
        self.load_patterns = {
            "constant": self._constant_load,
            "burst": self._burst_load,
            "ramp_up": self._ramp_up_load,
            "sine_wave": self._sine_wave_load,
            "random": self._random_load
        }
        
    def generate_load_pattern(self, pattern_name: str, duration_seconds: int, 
                            base_rate: float = 1.0) -> list:
        """Generate a load pattern over time."""
        if pattern_name not in self.load_patterns:
            pattern_name = "constant"
            
        pattern_func = self.load_patterns[pattern_name]
        return pattern_func(duration_seconds, base_rate)
        
    def _constant_load(self, duration: int, base_rate: float) -> list:
        """Constant load pattern."""
        return [base_rate] * duration
        
    def _burst_load(self, duration: int, base_rate: float) -> list:
        """Burst load pattern with periodic spikes."""
        load_values = []
        for i in range(duration):
            if i % 30 == 0 and i > 0:  # Burst every 30 seconds
                load_values.extend([base_rate * 5] * 5)  # 5-second burst
                i += 4  # Skip ahead
            else:
                load_values.append(base_rate)
        return load_values[:duration]
        
    def _ramp_up_load(self, duration: int, base_rate: float) -> list:
        """Gradually increasing load."""
        return [base_rate * (1 + i / duration) for i in range(duration)]
        
    def _sine_wave_load(self, duration: int, base_rate: float) -> list:
        """Sine wave load pattern."""
        import math
        return [base_rate * (1 + 0.8 * math.sin(2 * math.pi * i / 60)) 
                for i in range(duration)]
        
    def _random_load(self, duration: int, base_rate: float) -> list:
        """Random load with some correlation."""
        load_values = []
        current_rate = base_rate
        
        for _ in range(duration):
            # Random walk with bounds
            change = random.uniform(-0.2, 0.2) * base_rate
            current_rate = max(0.1 * base_rate, 
                             min(3.0 * base_rate, current_rate + change))
            load_values.append(current_rate)
            
        return load_values


@profile_performance("workload_execution")
def execute_photonic_workload(workload_data: dict) -> dict:
    """Execute a photonic workload with performance profiling."""
    simulator = PhotonicWorkloadSimulator()
    
    with measure_time("workload_processing", {"type": workload_data["type"]}):
        result = simulator.execute_workload(workload_data)
        
    return result


async def run_comprehensive_demo():
    """Run comprehensive demo of all advanced scaling capabilities."""
    logger.info("Starting Comprehensive Advanced Scaling Demo")
    logger.info("=" * 60)
    
    # 1. Initialize Configuration Management
    logger.info("Step 1: Initializing Enterprise Configuration Management")
    config_manager = get_config_manager(Environment.DEVELOPMENT)
    
    # Set up configuration
    scaling_config = {
        "min_workers": 2,
        "max_workers": 12,
        "target_cpu_utilization": 65.0,
        "predictive_scaling_enabled": True,
        "burst_detection_enabled": True
    }
    
    for key, value in scaling_config.items():
        set_config(f"scaling.{key}", value)
        
    logger.info(f"Configuration loaded: {len(scaling_config)} settings")
    
    # 2. Initialize Performance Monitoring
    logger.info("Step 2: Starting Performance Monitoring")
    start_performance_monitoring()
    
    # 3. Initialize Advanced Auto-Scaling
    logger.info("Step 3: Initializing Enterprise Auto-Scaling")
    
    advanced_config = AdvancedScalingConfig(
        min_workers=get_config("scaling.min_workers", 2),
        max_workers=get_config("scaling.max_workers", 8),
        target_cpu_utilization=get_config("scaling.target_cpu_utilization", 65.0),
        scaling_mode=ScalingMode.ADAPTIVE,
        load_balancing_algorithm=LoadBalancingAlgorithm.ADAPTIVE_WEIGHTED,
        enable_multi_tier=True,
        enable_resource_pooling=True,
        circuit_breaker_enabled=True
    )
    
    start_enterprise_scaling()
    enterprise_scaler = get_enterprise_scaler(advanced_config)
    
    # 4. Initialize Concurrent Processing
    logger.info("Step 4: Starting Distributed Task Execution")
    start_concurrent_processing()
    distributed_executor = get_distributed_executor()
    
    # 5. Initialize Intelligent Caching
    logger.info("Step 5: Setting up Intelligent Caching")
    intelligent_cache = create_intelligent_cache(
        cache_type="high_performance",
        max_size=5000,
        max_memory_mb=200
    )
    
    # 6. Initialize Components
    simulator = PhotonicWorkloadSimulator()
    load_tester = LoadTester(simulator)
    dashboard = PerformanceDashboard()
    
    # 7. Start Dashboard
    logger.info("Step 6: Starting Performance Dashboard")
    dashboard.start()
    
    # 8. Run Load Test Scenarios
    logger.info("Step 7: Running Load Test Scenarios")
    
    scenarios = [
        ("constant", 60, 1.0, "Steady state load"),
        ("burst", 90, 1.5, "Burst load with spikes"),
        ("ramp_up", 120, 0.5, "Gradually increasing load"),
        ("sine_wave", 150, 2.0, "Periodic load variation"),
        ("random", 180, 1.8, "Random load pattern")
    ]
    
    total_tasks_submitted = 0
    total_tasks_completed = 0
    
    for scenario_name, duration, base_rate, description in scenarios:
        logger.info(f"\nRunning Scenario: {scenario_name} - {description}")
        logger.info(f"Duration: {duration}s, Base Rate: {base_rate} tasks/sec")
        
        # Generate load pattern
        load_pattern = load_tester.generate_load_pattern(scenario_name, duration, base_rate)
        
        scenario_start = time.time()
        
        for second, target_rate in enumerate(load_pattern):
            second_start = time.time()
            
            # Calculate tasks to submit this second
            tasks_this_second = max(1, int(target_rate))
            
            # Submit tasks
            for _ in range(tasks_this_second):
                workload = simulator.generate_workload()
                
                # Cache workload data
                cache_key = f"workload_{workload['type']}_{int(time.time())}"
                intelligent_cache.put(cache_key, workload)
                
                # Submit to distributed executor
                try:
                    task_id = distributed_executor.submit_task(
                        execute_photonic_workload,
                        workload,
                        priority=TaskPriority.NORMAL,
                        timeout=30.0,
                        resource_requirements={"memory_mb": workload["memory_usage_mb"]}
                    )
                    total_tasks_submitted += 1
                    
                    # Occasionally submit high-priority tasks
                    if random.random() < 0.1:
                        urgent_workload = simulator.generate_workload("quantum_circuit_optimization")
                        distributed_executor.submit_task(
                            execute_photonic_workload,
                            urgent_workload,
                            priority=TaskPriority.HIGH,
                            timeout=60.0
                        )
                        total_tasks_submitted += 1
                        
                except Exception as e:
                    logger.error(f"Failed to submit task: {e}")
                    
            # Wait for next second
            elapsed = time.time() - second_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                
            # Check for completed tasks
            completed_results = distributed_executor.get_results()
            total_tasks_completed += len([r for r in completed_results if r.success])
            
            # Print progress every 30 seconds
            if second % 30 == 0:
                elapsed_time = time.time() - scenario_start
                completion_rate = total_tasks_completed / max(total_tasks_submitted, 1)
                logger.info(f"  Progress: {second}/{duration}s | "
                          f"Submitted: {total_tasks_submitted} | "
                          f"Completed: {total_tasks_completed} | "
                          f"Success Rate: {completion_rate:.1%}")
                          
        logger.info(f"Scenario '{scenario_name}' completed")
        
        # Brief pause between scenarios
        time.sleep(10)
    
    # 9. Final System Analysis
    logger.info("\nStep 8: Generating Final Performance Analysis")
    
    # Wait for remaining tasks to complete
    logger.info("Waiting for remaining tasks to complete...")
    distributed_executor.wait_for_completion(timeout=60)
    
    # Get final results
    final_results = distributed_executor.get_results()
    total_tasks_completed = len([r for r in final_results if r.success])
    
    # Generate comprehensive reports
    enterprise_status = enterprise_scaler.get_comprehensive_status()
    processing_stats = distributed_executor.get_comprehensive_stats()
    cache_stats = intelligent_cache.get_comprehensive_stats()
    performance_report = get_performance_analyzer().generate_performance_report(days=1)
    config_report = config_manager.export_configuration_report()
    
    # 10. Print Summary Report
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE DEMO SUMMARY REPORT")
    logger.info("=" * 80)
    
    logger.info(f"\nTask Execution Summary:")
    logger.info(f"  Total Tasks Submitted: {total_tasks_submitted}")
    logger.info(f"  Total Tasks Completed: {total_tasks_completed}")
    logger.info(f"  Overall Success Rate: {(total_tasks_completed / max(total_tasks_submitted, 1)):.1%}")
    
    logger.info(f"\nAuto-Scaling Performance:")
    scaling_config = enterprise_status["scaling_config"]
    logger.info(f"  Final Worker Count: {scaling_config['current_workers']}")
    logger.info(f"  Scaling Events (24h): {enterprise_status.get('scaling_events_24h', 0)}")
    logger.info(f"  Circuit Breaker State: {enterprise_status['circuit_breaker']['state']}")
    
    logger.info(f"\nCaching Performance:")
    basic_stats = cache_stats["basic_stats"]
    logger.info(f"  Hit Rate: {basic_stats['hit_rate']:.1f}%")
    logger.info(f"  Total Requests: {basic_stats['total_requests']}")
    logger.info(f"  Compression Ratio: {basic_stats['compression_ratio']:.2f}")
    
    logger.info(f"\nSystem Health:")
    system_health = performance_report.get("bottleneck_analysis", {})
    logger.info(f"  Performance Score: {system_health.get('performance_score', 0):.1f}/100")
    logger.info(f"  Identified Bottlenecks: {len(system_health.get('bottlenecks', []))}")
    
    logger.info(f"\nConfiguration Management:")
    config_stats = config_report["statistics"]
    logger.info(f"  Total Configurations: {config_stats['total_configurations']}")
    logger.info(f"  Encrypted Configurations: {config_stats['encrypted_configurations']}")
    logger.info(f"  Schema Validated: {config_stats['schema_validated_configurations']}")
    
    # 11. Cleanup
    logger.info("\nStep 9: Cleaning up resources")
    dashboard.stop()
    stop_concurrent_processing()
    stop_enterprise_scaling()
    stop_performance_monitoring()
    
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED SCALING DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nAll enterprise-grade scaling capabilities have been demonstrated:")
    logger.info("✓ Advanced Resource Management and Optimization")
    logger.info("✓ Sophisticated Load Balancing and Distributed Processing")
    logger.info("✓ Concurrent Processing Pools and Resource Pooling")
    logger.info("✓ Adaptive Caching with Machine Learning Optimization")
    logger.info("✓ Advanced Auto-Scaling Triggers and Predictive Scaling")
    logger.info("✓ Comprehensive Performance Monitoring and Metrics Collection")
    logger.info("✓ Enterprise-Grade Configuration Management")
    
    logger.info(f"\nThe system successfully processed {total_tasks_completed} tasks across")
    logger.info("multiple load patterns while automatically optimizing performance,")
    logger.info("scaling resources, and maintaining high availability.")


if __name__ == "__main__":
    print("🚀 Photonic Foundry Advanced Auto-Scaling Demo")
    print("=" * 60)
    print("This comprehensive demo showcases enterprise-grade auto-scaling")
    print("and performance optimization capabilities.")
    print()
    print("The demo will:")
    print("• Initialize all scaling and optimization systems")
    print("• Run realistic workload scenarios")
    print("• Demonstrate predictive scaling and load balancing")
    print("• Show intelligent caching and performance monitoring")
    print("• Generate comprehensive performance reports")
    print()
    
    try:
        # Run the async demo
        asyncio.run(run_comprehensive_demo())
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nThank you for exploring the Photonic Foundry scaling capabilities!")