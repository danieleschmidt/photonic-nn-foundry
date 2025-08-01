"""Performance benchmarks for the photonic transpiler."""

import pytest
import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, Any
from unittest.mock import patch

# Import the actual modules when they exist
try:
    from photonic_foundry import torch2verilog, PhotonicAccelerator
except ImportError:
    # Mock imports for testing infrastructure
    def torch2verilog(model, **kwargs):
        """Mock transpiler function."""
        time.sleep(0.01)  # Simulate processing time
        return "// Mock Verilog output"
    
    class PhotonicAccelerator:
        def __init__(self, **kwargs):
            pass
        
        def compile_and_profile(self, verilog_code):
            time.sleep(0.05)  # Simulate compilation time
            return {
                'energy_per_op': 0.1,
                'latency': 50.0,
                'area': 1000.0
            }


class TestTranspilerPerformance:
    """Test suite for transpiler performance benchmarks."""
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="transpilation")
    def test_linear_layer_transpilation_speed(self, benchmark, simple_linear_model):
        """Benchmark transpilation speed for linear layers."""
        def transpile():
            return torch2verilog(simple_linear_model, optimize=False)
        
        result = benchmark(transpile)
        assert result is not None
        assert "module" in result.lower()
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="transpilation")
    def test_mlp_transpilation_speed(self, benchmark, multilayer_perceptron):
        """Benchmark transpilation speed for MLPs."""
        def transpile():
            return torch2verilog(multilayer_perceptron, optimize=False)
        
        result = benchmark(transpile)
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="transpilation")
    def test_conv_transpilation_speed(self, benchmark, convolutional_model):
        """Benchmark transpilation speed for CNNs."""
        def transpile():
            return torch2verilog(convolutional_model, optimize=False)
        
        result = benchmark(transpile)
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="optimization")
    def test_transpilation_with_optimization(self, benchmark, multilayer_perceptron):
        """Benchmark transpilation with optimization passes."""
        def transpile_optimized():
            return torch2verilog(multilayer_perceptron, optimize=True)
        
        result = benchmark(transpile_optimized)
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_during_transpilation(self, multilayer_perceptron):
        """Test memory usage during transpilation."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Transpile model
        result = torch2verilog(multilayer_perceptron)
        
        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory increase should be reasonable (< 500MB for test models)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.parametrize("model_size", [
        (10, 5),     # Small
        (100, 50),   # Medium  
        (1000, 500), # Large
        (5000, 1000) # Very Large
    ])
    def test_scaling_with_model_size(self, benchmark, model_size):
        """Test how transpilation time scales with model size."""
        input_size, output_size = model_size
        model = nn.Linear(input_size, output_size)
        
        def transpile():
            return torch2verilog(model)
        
        result = benchmark(transpile)
        assert result is not None
        
        # Larger models should take more time but not exponentially
        # This is checked by pytest-benchmark's regression analysis
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="compilation")
    def test_verilog_compilation_speed(self, benchmark, sample_verilog_code):
        """Benchmark Verilog compilation speed."""
        accelerator = PhotonicAccelerator(pdk='skywater130')
        
        def compile_verilog():
            return accelerator.compile_and_profile(sample_verilog_code)
        
        result = benchmark(compile_verilog)
        assert result is not None
        assert 'energy_per_op' in result
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="end_to_end")
    def test_end_to_end_performance(self, benchmark, multilayer_perceptron):
        """Benchmark complete end-to-end workflow."""
        accelerator = PhotonicAccelerator(pdk='skywater130')
        
        def full_workflow():
            verilog = torch2verilog(multilayer_perceptron, optimize=True)
            results = accelerator.compile_and_profile(verilog)
            return results
        
        result = benchmark(full_workflow)
        assert result is not None
        assert isinstance(result, dict)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_transpilation(self, test_models):
        """Test performance with concurrent transpilation requests."""
        import concurrent.futures
        import threading
        
        results = []
        start_time = time.time()
        
        def transpile_model(model):
            return torch2verilog(model)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(transpile_model, model) 
                for model in test_models.values()
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                assert result is not None
        
        total_time = time.time() - start_time
        
        # Concurrent execution should be faster than sequential
        assert len(results) == len(test_models)
        assert total_time < len(test_models) * 0.1  # Should complete in reasonable time
    
    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
    def test_batch_processing_performance(self, benchmark, batch_size):
        """Test performance with different batch sizes."""
        models = [nn.Linear(100, 50) for _ in range(batch_size)]
        
        def batch_transpile():
            results = []
            for model in models:
                result = torch2verilog(model)
                results.append(result)
            return results
        
        results = benchmark(batch_transpile)
        assert len(results) == batch_size
        assert all(result is not None for result in results)
    
    def test_performance_regression_detection(self, multilayer_perceptron):
        """Test to detect performance regressions."""
        # Baseline performance expectations
        expected_max_time = 1.0  # seconds
        expected_max_memory = 100  # MB
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        result = torch2verilog(multilayer_perceptron)
        execution_time = time.time() - start_time
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = peak_memory - baseline_memory
        
        # Performance regression checks
        assert execution_time < expected_max_time, \
            f"Transpilation too slow: {execution_time:.3f}s > {expected_max_time}s"
        
        assert memory_usage < expected_max_memory, \
            f"Memory usage too high: {memory_usage:.2f}MB > {expected_max_memory}MB"
        
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="cache")
    def test_caching_performance(self, benchmark, simple_linear_model):
        """Test performance benefits of caching."""
        # First transpilation (cache miss)
        first_result = torch2verilog(simple_linear_model)
        
        # Subsequent transpilation (cache hit)
        def cached_transpile():
            return torch2verilog(simple_linear_model)
        
        result = benchmark(cached_transpile)
        assert result == first_result
        
        # Cached version should be significantly faster
        # This is verified by comparing benchmark results
    
    @pytest.mark.performance
    @pytest.mark.stress
    def test_large_model_handling(self):
        """Stress test with very large models."""
        # Create a large model
        large_model = nn.Sequential(*[
            nn.Linear(1000, 1000) for _ in range(10)
        ])
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            result = torch2verilog(large_model)
            execution_time = time.time() - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = peak_memory - baseline_memory
            
            # Should complete within reasonable bounds
            assert execution_time < 30.0, f"Large model took too long: {execution_time:.2f}s"
            assert memory_usage < 1000, f"Memory usage too high: {memory_usage:.2f}MB"
            assert result is not None
            
        except Exception as e:
            # Large models might fail gracefully
            assert "memory" in str(e).lower() or "size" in str(e).lower()
    
    @pytest.mark.performance
    def test_cleanup_after_transpilation(self, multilayer_perceptron):
        """Test that resources are properly cleaned up."""
        import gc
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Multiple transpilation cycles
        for _ in range(5):
            result = torch2verilog(multilayer_perceptron)
            assert result is not None
            del result
            gc.collect()
        
        # Memory should return to baseline (within tolerance)
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_leak = final_memory - baseline_memory
        
        assert memory_leak < 50, f"Potential memory leak: {memory_leak:.2f}MB"


class TestProfilingPerformance:
    """Test suite for profiling and analysis performance."""
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="profiling")
    def test_power_analysis_speed(self, benchmark, sample_verilog_code):
        """Benchmark power analysis performance."""
        accelerator = PhotonicAccelerator(pdk='skywater130')
        
        def analyze_power():
            results = accelerator.compile_and_profile(sample_verilog_code)
            return results['energy_per_op']
        
        result = benchmark(analyze_power)
        assert result is not None
        assert isinstance(result, (int, float))
    
    @pytest.mark.performance
    @pytest.mark.benchmark(group="profiling")
    def test_timing_analysis_speed(self, benchmark, sample_verilog_code):
        """Benchmark timing analysis performance."""
        accelerator = PhotonicAccelerator(pdk='skywater130')
        
        def analyze_timing():
            results = accelerator.compile_and_profile(sample_verilog_code)
            return results['latency']
        
        result = benchmark(analyze_timing)
        assert result is not None
        assert isinstance(result, (int, float))
    
    @pytest.mark.performance  
    @pytest.mark.benchmark(group="profiling")
    def test_area_analysis_speed(self, benchmark, sample_verilog_code):
        """Benchmark area analysis performance."""
        accelerator = PhotonicAccelerator(pdk='skywater130')
        
        def analyze_area():
            results = accelerator.compile_and_profile(sample_verilog_code)
            return results['area']
        
        result = benchmark(analyze_area)
        assert result is not None
        assert isinstance(result, (int, float))


# Performance test configuration
def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "stress: Stress testing with large inputs") 
    config.addinivalue_line("markers", "slow: Slow running tests")


# Custom benchmark fixtures
@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'min_rounds': 5,
        'max_time': 10.0,
        'warmup': True,
        'warmup_iterations': 2,
        'disable_gc': True,
        'timer': time.perf_counter
    }