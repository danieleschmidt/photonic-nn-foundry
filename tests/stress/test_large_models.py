"""Stress tests for large models and high-load scenarios."""

import pytest
import torch
import gc
import time
from typing import List

from tests.utils import (
    requires_large_memory,
    slow_test,
    performance_test,
    ResourceMonitor,
    PerformanceTimer,
    timeout
)


class TestLargeModelStress:
    """Stress tests for large neural network models."""
    
    @slow_test
    @requires_large_memory(8.0)
    def test_very_large_linear_model(self):
        """Test processing of very large linear models."""
        # Create a model with many large layers
        layers = []
        layer_sizes = [1024, 2048, 4096, 2048, 1024, 512, 256, 128, 64, 10]
        
        for i in range(len(layer_sizes) - 1):
            layers.extend([
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
        
        large_model = torch.nn.Sequential(*layers)
        
        with ResourceMonitor() as monitor:
            # Test forward pass with large batch
            batch_size = 1024
            input_data = torch.randn(batch_size, layer_sizes[0])
            
            large_model.eval()
            with torch.no_grad():
                output = large_model(input_data)
            
            assert output.shape == (batch_size, layer_sizes[-1])
        
        # Verify reasonable resource usage
        if monitor.memory_delta is not None:
            assert monitor.memory_delta < 8000  # Less than 8GB memory increase
    
    @slow_test
    @requires_large_memory(4.0)
    def test_deep_convolutional_model(self):
        """Test processing of very deep CNN models."""
        layers = []
        
        # Create a ResNet-like deep model
        in_channels = 3
        for block in range(20):  # 20 blocks
            out_channels = min(64 * (2 ** (block // 5)), 512)
            
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            ])
            
            in_channels = out_channels
        
        layers.extend([
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, 1000)
        ])
        
        deep_model = torch.nn.Sequential(*layers)
        
        with ResourceMonitor() as monitor:
            # Test with reasonable input size
            input_data = torch.randn(8, 3, 224, 224)
            
            deep_model.eval()
            with torch.no_grad():
                output = deep_model(input_data)
            
            assert output.shape == (8, 1000)
        
        # Clean up
        del deep_model
        gc.collect()
    
    @slow_test
    @performance_test
    @timeout(300)  # 5 minute timeout
    def test_batch_processing_stress(self, sample_linear_model):
        """Test stress with very large batch sizes."""
        model = sample_linear_model
        model.eval()
        
        batch_sizes = [1024, 2048, 4096, 8192]
        processing_times = []
        
        for batch_size in batch_sizes:
            try:
                input_data = torch.randn(batch_size, 10)
                
                with PerformanceTimer(f"batch_{batch_size}") as timer:
                    with torch.no_grad():
                        output = model(input_data)
                
                assert output.shape == (batch_size, 1)
                processing_times.append(timer.duration)
                
                # Clean up
                del input_data, output
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pytest.skip(f"Insufficient memory for batch size {batch_size}")
                else:
                    raise
        
        # Verify reasonable scaling
        if len(processing_times) >= 2:
            # Processing time should scale roughly linearly with batch size
            time_ratio = processing_times[-1] / processing_times[0]
            batch_ratio = batch_sizes[-1] / batch_sizes[0]
            
            # Allow up to 2x overhead for larger batches
            assert time_ratio < batch_ratio * 2


class TestMemoryStress:
    """Memory-related stress tests."""
    
    @slow_test
    @requires_large_memory(4.0)
    def test_memory_intensive_operations(self):
        """Test memory-intensive operations."""
        # Create multiple large tensors
        tensors = []
        tensor_size = (1000, 1000)
        num_tensors = 50
        
        with ResourceMonitor() as monitor:
            for i in range(num_tensors):
                tensor = torch.randn(*tensor_size)
                # Perform some operations to stress memory
                result = torch.matmul(tensor, tensor.t())
                tensors.append(result)
        
        # Verify all tensors are correct size
        for tensor in tensors:
            assert tensor.shape == tensor_size
        
        # Clean up
        del tensors
        gc.collect()
    
    @slow_test
    def test_memory_leak_detection(self, sample_linear_model):
        """Test for memory leaks in repeated operations."""
        model = sample_linear_model
        model.eval()
        
        initial_memory = None
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        # Perform many iterations
        for i in range(1000):
            input_data = torch.randn(32, 10)
            with torch.no_grad():
                output = model(input_data)
            
            # Clean up explicitly
            del input_data, output
            
            # Force garbage collection periodically
            if i % 100 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Allow some memory increase but not too much (potential leak indicator)
        assert memory_increase < 1000, f"Potential memory leak: {memory_increase:.1f}MB increase"
    
    @slow_test
    @requires_large_memory(2.0)
    def test_gradient_memory_stress(self):
        """Test memory usage with gradient computation."""
        # Create a model that requires significant memory for gradients
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 10)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        with ResourceMonitor() as monitor:
            for epoch in range(10):
                for batch in range(100):
                    # Large batch size to stress memory
                    input_data = torch.randn(128, 1024)
                    targets = torch.randint(0, 10, (128,))
                    
                    optimizer.zero_grad()
                    outputs = model(input_data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    # Clean up
                    del input_data, targets, outputs, loss
                
                # Force cleanup between epochs
                gc.collect()
        
        # Verify reasonable memory usage
        if monitor.memory_delta is not None:
            assert monitor.memory_delta < 4000  # Less than 4GB increase


class TestConcurrencyStress:
    """Concurrency and threading stress tests."""
    
    @slow_test
    @performance_test
    def test_parallel_model_inference(self, sample_linear_model):
        """Test parallel inference with multiple threads."""
        import threading
        import concurrent.futures
        
        model = sample_linear_model
        model.eval()
        
        def run_inference(thread_id: int) -> float:
            """Run inference in a separate thread."""
            start_time = time.time()
            
            for i in range(100):
                input_data = torch.randn(16, 10)
                with torch.no_grad():
                    output = model(input_data)
                assert output.shape == (16, 1)
            
            return time.time() - start_time
        
        # Run with multiple threads
        num_threads = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_inference, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all threads completed successfully
        assert len(results) == num_threads
        assert all(time_taken > 0 for time_taken in results)
    
    @slow_test
    def test_repeated_model_creation(self):
        """Test stress from creating and destroying many models."""
        models_created = 0
        
        with ResourceMonitor() as monitor:
            for i in range(500):
                # Create model
                model = torch.nn.Sequential(
                    torch.nn.Linear(100, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 1)
                )
                
                # Test forward pass
                input_data = torch.randn(10, 100)
                output = model(input_data)
                assert output.shape == (10, 1)
                
                models_created += 1
                
                # Clean up
                del model, input_data, output
                
                # Periodic cleanup
                if i % 50 == 0:
                    gc.collect()
        
        assert models_created == 500
        
        # Final cleanup
        gc.collect()


class TestResourceLimits:
    """Test behavior under resource constraints."""
    
    @slow_test
    def test_model_size_limits(self):
        """Test handling of models approaching memory limits."""
        max_attempts = 10
        successful_size = 0
        
        for attempt in range(max_attempts):
            try:
                # Exponentially increase model size
                size = 512 * (2 ** attempt)
                
                model = torch.nn.Sequential(
                    torch.nn.Linear(size, size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(size, 1)
                )
                
                # Test forward pass
                input_data = torch.randn(1, size)
                output = model(input_data)
                assert output.shape == (1, 1)
                
                successful_size = size
                
                # Clean up
                del model, input_data, output
                gc.collect()
                
            except (RuntimeError, MemoryError) as e:
                if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                    # Expected when hitting limits
                    break
                else:
                    raise
        
        # Should successfully handle at least some reasonable size
        assert successful_size >= 512, f"Failed at minimum expected size: {successful_size}"
    
    @slow_test
    @timeout(60)  # 1 minute timeout
    def test_time_limits(self, large_model):
        """Test behavior with time-constrained operations."""
        model = large_model
        model.eval()
        
        start_time = time.time()
        iterations = 0
        max_time = 30  # 30 seconds
        
        while time.time() - start_time < max_time:
            input_data = torch.randn(8, 512)
            with torch.no_grad():
                output = model(input_data)
            
            iterations += 1
            
            # Clean up
            del input_data, output
            
            # Periodic cleanup
            if iterations % 10 == 0:
                gc.collect()
        
        elapsed_time = time.time() - start_time
        throughput = iterations / elapsed_time
        
        # Should achieve reasonable throughput
        assert throughput > 0.1, f"Throughput too low: {throughput:.3f} iterations/second"
        assert iterations > 0, "No iterations completed"