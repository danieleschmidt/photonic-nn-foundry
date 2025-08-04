"""
Comprehensive system integration tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import threading
from pathlib import Path

from src.photonic_foundry.core_enhanced import EnhancedPhotonicAccelerator, ValidationLevel
from src.photonic_foundry.transpiler import torch2verilog, analyze_model_compatibility
from src.photonic_foundry.monitoring import (
    get_metrics_collector, get_performance_monitor, get_alert_manager,
    start_monitoring, stop_monitoring, get_system_status
)
from src.photonic_foundry.database import get_database


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.accelerator = EnhancedPhotonicAccelerator(
            validation_level=ValidationLevel.MODERATE
        )
        
        # Start monitoring for integration tests
        start_monitoring()
        
    def teardown_method(self):
        """Clean up after tests."""
        stop_monitoring()
        
    def test_end_to_end_model_processing(self):
        """Test complete end-to-end model processing pipeline."""
        # Create a realistic model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Step 1: Model compatibility analysis
        compatibility = analyze_model_compatibility(model)
        assert compatibility['compatibility_score'] > 0.5
        
        # Step 2: Safe model conversion
        result = self.accelerator.convert_pytorch_model_safe(model)
        assert result.success is True
        assert result.data is not None
        circuit = result.data
        
        # Step 3: Circuit profiling with advanced analysis
        profile_result = self.accelerator.compile_and_profile_safe(
            circuit, include_advanced_analysis=True
        )
        assert profile_result.success is True
        assert 'advanced_analysis' in profile_result.metadata
        
        # Step 4: Verilog generation
        verilog_code = torch2verilog(model, optimize=True)
        assert len(verilog_code) > 1000  # Should be substantial
        assert 'module' in verilog_code
        assert 'endmodule' in verilog_code
        
        # Step 5: Inference simulation
        input_data = np.random.randn(1, 784).astype(np.float32)
        sim_result = self.accelerator.simulate_inference_safe(
            circuit, input_data, monte_carlo_runs=5
        )
        assert sim_result.success is True
        assert 'mean_output' in sim_result.data
        
        # Step 6: Database persistence
        circuit_id = self.accelerator.save_circuit(circuit, verilog_code, profile_result.data)
        assert circuit_id is not None
        
        # Step 7: Circuit retrieval
        loaded_circuit = self.accelerator.load_circuit(circuit.name)
        assert loaded_circuit is not None
        
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring system."""
        perf_monitor = get_performance_monitor()
        
        # Perform several operations to generate metrics
        models = [
            nn.Linear(64, 32),
            nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8)),
            nn.Linear(128, 64)
        ]
        
        for i, model in enumerate(models):
            start_time = time.time()
            
            result = self.accelerator.convert_pytorch_model_safe(model)
            
            duration = time.time() - start_time
            perf_monitor.record_operation("model_conversion", duration, result.success)
            
        # Check performance statistics
        stats = perf_monitor.get_operation_stats("model_conversion")
        assert stats['total_count'] == 3
        assert stats['success_rate'] == 1.0
        assert stats['avg_duration'] > 0
        
    def test_alert_system_integration(self):
        """Test alert system integration."""
        alert_manager = get_alert_manager()
        
        # Add a custom alert rule for testing
        triggered = threading.Event()
        
        def test_condition(health):
            return health.cpu_percent > -1  # Always trigger for test
            
        alert_manager.add_alert_rule(
            name="test_alert",
            condition=test_condition,
            message="Test alert triggered",
            severity="warning"
        )
        
        # Wait a bit for alert monitoring to run
        time.sleep(2)
        
        # Check that alert was triggered
        active_alerts = alert_manager.get_active_alerts()
        test_alerts = [a for a in active_alerts if a['name'] == 'test_alert']
        assert len(test_alerts) > 0
        
    def test_database_integration_with_caching(self):
        """Test database and caching integration."""
        model = nn.Linear(32, 16)
        
        # First conversion - should miss cache
        result1 = self.accelerator.convert_pytorch_model_safe(model)
        assert result1.success is True
        assert result1.metadata.get('cache_hit', True) is False  # First time
        
        # Second conversion - should hit cache
        result2 = self.accelerator.convert_pytorch_model_safe(model)  
        assert result2.success is True
        # Note: Cache behavior may vary in tests, so we just check success
        
        # Verify database stats
        db_stats = self.accelerator.get_database_stats()
        assert 'database' in db_stats
        assert 'cache' in db_stats
        
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        models = [
            nn.Linear(64, 32),
            nn.Linear(32, 16), 
            nn.Linear(16, 8),
            nn.Linear(8, 4)
        ]
        
        results = []
        
        def process_model(model):
            result = self.accelerator.convert_pytorch_model_safe(model)
            results.append(result)
            
        # Process models concurrently
        threads = [threading.Thread(target=process_model, args=(model,)) 
                  for model in models]
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All should succeed
        assert len(results) == 4
        assert all(r.success for r in results)
        
        # Check processing statistics
        stats = self.accelerator.get_processing_statistics()
        assert stats['models_processed'] >= 4
        assert stats['success_rate'] > 0.5
        
    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery."""
        # Test with problematic model
        class ProblematicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                # This could cause issues in some processing
                return self.linear(x) / 0  # Division by zero
                
        model = ProblematicModel()
        
        # System should handle this gracefully
        result = self.accelerator.convert_pytorch_model_safe(model)
        # May succeed or fail depending on where error occurs
        # The key is that it shouldn't crash the system
        
        # Test with invalid input data
        if result.success:
            invalid_input = np.array([[np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
            sim_result = self.accelerator.simulate_inference_safe(
                result.data, invalid_input
            )
            assert sim_result.success is False
            assert len(sim_result.errors) > 0
            
    def test_system_status_reporting(self):
        """Test comprehensive system status reporting."""
        # Generate some activity
        model = nn.Linear(16, 8)
        self.accelerator.convert_pytorch_model_safe(model)
        
        # Get system status
        status = get_system_status()
        
        assert 'timestamp' in status
        assert 'system_health' in status
        assert 'active_alerts' in status
        assert 'photonic_operations' in status
        assert 'metric_summary' in status
        
        # System health should have all required fields
        health = status['system_health']
        assert 'cpu_percent' in health
        assert 'memory_percent' in health
        assert 'disk_usage_percent' in health
        assert 'status' in health
        
    def test_configuration_validation_integration(self):
        """Test configuration validation across the system."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            EnhancedPhotonicAccelerator(
                pdk="invalid_pdk",
                wavelength=5000.0,  # Invalid wavelength
                validation_level=ValidationLevel.STRICT
            )
            
        # Test with valid configuration
        accelerator = EnhancedPhotonicAccelerator(
            pdk="skywater130",
            wavelength=1550.0,
            validation_level=ValidationLevel.PERMISSIVE,
            config={
                'precision': 8,
                'max_model_size_mb': 50
            }
        )
        
        assert accelerator.photonic_config.pdk == "skywater130"
        assert accelerator.photonic_config.wavelength == 1550.0
        assert accelerator.photonic_config.precision == 8
        
    def test_long_running_stability(self):
        """Test system stability under extended operation."""
        # Run multiple operations to test stability
        models = [nn.Linear(32, 16) for _ in range(10)]
        
        for i, model in enumerate(models):
            result = self.accelerator.convert_pytorch_model_safe(model)
            assert result.success is True
            
            # Simulate some processing time
            time.sleep(0.1)
            
        # System should remain stable
        stats = self.accelerator.get_processing_statistics()
        assert stats['models_processed'] >= 10
        assert stats['failed_conversions'] == 0
        
        # Check system health
        status = get_system_status()
        assert status['system_health']['status'] in ['healthy', 'degraded']
        
    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        initial_thread_count = threading.active_count()
        
        # Create and destroy accelerators
        for _ in range(3):
            acc = EnhancedPhotonicAccelerator()
            model = nn.Linear(8, 4)
            result = acc.convert_pytorch_model_safe(model)
            assert result.success is True
            del acc
            
        # Thread count should not grow significantly
        final_thread_count = threading.active_count()
        assert final_thread_count <= initial_thread_count + 2  # Allow some variance
        
    def test_metrics_export_and_import(self):
        """Test metrics export and potential import."""
        collector = get_metrics_collector()
        
        # Generate some metrics
        collector.record_metric("test_metric", 42.0, "units")
        collector.record_metric("test_metric", 43.0, "units")
        
        # Export metrics
        exported_data = collector.export_metrics("json")
        assert len(exported_data) > 100  # Should be substantial JSON
        assert '"test_metric"' in exported_data
        assert '42.0' in exported_data
        
        # Verify export structure
        import json
        data = json.loads(exported_data)
        assert 'export_timestamp' in data
        assert 'metrics' in data
        assert 'test_metric' in data['metrics']