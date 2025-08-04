"""
Test enhanced core functionality with error handling and validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.photonic_foundry.core_enhanced import (
    EnhancedPhotonicAccelerator, 
    ValidationLevel, 
    ProcessingResult
)
from src.photonic_foundry.utils.validators import PhotonicConfig


class TestEnhancedPhotonicAccelerator:
    """Test enhanced photonic accelerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.accelerator = EnhancedPhotonicAccelerator(
            validation_level=ValidationLevel.MODERATE
        )
        
    def test_init_with_validation_levels(self):
        """Test initialization with different validation levels."""
        # Strict validation
        strict_acc = EnhancedPhotonicAccelerator(
            validation_level=ValidationLevel.STRICT
        )
        assert strict_acc.validation_level == ValidationLevel.STRICT
        
        # Permissive validation
        permissive_acc = EnhancedPhotonicAccelerator(
            validation_level=ValidationLevel.PERMISSIVE
        )
        assert permissive_acc.validation_level == ValidationLevel.PERMISSIVE
        
    def test_convert_pytorch_model_safe_success(self):
        """Test safe model conversion with successful case."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Convert with validation
        result = self.accelerator.convert_pytorch_model_safe(model, validate_input=True)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.errors) == 0
        assert 'model_class' in result.metadata
        assert result.processing_time > 0
        
    def test_convert_pytorch_model_safe_with_validation_warnings(self):
        """Test model conversion with validation warnings."""
        # Create model with potential issues
        model = nn.Sequential(
            nn.Linear(1000, 1000),  # Large model
            nn.ReLU(),
            nn.Linear(1000, 1000)
        )
        
        result = self.accelerator.convert_pytorch_model_safe(model, validate_input=True)
        
        # Should succeed with warnings in moderate mode
        assert result.success is True
        assert result.validation_report is not None
        
    def test_convert_pytorch_model_safe_strict_validation_failure(self):
        """Test strict validation failure."""
        strict_acc = EnhancedPhotonicAccelerator(
            validation_level=ValidationLevel.STRICT
        )
        
        # Create problematic model
        model = nn.Sequential(
            nn.Linear(10000, 10000),  # Very large model
        )
        
        with patch.object(strict_acc.model_validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                'is_valid': False,
                'errors': ['Model too large'],
                'warnings': []
            }
            
            result = strict_acc.convert_pytorch_model_safe(model, validate_input=True)
            
            assert result.success is False
            assert 'Model too large' in result.errors
            
    def test_compile_and_profile_safe_success(self):
        """Test safe circuit profiling."""
        # Create and convert a model first
        model = nn.Linear(4, 2)
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        # Profile the circuit
        result = self.accelerator.compile_and_profile_safe(circuit)
        
        assert result.success is True
        assert result.data is not None
        assert hasattr(result.data, 'energy_per_op')
        assert 'circuit_layers' in result.metadata
        
    def test_compile_and_profile_safe_with_advanced_analysis(self):
        """Test circuit profiling with advanced analysis."""
        model = nn.Linear(4, 2)
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        result = self.accelerator.compile_and_profile_safe(
            circuit, include_advanced_analysis=True
        )
        
        assert result.success is True
        assert 'advanced_analysis' in result.metadata
        assert 'energy_breakdown' in result.metadata['advanced_analysis']
        assert 'thermal_analysis' in result.metadata['advanced_analysis']
        
    def test_simulate_inference_safe_single_run(self):
        """Test safe inference simulation with single run."""
        model = nn.Linear(4, 2)
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        input_data = np.random.randn(1, 4).astype(np.float32)
        
        result = self.accelerator.simulate_inference_safe(
            circuit, input_data, monte_carlo_runs=1
        )
        
        assert result.success is True
        assert 'output' in result.data
        assert 'inference_time' in result.data
        assert result.metadata['monte_carlo_runs'] == 1
        
    def test_simulate_inference_safe_monte_carlo(self):
        """Test safe inference simulation with Monte Carlo runs."""
        model = nn.Linear(4, 2)
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        input_data = np.random.randn(1, 4).astype(np.float32)
        
        result = self.accelerator.simulate_inference_safe(
            circuit, input_data, monte_carlo_runs=10
        )
        
        assert result.success is True
        assert 'mean_output' in result.data
        assert 'std_output' in result.data
        assert 'mean_inference_time' in result.data
        assert result.metadata['statistical_analysis'] is True
        
    def test_simulate_inference_safe_invalid_input(self):
        """Test inference simulation with invalid input data."""
        model = nn.Linear(4, 2)
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        # Test with NaN input
        invalid_input = np.array([[1.0, np.nan, 2.0, 3.0]])
        
        result = self.accelerator.simulate_inference_safe(circuit, invalid_input)
        
        assert result.success is False
        assert 'Invalid input data' in result.errors[0]
        
    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        # Perform some operations
        model = nn.Linear(4, 2)
        self.accelerator.convert_pytorch_model_safe(model)
        self.accelerator.convert_pytorch_model_safe(model)  # Should hit cache
        
        stats = self.accelerator.get_processing_statistics()
        
        assert stats['models_processed'] == 2
        assert stats['successful_conversions'] == 2
        assert stats['failed_conversions'] == 0
        assert stats['success_rate'] == 1.0
        assert 'average_processing_time' in stats
        assert 'cache_hit_rate' in stats
        
    def test_error_handling_in_conversion(self):
        """Test error handling during model conversion."""
        # Create mock that raises exception
        with patch.object(self.accelerator, 'convert_pytorch_model') as mock_convert:
            mock_convert.side_effect = RuntimeError("Conversion failed")
            
            model = nn.Linear(4, 2)
            result = self.accelerator.convert_pytorch_model_safe(model, validate_input=False)
            
            assert result.success is False
            assert 'Conversion failed' in result.errors[0]
            assert result.metadata['exception_type'] == 'RuntimeError'
            
    def test_model_hash_calculation(self):
        """Test model hash calculation for caching."""
        model1 = nn.Linear(4, 2)
        model2 = nn.Linear(4, 2)
        model3 = nn.Linear(8, 4)
        
        hash1 = self.accelerator._calculate_model_hash(model1)
        hash2 = self.accelerator._calculate_model_hash(model2)
        hash3 = self.accelerator._calculate_model_hash(model3)
        
        # Same architecture should have same hash
        assert hash1 == hash2
        # Different architecture should have different hash
        assert hash1 != hash3
        assert len(hash1) == 16  # 16 character hash
        
    def test_metrics_validation(self):
        """Test metrics validation logic."""
        from src.photonic_foundry.core import CircuitMetrics
        
        # Valid metrics
        valid_metrics = CircuitMetrics(
            energy_per_op=1.5,
            latency=100.0,
            area=0.5,
            power=10.0,
            throughput=1000.0,
            accuracy=0.95
        )
        assert self.accelerator._validate_metrics(valid_metrics) is True
        
        # Invalid metrics - negative energy
        invalid_metrics = CircuitMetrics(
            energy_per_op=-1.0,
            latency=100.0,
            area=0.5,
            power=10.0,
            throughput=1000.0,
            accuracy=0.95
        )
        assert self.accelerator._validate_metrics(invalid_metrics) is False
        
    def test_input_data_validation(self):
        """Test input data validation."""
        # Valid data
        valid_data = np.array([[1.0, 2.0, 3.0]])
        assert self.accelerator._validate_input_data(valid_data) is True
        
        # Invalid data with NaN
        invalid_data = np.array([[1.0, np.nan, 3.0]])
        assert self.accelerator._validate_input_data(invalid_data) is False
        
        # Invalid data with inf
        invalid_data = np.array([[1.0, np.inf, 3.0]])
        assert self.accelerator._validate_input_data(invalid_data) is False
        
        # Empty data
        empty_data = np.array([])
        assert self.accelerator._validate_input_data(empty_data) is False
        
    def test_thread_safety(self):
        """Test thread safety of processing operations."""
        import threading
        
        model = nn.Linear(4, 2)
        results = []
        
        def convert_model():
            result = self.accelerator.convert_pytorch_model_safe(model)
            results.append(result)
            
        # Run multiple conversions in parallel
        threads = [threading.Thread(target=convert_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # All should succeed
        assert len(results) == 5
        assert all(r.success for r in results)
        
    def test_component_utilization_analysis(self):
        """Test component utilization analysis."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        utilization = self.accelerator._analyze_component_utilization(circuit)
        
        assert 'component_distribution' in utilization
        assert 'utilization_efficiency' in utilization
        assert 'component_density' in utilization
        assert utilization['utilization_efficiency'] >= 0
        assert utilization['utilization_efficiency'] <= 1
        
    def test_scaling_characteristics_analysis(self):
        """Test scaling characteristics analysis."""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        conversion_result = self.accelerator.convert_pytorch_model_safe(model)
        circuit = conversion_result.data
        
        scaling = self.accelerator._analyze_scaling_characteristics(circuit)
        
        assert 'parameter_to_component_ratio' in scaling
        assert 'layers_to_components_ratio' in scaling
        assert 'scaling_complexity' in scaling
        assert 'estimated_max_frequency_ghz' in scaling
        assert scaling['estimated_max_frequency_ghz'] > 0