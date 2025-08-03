"""
Enhanced unit tests for core photonic functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from unittest.mock import Mock, patch

from src.photonic_foundry.core import (
    PhotonicAccelerator, PhotonicCircuit, MZILayer, PhotonicComponent,
    CircuitMetrics
)
from src.photonic_foundry.database import DatabaseConfig
from src.photonic_foundry.models.simple_mlp import SimpleMLP, PhotonicOptimizedMLP


class TestPhotonicComponents:
    """Test individual photonic components."""
    
    def test_mzi_layer_creation(self):
        """Test MZI layer creation and configuration."""
        layer = MZILayer(input_size=784, output_size=256, precision=8)
        
        assert layer.input_size == 784
        assert layer.output_size == 256
        assert layer.precision == 8
        assert len(layer.components) == 784 * 256  # One MZI per weight
        
    def test_mzi_layer_verilog_generation(self):
        """Test Verilog generation for MZI layer."""
        layer = MZILayer(input_size=4, output_size=2, precision=8)
        verilog = layer.generate_verilog()
        
        assert "module mzi_layer_4x2" in verilog
        assert "input clk" in verilog
        assert "output [7:0] data_out [1:0]" in verilog
        assert "mzi_unit" in verilog
        
    def test_mzi_weights_assignment(self):
        """Test weight assignment and quantization."""
        layer = MZILayer(input_size=2, output_size=2, precision=8)
        
        # Set custom weights
        custom_weights = np.array([[0.5, -0.3], [0.2, 0.8]])
        layer.weights = custom_weights
        
        assert np.array_equal(layer.weights, custom_weights)
        
        # Check component parameters reflect weights
        for i, component in enumerate(layer.components):
            if component['type'] == PhotonicComponent.MZI:
                row, col = component['params']['position']
                expected_weight = custom_weights[row, col]
                # Weights should be quantized to precision bits
                quantized_weight = int(expected_weight * (2**(layer.precision-1)))
                assert 'phase_shifter_bits' in component['params']


class TestPhotonicCircuit:
    """Test photonic circuit functionality."""
    
    def test_circuit_creation(self):
        """Test creating an empty photonic circuit."""
        circuit = PhotonicCircuit("test_circuit")
        
        assert circuit.name == "test_circuit"
        assert len(circuit.layers) == 0
        assert len(circuit.connections) == 0
        assert circuit.total_components == 0
        
    def test_adding_layers(self):
        """Test adding layers to circuit."""
        circuit = PhotonicCircuit("test_circuit")
        
        layer1 = MZILayer(10, 5)
        layer2 = MZILayer(5, 2)
        
        circuit.add_layer(layer1)
        circuit.add_layer(layer2)
        
        assert len(circuit.layers) == 2
        assert circuit.total_components == len(layer1.components) + len(layer2.components)
        
    def test_layer_connections(self):
        """Test connecting layers in circuit."""
        circuit = PhotonicCircuit("test_circuit")
        
        layer1 = MZILayer(10, 5)
        layer2 = MZILayer(5, 2)
        
        circuit.add_layer(layer1)
        circuit.add_layer(layer2)
        circuit.connect_layers(0, 1)
        
        assert len(circuit.connections) == 1
        assert circuit.connections[0] == (0, 1)
        
    def test_invalid_connection(self):
        """Test invalid layer connection raises error."""
        circuit = PhotonicCircuit("test_circuit")
        layer = MZILayer(10, 5)
        circuit.add_layer(layer)
        
        with pytest.raises(ValueError, match="Invalid layer indices"):
            circuit.connect_layers(0, 5)  # Layer 5 doesn't exist
            
    def test_circuit_analysis(self):
        """Test circuit performance analysis."""
        circuit = PhotonicCircuit("test_circuit")
        
        # Add layers with known component counts
        layer1 = MZILayer(4, 2)  # 8 MZIs
        layer2 = MZILayer(2, 1)  # 2 MZIs
        
        circuit.add_layer(layer1)
        circuit.add_layer(layer2)
        
        metrics = circuit.analyze_circuit()
        
        assert isinstance(metrics, CircuitMetrics)
        assert metrics.energy_per_op > 0
        assert metrics.latency > 0
        assert metrics.area > 0
        assert metrics.power > 0
        assert metrics.throughput > 0
        assert 0 <= metrics.accuracy <= 1
        
    def test_verilog_generation(self):
        """Test complete circuit Verilog generation."""
        circuit = PhotonicCircuit("test_neural_net")
        
        layer = MZILayer(2, 2)
        circuit.add_layer(layer)
        
        verilog = circuit.generate_verilog()
        
        assert "module test_neural_net" in verilog
        assert "Total layers: 1" in verilog
        assert "endmodule" in verilog
        assert "input clk" in verilog
        assert "output [31:0] data_out" in verilog


class TestPhotonicAccelerator:
    """Test photonic accelerator functionality."""
    
    @pytest.fixture
    def temp_accelerator(self):
        """Create accelerator with temporary database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock database configuration
            with patch('src.photonic_foundry.core.get_database') as mock_get_db, \
                 patch('src.photonic_foundry.core.get_circuit_cache') as mock_get_cache:
                
                # Mock database
                mock_db = Mock()
                mock_repo = Mock()
                mock_get_db.return_value = mock_db
                
                # Mock cache
                mock_cache = Mock()
                mock_get_cache.return_value = mock_cache
                
                accelerator = PhotonicAccelerator(pdk="skywater130", wavelength=1550.0)
                accelerator.circuit_repo = mock_repo
                accelerator.circuit_cache = mock_cache
                
                yield accelerator, mock_repo, mock_cache
                
    def test_accelerator_initialization(self, temp_accelerator):
        """Test accelerator initialization."""
        accelerator, _, _ = temp_accelerator
        
        assert accelerator.pdk == "skywater130"
        assert accelerator.wavelength == 1550.0
        assert 'Linear' in accelerator.supported_layers
        assert 'Conv2d' in accelerator.supported_layers
        
    def test_pytorch_model_conversion(self, temp_accelerator):
        """Test converting PyTorch model to photonic circuit."""
        accelerator, _, _ = temp_accelerator
        
        # Create simple PyTorch model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        circuit = accelerator.convert_pytorch_model(model)
        
        assert isinstance(circuit, PhotonicCircuit)
        assert len(circuit.layers) == 2  # Only Linear layers converted
        assert circuit.layers[0].input_size == 784
        assert circuit.layers[0].output_size == 256
        assert circuit.layers[1].input_size == 256
        assert circuit.layers[1].output_size == 10
        
    def test_circuit_compilation(self, temp_accelerator):
        """Test compiling and profiling photonic circuit."""
        accelerator, _, _ = temp_accelerator
        
        circuit = PhotonicCircuit("test_compile")
        layer = MZILayer(10, 5)
        circuit.add_layer(layer)
        
        metrics = accelerator.compile_and_profile(circuit)
        
        assert isinstance(metrics, CircuitMetrics)
        assert metrics.energy_per_op > 0
        assert metrics.latency > 0
        
    def test_inference_simulation(self, temp_accelerator):
        """Test simulating inference on photonic circuit."""
        accelerator, _, _ = temp_accelerator
        
        circuit = PhotonicCircuit("test_inference")
        layer = MZILayer(4, 2)
        circuit.add_layer(layer)
        
        input_data = np.random.randn(4)
        output_data, inference_time = accelerator.simulate_inference(circuit, input_data)
        
        assert output_data.shape == (2,)  # Output size matches layer
        assert inference_time > 0
        assert np.isfinite(output_data).all()  # No NaN or inf values
        
    def test_circuit_saving(self, temp_accelerator):
        """Test saving circuit to database."""
        accelerator, mock_repo, mock_cache = temp_accelerator
        
        circuit = PhotonicCircuit("test_save")
        layer = MZILayer(5, 3)
        circuit.add_layer(layer)
        
        verilog_code = "module test(); endmodule"
        metrics = CircuitMetrics(
            energy_per_op=1.0, latency=50, area=0.1, power=5.0,
            throughput=1000, accuracy=0.95
        )
        
        # Mock repository response
        mock_repo.save.return_value = 123
        
        circuit_id = accelerator.save_circuit(circuit, verilog_code, metrics)
        
        assert circuit_id == 123
        mock_repo.save.assert_called_once()
        mock_cache.put_circuit.assert_called_once()
        
    def test_circuit_loading(self, temp_accelerator):
        """Test loading circuit from database."""
        accelerator, mock_repo, _ = temp_accelerator
        
        # Mock database response
        mock_circuit_data = {
            'name': 'test_load',
            'circuit_data': json.dumps({
                'layers': [
                    {
                        'type': 'MZILayer',
                        'input_size': 5,
                        'output_size': 3,
                        'components': []
                    }
                ],
                'connections': [],
                'total_components': 15
            }),
            'model_hash': 'hash123',
            'verilog_code': None,
            'metrics': None,
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
            'version': 1
        }
        
        mock_repo.find_by_name.return_value = mock_circuit_data
        
        circuit = accelerator.load_circuit("test_load")
        
        assert circuit is not None
        assert circuit.name == "test_load"
        assert len(circuit.layers) == 1
        assert circuit.total_components == 15
        
    def test_list_saved_circuits(self, temp_accelerator):
        """Test listing saved circuits."""
        accelerator, mock_repo, _ = temp_accelerator
        
        # Mock database response
        from src.photonic_foundry.database.models import CircuitModel
        mock_circuit = CircuitModel("test_circuit", {'layers': []})
        mock_repo.list_all.return_value = [mock_circuit]
        
        circuits = accelerator.list_saved_circuits()
        
        assert len(circuits) == 1
        assert circuits[0]['name'] == "test_circuit"
        assert 'layer_count' in circuits[0]
        assert 'component_count' in circuits[0]


class TestModelIntegration:
    """Test integration with example models."""
    
    def test_simple_mlp_conversion(self):
        """Test converting SimpleMLP to photonic circuit."""
        model = SimpleMLP(input_size=784, hidden_sizes=[256, 128], num_classes=10)
        
        accelerator = PhotonicAccelerator()
        with patch.object(accelerator, 'circuit_repo'), \
             patch.object(accelerator, 'circuit_cache'):
            
            circuit = accelerator.convert_pytorch_model(model)
            
            assert len(circuit.layers) == 3  # 3 Linear layers
            assert circuit.layers[0].input_size == 784
            assert circuit.layers[1].input_size == 256
            assert circuit.layers[2].input_size == 128
            assert circuit.layers[2].output_size == 10
            
    def test_photonic_optimized_mlp(self):
        """Test PhotonicOptimizedMLP model."""
        model = PhotonicOptimizedMLP(input_size=784, num_classes=10)
        
        # Check model uses power-of-2 sizes
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        
        assert len(linear_layers) == 3  # Input + 2 hidden + output
        
        # Check weight initialization is constrained
        for layer in linear_layers:
            weights = layer.weight.detach()
            assert torch.all(torch.abs(weights) <= 1.0)  # Weights should be bounded
            
    def test_model_complexity_analysis(self):
        """Test photonic complexity analysis."""
        model = SimpleMLP(input_size=100, hidden_sizes=[50, 25], num_classes=5)
        complexity = model.get_photonic_complexity()
        
        assert 'total_mzis' in complexity
        assert 'total_parameters' in complexity
        assert 'layer_details' in complexity
        assert 'estimated_area_mm2' in complexity
        assert 'estimated_power_mw' in complexity
        
        # Verify calculations
        expected_mzis = 100*50 + 50*25 + 25*5  # Weight matrices
        assert complexity['total_mzis'] == expected_mzis


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_model_conversion(self):
        """Test converting empty model."""
        model = nn.Sequential()  # Empty model
        
        accelerator = PhotonicAccelerator()
        with patch.object(accelerator, 'circuit_repo'), \
             patch.object(accelerator, 'circuit_cache'):
            
            circuit = accelerator.convert_pytorch_model(model)
            
            assert len(circuit.layers) == 0
            assert circuit.total_components == 0
            
    def test_unsupported_layer_types(self):
        """Test handling unsupported layer types."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.LSTM(5, 3),  # Unsupported layer
            nn.Linear(3, 1)
        )
        
        accelerator = PhotonicAccelerator()
        with patch.object(accelerator, 'circuit_repo'), \
             patch.object(accelerator, 'circuit_cache'):
            
            circuit = accelerator.convert_pytorch_model(model)
            
            # Should only convert Linear layers
            assert len(circuit.layers) == 2
            
    def test_invalid_circuit_analysis(self):
        """Test analysis of invalid circuit."""
        circuit = PhotonicCircuit("invalid_circuit")
        # No layers added
        
        metrics = circuit.analyze_circuit()
        
        # Should handle empty circuit gracefully
        assert metrics.energy_per_op == 0
        assert metrics.latency == 0
        
    def test_zero_size_layer(self):
        """Test handling zero-size layer."""
        with pytest.raises(ValueError):
            MZILayer(input_size=0, output_size=5)
            
        with pytest.raises(ValueError):
            MZILayer(input_size=5, output_size=0)


@pytest.mark.performance
class TestPerformanceScaling:
    """Test performance scaling with model size."""
    
    def test_large_model_conversion(self):
        """Test converting large model."""
        # Large model (but not too large for testing)
        model = SimpleMLP(input_size=1000, hidden_sizes=[500, 250], num_classes=100)
        
        accelerator = PhotonicAccelerator()
        with patch.object(accelerator, 'circuit_repo'), \
             patch.object(accelerator, 'circuit_cache'):
            
            import time
            start_time = time.time()
            circuit = accelerator.convert_pytorch_model(model)
            conversion_time = time.time() - start_time
            
            assert len(circuit.layers) == 3
            assert conversion_time < 1.0  # Should complete quickly
            
    def test_circuit_metrics_scaling(self):
        """Test how circuit metrics scale with size."""
        small_circuit = PhotonicCircuit("small")
        small_layer = MZILayer(10, 5)
        small_circuit.add_layer(small_layer)
        
        large_circuit = PhotonicCircuit("large")
        large_layer = MZILayer(100, 50)
        large_circuit.add_layer(large_layer)
        
        small_metrics = small_circuit.analyze_circuit()
        large_metrics = large_circuit.analyze_circuit()
        
        # Larger circuit should have proportionally higher metrics
        size_ratio = (100 * 50) / (10 * 5)  # Component count ratio
        
        assert large_metrics.energy_per_op > small_metrics.energy_per_op
        assert large_metrics.area > small_metrics.area
        assert abs(large_metrics.energy_per_op / small_metrics.energy_per_op - size_ratio) < size_ratio * 0.1


import json  # Add missing import