"""
Tests for PhotonicLayer validation and basic functionality.
"""

import pytest
import numpy as np
from src.photonic_foundry.core import PhotonicLayer, MZILayer, PhotonicComponent


class TestPhotonicLayerValidation:
    """Test PhotonicLayer input validation and edge cases."""

    def test_photonic_layer_init_valid(self):
        """Test valid PhotonicLayer initialization."""
        layer = PhotonicLayer(input_size=10, output_size=5)
        assert layer.input_size == 10
        assert layer.output_size == 5
        assert layer.components == []

    def test_photonic_layer_init_zero_input(self):
        """Test PhotonicLayer with zero input size."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            PhotonicLayer(input_size=0, output_size=5)

    def test_photonic_layer_init_zero_output(self):
        """Test PhotonicLayer with zero output size."""
        with pytest.raises(ValueError, match="output_size must be positive"):
            PhotonicLayer(input_size=5, output_size=0)

    def test_photonic_layer_init_negative_input(self):
        """Test PhotonicLayer with negative input size."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            PhotonicLayer(input_size=-5, output_size=5)

    def test_photonic_layer_init_negative_output(self):
        """Test PhotonicLayer with negative output size."""
        with pytest.raises(ValueError, match="output_size must be positive"):
            PhotonicLayer(input_size=5, output_size=-5)

    def test_add_component(self):
        """Test adding components to a layer."""
        layer = PhotonicLayer(input_size=4, output_size=2)
        params = {'phase_shifter_bits': 8, 'insertion_loss': 0.1}
        
        layer.add_component(PhotonicComponent.MZI, params)
        
        assert len(layer.components) == 1
        assert layer.components[0]['type'] == PhotonicComponent.MZI
        assert layer.components[0]['params'] == params

    def test_mzi_layer_creation(self):
        """Test MZILayer creation and component generation."""
        layer = MZILayer(input_size=2, output_size=2)
        
        # Should create MZI components for 2x2 mesh
        expected_components = 2 * 2  # output_size * input_size
        assert len(layer.components) == expected_components
        
        # All components should be MZI type
        for component in layer.components:
            assert component['type'] == PhotonicComponent.MZI

    def test_mzi_layer_weights(self):
        """Test MZILayer weight initialization."""
        layer = MZILayer(input_size=3, output_size=4)
        
        assert layer.weights.shape == (4, 3)  # (output_size, input_size)
        assert np.all(np.isfinite(layer.weights))  # Check for valid values

    def test_mzi_layer_verilog_generation(self):
        """Test Verilog generation for MZI layer."""
        layer = MZILayer(input_size=2, output_size=2)
        verilog_code = layer.generate_verilog()
        
        # Check basic structure
        assert "module mzi_layer_2x2" in verilog_code
        assert "input" in verilog_code
        assert "output" in verilog_code
        assert "endmodule" in verilog_code


class TestMZILayerEdgeCases:
    """Test edge cases for MZI layers."""

    def test_mzi_layer_single_input_output(self):
        """Test MZI layer with 1x1 configuration."""
        layer = MZILayer(input_size=1, output_size=1)
        
        assert layer.input_size == 1
        assert layer.output_size == 1
        assert len(layer.components) == 1

    def test_mzi_layer_different_precisions(self):
        """Test MZI layer with different precision settings."""
        for precision in [4, 8, 12, 16]:
            layer = MZILayer(input_size=2, output_size=2, precision=precision)
            assert layer.precision == precision
            
            # Check all components have correct precision
            for component in layer.components:
                assert component['params']['phase_shifter_bits'] == precision

    def test_mzi_layer_large_dimensions(self):
        """Test MZI layer with larger dimensions."""
        layer = MZILayer(input_size=10, output_size=8)
        
        expected_components = 10 * 8
        assert len(layer.components) == expected_components
        assert layer.weights.shape == (8, 10)