"""
Tests for CircuitMetrics and PhotonicCircuit analysis functionality.
"""

import pytest
from src.photonic_foundry.core import CircuitMetrics, PhotonicCircuit, MZILayer


class TestCircuitMetrics:
    """Test CircuitMetrics data class."""

    def test_circuit_metrics_creation(self):
        """Test CircuitMetrics initialization."""
        metrics = CircuitMetrics(
            energy_per_op=1.5,
            latency=100.0,
            area=0.5,
            power=10.0,
            throughput=1000.0,
            accuracy=0.95
        )
        
        assert metrics.energy_per_op == 1.5
        assert metrics.latency == 100.0
        assert metrics.area == 0.5
        assert metrics.power == 10.0
        assert metrics.throughput == 1000.0
        assert metrics.accuracy == 0.95

    def test_circuit_metrics_to_dict(self):
        """Test CircuitMetrics to_dict conversion."""
        metrics = CircuitMetrics(
            energy_per_op=2.0,
            latency=150.0,
            area=1.0,
            power=20.0,
            throughput=500.0,
            accuracy=0.90
        )
        
        metrics_dict = metrics.to_dict()
        expected = {
            'energy_per_op': 2.0,
            'latency': 150.0,
            'area': 1.0,
            'power': 20.0,
            'throughput': 500.0,
            'accuracy': 0.90
        }
        
        assert metrics_dict == expected


class TestPhotonicCircuitAnalysis:
    """Test PhotonicCircuit analysis functionality."""

    def test_empty_circuit_analysis(self):
        """Test analysis of empty circuit."""
        circuit = PhotonicCircuit("empty_test")
        metrics = circuit.analyze_circuit()
        
        # Empty circuit should have minimal metrics
        assert metrics.energy_per_op == 0.0  # No MZIs
        assert metrics.latency == 0.0  # No layers
        assert metrics.area == 0.0  # No MZIs
        assert metrics.power == 0.0  # No MZIs
        assert metrics.throughput == 1e12  # 1e12 / 1.0 (fallback)
        assert metrics.accuracy == 0.98

    def test_single_layer_circuit_analysis(self):
        """Test analysis of circuit with single layer."""
        circuit = PhotonicCircuit("single_layer_test")
        layer = MZILayer(input_size=4, output_size=2)  # 4*2 = 8 MZIs
        circuit.add_layer(layer)
        
        metrics = circuit.analyze_circuit()
        
        # Should have metrics for 8 MZIs and 1 layer
        expected_energy = 0.5 * 8  # 0.5 pJ per MZI * 8 MZIs
        expected_latency = 50 * 1  # 50 ps per layer * 1 layer
        expected_area = 0.001 * 8  # 0.001 mm² per MZI * 8 MZIs
        expected_power = expected_energy * 1e6  # Assuming 1 GHz
        expected_throughput = 1e12 / expected_latency
        
        assert metrics.energy_per_op == expected_energy
        assert metrics.latency == expected_latency
        assert metrics.area == expected_area
        assert metrics.power == expected_power
        assert metrics.throughput == expected_throughput
        assert metrics.accuracy == 0.98

    def test_multi_layer_circuit_analysis(self):
        """Test analysis of circuit with multiple layers."""
        circuit = PhotonicCircuit("multi_layer_test")
        
        # Add multiple layers of different sizes
        layer1 = MZILayer(input_size=8, output_size=4)  # 32 MZIs
        layer2 = MZILayer(input_size=4, output_size=2)  # 8 MZIs
        layer3 = MZILayer(input_size=2, output_size=1)  # 2 MZIs
        
        circuit.add_layer(layer1)
        circuit.add_layer(layer2)
        circuit.add_layer(layer3)
        
        metrics = circuit.analyze_circuit()
        
        # Should have metrics for 42 total MZIs and 3 layers
        total_mzis = 32 + 8 + 2
        expected_energy = 0.5 * total_mzis
        expected_latency = 50 * 3  # 3 layers
        expected_area = 0.001 * total_mzis
        expected_power = expected_energy * 1e6
        expected_throughput = 1e12 / expected_latency
        
        assert metrics.energy_per_op == expected_energy
        assert metrics.latency == expected_latency
        assert metrics.area == expected_area
        assert metrics.power == expected_power
        assert metrics.throughput == expected_throughput

    def test_circuit_verilog_generation(self):
        """Test Verilog generation for complete circuit."""
        circuit = PhotonicCircuit("verilog_test")
        layer1 = MZILayer(input_size=2, output_size=2)
        layer2 = MZILayer(input_size=2, output_size=1)
        
        circuit.add_layer(layer1)
        circuit.add_layer(layer2)
        
        verilog_code = circuit.generate_verilog()
        
        # Should contain both layer modules
        assert "mzi_layer_2x2" in verilog_code
        assert "mzi_layer_2x1" in verilog_code
        assert "module verilog_test" in verilog_code
        assert "endmodule" in verilog_code

    def test_circuit_layer_management(self):
        """Test adding and managing circuit layers."""
        circuit = PhotonicCircuit("layer_management_test")
        
        # Initially empty
        assert len(circuit.layers) == 0
        
        # Add layers
        layer1 = MZILayer(input_size=4, output_size=3)
        layer2 = MZILayer(input_size=3, output_size=2)
        
        circuit.add_layer(layer1)
        assert len(circuit.layers) == 1
        assert circuit.layers[0] == layer1
        
        circuit.add_layer(layer2)
        assert len(circuit.layers) == 2
        assert circuit.layers[1] == layer2