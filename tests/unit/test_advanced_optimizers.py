"""
Unit tests for advanced photonic optimization algorithms.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.photonic_foundry.utils.advanced_optimizers import (
    MZIMeshOptimizer,
    RingResonatorOptimizer,
    PhotonicActivationSynthesizer,
    CircuitLayoutOptimizer,
    QuantumPhotonicOptimizer
)


class TestMZIMeshOptimizer:
    """Test MZI mesh optimization algorithms."""
    
    @pytest.fixture
    def optimizer(self):
        """Create MZI mesh optimizer."""
        return MZIMeshOptimizer(precision=8, loss_budget_db=3.0)
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample weight matrix."""
        return np.random.randn(4, 4)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.precision == 8
        assert optimizer.loss_budget_db == 3.0
        assert optimizer.phase_resolution > 0
    
    def test_optimize_mesh_topology(self, optimizer, sample_weights):
        """Test mesh topology optimization."""
        result = optimizer.optimize_mesh_topology(sample_weights)
        
        assert isinstance(result, dict)
        assert 'topology' in result
        assert 'dimensions' in result
        assert 'phase_configuration' in result
        assert 'singular_values' in result
        assert 'insertion_loss_db' in result
        assert 'power_consumption_mw' in result
        assert 'phase_resolution' in result
        assert 'complexity_score' in result
        
        # Check values
        assert result['topology'] == 'triangular_mesh'
        assert result['dimensions'] == sample_weights.shape
        assert result['insertion_loss_db'] >= 0
        assert result['power_consumption_mw'] >= 0
        assert result['complexity_score'] >= 0
    
    def test_optimize_phase_shifters(self, optimizer, sample_weights):
        """Test phase shifter optimization."""
        U, S, Vt = np.linalg.svd(sample_weights, full_matrices=False)
        
        phase_config = optimizer._optimize_phase_shifters(U, S, Vt)
        
        assert isinstance(phase_config, list)
        assert len(phase_config) == 2  # U and Vt matrices
        
        for phase_list in phase_config:
            assert isinstance(phase_list, list)
            assert all(isinstance(p, float) for p in phase_list)
            # Check quantization
            for phase in phase_list:
                quantized = np.round(phase / optimizer.phase_resolution) * optimizer.phase_resolution
                assert abs(phase - quantized) < 1e-10
    
    def test_calculate_insertion_loss(self, optimizer):
        """Test insertion loss calculation."""
        loss = optimizer._calculate_insertion_loss(4, 4)
        
        assert isinstance(loss, float)
        assert loss >= 0
        # Should be reasonable for a 4x4 mesh
        assert loss < 10  # Less than 10 dB
    
    def test_estimate_power_consumption(self, optimizer):
        """Test power consumption estimation."""
        phase_config = [[0.0, 0.5, 1.0], [0.1, 0.0, 0.8]]
        
        power = optimizer._estimate_power_consumption(3, 3, phase_config)
        
        assert isinstance(power, float)
        assert power >= 0
        # Power should be proportional to active shifters
        assert power > 0  # Should have some active shifters
    
    def test_different_matrix_sizes(self, optimizer):
        """Test optimization with different matrix sizes."""
        sizes = [(2, 2), (8, 4), (16, 16)]
        
        for m, n in sizes:
            weights = np.random.randn(m, n)
            result = optimizer.optimize_mesh_topology(weights)
            
            assert result['dimensions'] == (m, n)
            assert result['complexity_score'] >= 0
    
    def test_zero_weights(self, optimizer):
        """Test optimization with zero weight matrix."""
        zero_weights = np.zeros((3, 3))
        
        result = optimizer.optimize_mesh_topology(zero_weights)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result['power_consumption_mw'] >= 0


class TestRingResonatorOptimizer:
    """Test ring resonator optimization algorithms."""
    
    @pytest.fixture
    def optimizer(self):
        """Create ring resonator optimizer."""
        return RingResonatorOptimizer(q_factor=10000, fsr_ghz=100)
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample weights."""
        return np.array([1.0, -0.5, 0.8, -1.2, 0.3])
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.q_factor == 10000
        assert optimizer.fsr_ghz == 100
        assert optimizer.wavelength_nm == 1550
    
    def test_design_resonator_bank(self, optimizer, sample_weights):
        """Test resonator bank design."""
        result = optimizer.design_resonator_bank(sample_weights)
        
        assert isinstance(result, dict)
        assert 'num_resonators' in result
        assert 'wavelength_channels' in result
        assert 'coupling_coefficients' in result
        assert 'q_factor' in result
        assert 'fsr_ghz' in result
        assert 'thermal_configuration' in result
        assert 'power_consumption_mw' in result
        
        # Check values
        assert result['num_resonators'] == len(sample_weights)
        assert result['q_factor'] == 10000
        assert result['fsr_ghz'] == 100
        assert len(result['wavelength_channels']) == len(sample_weights)
        assert len(result['coupling_coefficients']) == len(sample_weights)
    
    def test_assign_wavelength_channels(self, optimizer, sample_weights):
        """Test wavelength channel assignment."""
        channels = optimizer._assign_wavelength_channels(sample_weights)
        
        assert isinstance(channels, list)
        assert len(channels) == len(sample_weights)
        
        # Check that channels are around 1550nm
        for channel in channels:
            assert 1540 < channel < 1560  # Reasonable range
        
        # Check that channels are different
        assert len(set(channels)) == len(channels)
    
    def test_optimize_coupling_coefficients(self, optimizer, sample_weights):
        """Test coupling coefficient optimization."""
        coeffs = optimizer._optimize_coupling_coefficients(sample_weights)
        
        assert isinstance(coeffs, list)
        assert len(coeffs) == len(sample_weights)
        
        # Check range
        for coeff in coeffs:
            assert 0.01 <= coeff <= 0.3
        
        # Larger weights should have larger coefficients
        max_weight_idx = np.argmax(np.abs(sample_weights))
        assert coeffs[max_weight_idx] >= min(coeffs)
    
    def test_calculate_thermal_tuning(self, optimizer, sample_weights):
        """Test thermal tuning calculation."""
        thermal_config = optimizer._calculate_thermal_tuning(sample_weights)
        
        assert isinstance(thermal_config, dict)
        assert 'tuning_range_nm' in thermal_config
        assert 'thermal_efficiency_nm_per_mw' in thermal_config
        assert 'tuning_powers' in thermal_config
        assert 'max_power_per_ring_mw' in thermal_config
        
        # Check values
        assert thermal_config['thermal_efficiency_nm_per_mw'] == 0.1
        assert len(thermal_config['tuning_powers']) == len(sample_weights)
        assert all(p >= 0 for p in thermal_config['tuning_powers'])
    
    def test_empty_weights(self, optimizer):
        """Test with empty weight array."""
        empty_weights = np.array([])
        
        result = optimizer.design_resonator_bank(empty_weights)
        
        assert result['num_resonators'] == 0
        assert len(result['wavelength_channels']) == 0
        assert len(result['coupling_coefficients']) == 0
    
    def test_single_weight(self, optimizer):
        """Test with single weight."""
        single_weight = np.array([0.5])
        
        result = optimizer.design_resonator_bank(single_weight)
        
        assert result['num_resonators'] == 1
        assert len(result['wavelength_channels']) == 1
        assert len(result['coupling_coefficients']) == 1


class TestPhotonicActivationSynthesizer:
    """Test photonic activation function synthesis."""
    
    @pytest.fixture
    def synthesizer(self):
        """Create activation synthesizer."""
        return PhotonicActivationSynthesizer(precision=8)
    
    def test_synthesizer_initialization(self, synthesizer):
        """Test synthesizer initialization."""
        assert synthesizer.precision == 8
        assert 'relu' in synthesizer.supported_activations
        assert 'sigmoid' in synthesizer.supported_activations
        assert 'tanh' in synthesizer.supported_activations
        assert 'linear' in synthesizer.supported_activations
    
    def test_synthesize_relu(self, synthesizer):
        """Test ReLU synthesis."""
        result = synthesizer.synthesize_activation('relu', dynamic_range=2.0)
        
        assert result['type'] == 'relu'
        assert result['implementation'] == 'amplitude_clipping'
        assert 'components' in result
        assert result['power_consumption_mw'] > 0
        assert result['bandwidth_ghz'] > 0
        assert result['linearity_error_percent'] >= 0
        
        # Check components
        components = result['components']
        assert len(components) == 2
        assert any(c['type'] == 'photodetector' for c in components)
        assert any(c['type'] == 'amplitude_limiter' for c in components)
    
    def test_synthesize_sigmoid(self, synthesizer):
        """Test sigmoid synthesis."""
        result = synthesizer.synthesize_activation('sigmoid', dynamic_range=2.0)
        
        assert result['type'] == 'sigmoid'
        assert result['implementation'] == 'mzi_nonlinearity'
        assert 'components' in result
        assert result['power_consumption_mw'] > 0
        
        # Sigmoid should be more power hungry than ReLU
        relu_result = synthesizer.synthesize_activation('relu')
        assert result['power_consumption_mw'] > relu_result['power_consumption_mw']
    
    def test_synthesize_tanh(self, synthesizer):
        """Test tanh synthesis."""
        result = synthesizer.synthesize_activation('tanh', dynamic_range=2.0)
        
        assert result['type'] == 'tanh'
        assert result['implementation'] == 'dual_rail_encoding'
        assert 'components' in result
        
        # Tanh should be most power hungry
        relu_result = synthesizer.synthesize_activation('relu')
        sigmoid_result = synthesizer.synthesize_activation('sigmoid')
        assert result['power_consumption_mw'] > relu_result['power_consumption_mw']
        assert result['power_consumption_mw'] > sigmoid_result['power_consumption_mw']
    
    def test_synthesize_linear(self, synthesizer):
        """Test linear activation synthesis."""
        result = synthesizer.synthesize_activation('linear', dynamic_range=2.0)
        
        assert result['type'] == 'linear'
        assert result['implementation'] == 'direct_connection'
        assert 'components' in result
        
        # Linear should be least power hungry
        relu_result = synthesizer.synthesize_activation('relu')
        assert result['power_consumption_mw'] < relu_result['power_consumption_mw']
    
    def test_unsupported_activation(self, synthesizer):
        """Test unsupported activation function."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            synthesizer.synthesize_activation('gelu')
    
    def test_different_dynamic_ranges(self, synthesizer):
        """Test with different dynamic ranges."""
        ranges = [1.0, 2.0, 5.0]
        
        for dynamic_range in ranges:
            result = synthesizer.synthesize_activation('relu', dynamic_range)
            
            # Find amplitude limiter component
            limiter = next(c for c in result['components'] if c['type'] == 'amplitude_limiter')
            assert limiter['dynamic_range'] == dynamic_range


class TestCircuitLayoutOptimizer:
    """Test circuit layout optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create layout optimizer."""
        return CircuitLayoutOptimizer(chip_size_mm=(10.0, 10.0))
    
    @pytest.fixture
    def sample_components(self):
        """Create sample components."""
        return [
            {'type': 'mzi', 'id': 'mzi_1'},
            {'type': 'ring', 'id': 'ring_1'},
            {'type': 'waveguide', 'length_um': 200},
            {'type': 'mzi', 'id': 'mzi_2'},
            {'type': 'ring', 'id': 'ring_2'}
        ]
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.chip_size_mm == (10.0, 10.0)
        assert optimizer.component_spacing_um == 50
    
    def test_optimize_layout(self, optimizer, sample_components):
        """Test layout optimization."""
        result = optimizer.optimize_layout(sample_components)
        
        assert isinstance(result, dict)
        assert 'component_placement' in result
        assert 'total_area_mm2' in result
        assert 'chip_utilization' in result
        assert 'thermal_zones' in result
        assert 'routing_length_mm' in result
        assert 'layout_efficiency' in result
        
        # Check values
        assert result['total_area_mm2'] >= 0
        assert 0 <= result['chip_utilization'] <= 1
        assert result['routing_length_mm'] >= 0
        assert 0 <= result['layout_efficiency'] <= 1
        assert isinstance(result['thermal_zones'], list)
    
    def test_estimate_component_areas(self, optimizer, sample_components):
        """Test component area estimation."""
        areas = optimizer._estimate_component_areas(sample_components)
        
        assert isinstance(areas, dict)
        assert len(areas) == len(sample_components)
        
        # Check that different component types have different areas
        mzi_areas = [areas[k] for k, comp in zip(areas.keys(), sample_components) if comp['type'] == 'mzi']
        ring_areas = [areas[k] for k, comp in zip(areas.keys(), sample_components) if comp['type'] == 'ring']
        
        if mzi_areas and ring_areas:
            assert mzi_areas[0] > ring_areas[0]  # MZI should be larger than ring
    
    def test_generate_initial_placement(self, optimizer, sample_components):
        """Test initial placement generation."""
        areas = optimizer._estimate_component_areas(sample_components)
        placement = optimizer._generate_initial_placement(sample_components, areas)
        
        assert isinstance(placement, dict)
        assert len(placement) == len(areas)
        
        # Check that all positions are within chip boundaries
        for pos in placement.values():
            x, y = pos
            assert 0 <= x <= optimizer.chip_size_mm[0]
            assert 0 <= y <= optimizer.chip_size_mm[1]
    
    def test_calculate_thermal_zones(self, optimizer, sample_components):
        """Test thermal zone calculation."""
        areas = optimizer._estimate_component_areas(sample_components)
        placement = optimizer._generate_initial_placement(sample_components, areas)
        thermal_zones = optimizer._calculate_thermal_zones(placement)
        
        assert isinstance(thermal_zones, list)
        assert len(thermal_zones) >= 1
        
        for zone in thermal_zones:
            assert 'zone_id' in zone
            assert 'components' in zone
            assert 'center' in zone
            assert 'temperature_sensors' in zone
            assert 'thermal_actuators' in zone
    
    def test_empty_components(self, optimizer):
        """Test with empty component list."""
        result = optimizer.optimize_layout([])
        
        assert result['total_area_mm2'] == 0
        assert result['chip_utilization'] == 0
        assert result['routing_length_mm'] == 0
        assert len(result['thermal_zones']) <= 1
    
    def test_single_component(self, optimizer):
        """Test with single component."""
        component = [{'type': 'mzi', 'id': 'mzi_1'}]
        result = optimizer.optimize_layout(component)
        
        assert result['total_area_mm2'] > 0
        assert result['chip_utilization'] > 0
        assert len(result['thermal_zones']) == 1


class TestQuantumPhotonicOptimizer:
    """Test quantum photonic optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create quantum optimizer."""
        return QuantumPhotonicOptimizer()
    
    @pytest.fixture
    def sample_circuit_spec(self):
        """Create sample quantum circuit specification."""
        return {
            'qubits': 4,
            'gates': 10,
            'entanglement_depth': 2
        }
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert 'fock' in optimizer.quantum_states
        assert 'coherent' in optimizer.quantum_states
        assert 'squeezed' in optimizer.quantum_states
        assert len(optimizer.entanglement_protocols) == 2
    
    def test_optimize_quantum_circuit(self, optimizer, sample_circuit_spec):
        """Test quantum circuit optimization."""
        result = optimizer.optimize_quantum_circuit(sample_circuit_spec)
        
        assert isinstance(result, dict)
        assert 'quantum_efficiency' in result
        assert 'state_preparation' in result
        assert 'gate_optimization' in result
        assert 'measurement_strategy' in result
        assert 'decoherence_time_us' in result
        assert 'fidelity' in result
        assert 'error_correction_overhead' in result
        
        # Check reasonable values
        assert 0 <= result['quantum_efficiency'] <= 1
        assert 0 <= result['fidelity'] <= 1
        assert result['decoherence_time_us'] > 0
        assert result['error_correction_overhead'] >= 0
    
    def test_optimize_state_preparation(self, optimizer):
        """Test state preparation optimization."""
        result = optimizer._optimize_state_preparation(4)
        
        assert isinstance(result, dict)
        assert 'method' in result
        assert 'preparation_time_us' in result
        assert 'fidelity' in result
        assert 'resource_overhead' in result
        
        assert result['preparation_time_us'] > 0
        assert 0 <= result['fidelity'] <= 1
        assert result['resource_overhead'] >= 0
    
    def test_optimize_quantum_gates(self, optimizer):
        """Test quantum gate optimization."""
        result = optimizer._optimize_quantum_gates(10)
        
        assert isinstance(result, dict)
        assert 'gate_implementation' in result
        assert 'gate_time_ns' in result
        assert 'gate_fidelity' in result
        assert 'crosstalk_suppression_db' in result
        
        assert result['gate_time_ns'] > 0
        assert 0 <= result['gate_fidelity'] <= 1
        assert result['crosstalk_suppression_db'] > 0
    
    def test_optimize_measurements(self, optimizer):
        """Test measurement optimization."""
        result = optimizer._optimize_measurements(4)
        
        assert isinstance(result, dict)
        assert 'measurement_type' in result
        assert 'measurement_time_ns' in result
        assert 'detection_efficiency' in result
        assert 'dark_count_rate_hz' in result
        
        assert result['measurement_time_ns'] > 0
        assert 0 <= result['detection_efficiency'] <= 1
        assert result['dark_count_rate_hz'] >= 0
    
    def test_different_qubit_counts(self, optimizer):
        """Test with different qubit counts."""
        qubit_counts = [1, 2, 4, 8]
        
        for qubits in qubit_counts:
            circuit_spec = {'qubits': qubits, 'gates': qubits * 2, 'entanglement_depth': 1}
            result = optimizer.optimize_quantum_circuit(circuit_spec)
            
            assert result['quantum_efficiency'] > 0
            assert result['fidelity'] > 0
    
    def test_edge_cases(self, optimizer):
        """Test edge cases."""
        # Zero qubits
        zero_spec = {'qubits': 0, 'gates': 0, 'entanglement_depth': 0}
        result = optimizer.optimize_quantum_circuit(zero_spec)
        assert isinstance(result, dict)
        
        # Large system
        large_spec = {'qubits': 100, 'gates': 1000, 'entanglement_depth': 10}
        result = optimizer.optimize_quantum_circuit(large_spec)
        assert isinstance(result, dict)


class TestIntegrationOptimization:
    """Test integration between different optimizers."""
    
    def test_mzi_and_ring_optimization_compatibility(self):
        """Test that MZI and ring optimizers work with same weights."""
        weights = np.random.randn(4, 4)
        
        mzi_optimizer = MZIMeshOptimizer()
        ring_optimizer = RingResonatorOptimizer()
        
        mzi_result = mzi_optimizer.optimize_mesh_topology(weights)
        ring_result = ring_optimizer.design_resonator_bank(weights)
        
        # Both should handle the same weight matrix
        assert mzi_result['dimensions'] == weights.shape
        assert ring_result['num_resonators'] == weights.size
    
    def test_activation_and_layout_integration(self):
        """Test integration between activation synthesis and layout optimization."""
        synthesizer = PhotonicActivationSynthesizer()
        layout_optimizer = CircuitLayoutOptimizer()
        
        # Create activation components
        activations = ['relu', 'sigmoid', 'tanh']
        components = []
        
        for i, activation in enumerate(activations):
            result = synthesizer.synthesize_activation(activation)
            for j, component in enumerate(result['components']):
                components.append({
                    'type': component['type'],
                    'activation_id': f"{activation}_{i}_{j}"
                })
        
        # Optimize layout
        layout_result = layout_optimizer.optimize_layout(components)
        
        assert layout_result['total_area_mm2'] > 0
        assert len(layout_result['component_placement']) == len(components)
    
    def test_optimization_consistency(self):
        """Test that optimization results are consistent across runs."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        weights1 = np.random.randn(3, 3)
        np.random.seed(42)
        weights2 = np.random.randn(3, 3)
        
        optimizer = MZIMeshOptimizer(precision=8)
        
        result1 = optimizer.optimize_mesh_topology(weights1)
        result2 = optimizer.optimize_mesh_topology(weights2)
        
        # Results should be identical for identical inputs
        assert result1['dimensions'] == result2['dimensions']
        assert abs(result1['complexity_score'] - result2['complexity_score']) < 1e-10