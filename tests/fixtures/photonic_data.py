"""
Photonic-specific test fixtures and data for comprehensive testing.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import pytest


@pytest.fixture
def sample_wavelengths() -> np.ndarray:
    """Standard telecommunications wavelengths for testing."""
    return np.linspace(1530, 1570, 40)  # C-band wavelengths in nm


@pytest.fixture
def photonic_parameters() -> Dict[str, Any]:
    """Common photonic system parameters."""
    return {
        'wavelength': 1550,  # nm
        'power_budget': 10,  # dBm
        'temperature': 25,   # Celsius
        'pdk': 'skywater130',
        'process_variation': 0.05,  # 5% variation
        'coupling_loss': 0.5,  # dB per coupler
        'propagation_loss': 0.1,  # dB/cm
        'detector_responsivity': 0.8,  # A/W
        'modulator_bandwidth': 40,  # GHz
    }


@pytest.fixture
def simple_linear_model() -> nn.Module:
    """Simple linear model for basic transpilation testing."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )


@pytest.fixture
def conv_model() -> nn.Module:
    """Convolutional model for advanced transpilation testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10)
    )


@pytest.fixture
def attention_model() -> nn.Module:
    """Attention-based model for complex transpilation testing."""
    class SimpleAttention(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.q_linear = nn.Linear(dim, dim)
            self.k_linear = nn.Linear(dim, dim)
            self.v_linear = nn.Linear(dim, dim)
            self.out_linear = nn.Linear(dim, dim)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self.q_linear(x)
            k = self.k_linear(x)
            v = self.v_linear(x)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dim)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            return self.out_linear(out)
    
    return SimpleAttention(64)


@pytest.fixture
def sample_input_data() -> Dict[str, torch.Tensor]:
    """Sample input tensors for different model types."""
    return {
        'linear': torch.randn(32, 4),
        'conv': torch.randn(8, 3, 32, 32),
        'attention': torch.randn(16, 10, 64),
        'sequence': torch.randn(8, 100, 512),
    }


@pytest.fixture
def photonic_mac_parameters() -> Dict[str, Any]:
    """Parameters for photonic MAC unit testing."""
    return {
        'num_wavelengths': 8,
        'mzi_length': 100,  # μm
        'coupling_ratio': 0.5,
        'phase_shifter_efficiency': 0.8,  # π/V
        'thermal_coefficient': 1e-4,  # per Kelvin
        'shot_noise_limit': True,
        'quantum_efficiency': 0.9,
        'dark_current': 1e-9,  # A
    }


@pytest.fixture
def performance_targets() -> Dict[str, float]:
    """Performance targets for validation testing."""
    return {
        'energy_per_op': 1.0,  # pJ/Op
        'latency_target': 1e-6,  # 1 microsecond
        'accuracy_tolerance': 0.01,  # 1% degradation
        'throughput_target': 1e12,  # 1 TOP/s
        'area_efficiency': 1e-3,  # TOP/s/mm²
        'power_efficiency': 1e12,  # TOP/s/W
    }


@pytest.fixture
def noise_models() -> Dict[str, Any]:
    """Noise models for realistic simulation testing."""
    return {
        'thermal_noise': {
            'temperature': 300,  # K
            'bandwidth': 1e9,    # Hz
        },
        'shot_noise': {
            'current': 1e-3,     # A
            'bandwidth': 1e9,    # Hz
        },
        'phase_noise': {
            'linewidth': 1e6,    # Hz
            'power': -100,       # dBm
        },
        'amplitude_noise': {
            'rms': 0.01,         # relative
            'correlation_time': 1e-6,  # s
        }
    }


@pytest.fixture
def test_circuits() -> Dict[str, str]:
    """Sample Verilog circuits for validation."""
    return {
        'simple_mac': '''
module photonic_mac #(
    parameter WIDTH = 8,
    parameter NUM_INPUTS = 4
)(
    input wire clk,
    input wire reset,
    input wire [WIDTH-1:0] weights [NUM_INPUTS-1:0],
    input wire [WIDTH-1:0] inputs [NUM_INPUTS-1:0],
    output reg [WIDTH+$clog2(NUM_INPUTS)-1:0] result
);
    // Simple MAC implementation
    always @(posedge clk) begin
        if (reset) begin
            result <= 0;
        end else begin
            result <= weights[0] * inputs[0] + 
                     weights[1] * inputs[1] + 
                     weights[2] * inputs[2] + 
                     weights[3] * inputs[3];
        end
    end
endmodule
        ''',
        
        'mzi_array': '''
module mzi_array #(
    parameter NUM_STAGES = 4,
    parameter PRECISION = 8
)(
    input wire [PRECISION-1:0] phase_controls [NUM_STAGES-1:0],
    input wire optical_input,
    output wire optical_output_0,
    output wire optical_output_1
);
    // Mach-Zehnder Interferometer array
    // Implementation details would go here
endmodule
        '''
    }


@pytest.fixture
def benchmark_models() -> Dict[str, nn.Module]:
    """Standard benchmark models for performance testing."""
    models = {}
    
    # ResNet-like model
    class MiniResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(32 * 4 * 4, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    models['mini_resnet'] = MiniResNet()
    
    # Transformer-like model
    class MiniTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.pos_encoding = nn.Parameter(torch.randn(100, 64))
            self.attention = nn.MultiheadAttention(64, 8, batch_first=True)
            self.fc = nn.Linear(64, 10)
            
        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:x.size(1)]
            x, _ = self.attention(x, x, x)
            return self.fc(x.mean(dim=1))
    
    models['mini_transformer'] = MiniTransformer()
    
    return models


@pytest.fixture
def error_injection_scenarios() -> List[Dict[str, Any]]:
    """Error scenarios for robustness testing."""
    return [
        {
            'name': 'temperature_drift',
            'parameters': {'temperature_change': 10},  # ±10°C
            'expected_impact': 'phase_shift'
        },
        {
            'name': 'power_fluctuation',
            'parameters': {'power_variation': 0.1},  # ±10%
            'expected_impact': 'amplitude_change'
        },
        {
            'name': 'wavelength_drift',
            'parameters': {'wavelength_shift': 0.1},  # ±0.1 nm
            'expected_impact': 'spectral_response'
        },
        {
            'name': 'fabrication_error',
            'parameters': {'dimension_error': 0.05},  # ±5%
            'expected_impact': 'coupling_efficiency'
        }
    ]


class PhotonicTestHelper:
    """Helper class for photonic-specific test utilities."""
    
    @staticmethod
    def generate_random_weights(shape: Tuple[int, ...], 
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generate random weights with appropriate scaling for photonic systems."""
        weights = torch.randn(shape, dtype=dtype)
        # Scale weights to typical photonic range
        return weights * 0.1  # Keep weights small for realistic optical powers
    
    @staticmethod
    def calculate_optical_loss(distance: float, loss_per_cm: float = 0.1) -> float:
        """Calculate optical propagation loss."""
        return distance * loss_per_cm  # dB
    
    @staticmethod
    def thermal_phase_shift(temperature_change: float, 
                          thermo_optic_coeff: float = 1e-4) -> float:
        """Calculate thermal-induced phase shift."""
        return 2 * np.pi * thermo_optic_coeff * temperature_change
    
    @staticmethod
    def validate_energy_efficiency(energy_per_op: float, 
                                 target: float = 1.0) -> bool:
        """Validate energy efficiency meets targets."""
        return energy_per_op <= target
    
    @staticmethod
    def compare_accuracy(photonic_output: torch.Tensor, 
                        electronic_output: torch.Tensor,
                        tolerance: float = 0.01) -> bool:
        """Compare photonic vs electronic implementation accuracy."""
        relative_error = torch.abs(photonic_output - electronic_output) / torch.abs(electronic_output)
        return torch.all(relative_error <= tolerance)


@pytest.fixture
def photonic_helper() -> PhotonicTestHelper:
    """Photonic test helper instance."""
    return PhotonicTestHelper()


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[1, 4, 8, 16])
def wavelength_count(request) -> int:
    """Parametrized number of wavelengths for testing."""
    return request.param


@pytest.fixture(params=[8, 16, 32])
def precision_bits(request) -> int:
    """Parametrized precision for numerical testing."""
    return request.param


@pytest.fixture(params=['skywater130', 'aim_pdk', 'generic'])
def pdk_variant(request) -> str:
    """Parametrized PDK for compatibility testing."""
    return request.param