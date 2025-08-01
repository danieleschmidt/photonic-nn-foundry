"""Test model fixtures for photonic-nn-foundry tests."""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import pytest


@pytest.fixture
def simple_linear_model() -> nn.Module:
    """Simple linear model for basic testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def multilayer_perceptron() -> nn.Module:
    """Multi-layer perceptron for testing."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


@pytest.fixture
def convolutional_model() -> nn.Module:
    """Simple CNN model for testing."""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


@pytest.fixture
def residual_block() -> nn.Module:
    """ResNet-style residual block for testing."""
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            
            # Skip connection
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, 1)
            else:
                self.skip = nn.Identity()
        
        def forward(self, x):
            residual = self.skip(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + residual)
    
    return ResidualBlock(32, 64)


@pytest.fixture
def attention_model() -> nn.Module:
    """Simple attention mechanism for testing."""
    class SimpleAttention(nn.Module):
        def __init__(self, d_model: int = 512, n_heads: int = 8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            batch_size, seq_len, d_model = x.size()
            
            Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # Simplified attention (missing softmax for testing)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn = torch.matmul(attn, V)
            
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            return self.w_o(attn)
    
    return SimpleAttention()


@pytest.fixture
def lstm_model() -> nn.Module:
    """LSTM model for sequence testing."""
    class LSTMModel(nn.Module):
        def __init__(self, input_size: int = 100, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 10):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    return LSTMModel()


@pytest.fixture
def test_models() -> Dict[str, nn.Module]:
    """Dictionary of all test models."""
    return {
        'linear': nn.Linear(10, 5),
        'mlp': nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ),
        'conv': nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 10)
        )
    }


@pytest.fixture
def sample_inputs() -> Dict[str, torch.Tensor]:
    """Sample input tensors for different model types."""
    return {
        'linear': torch.randn(1, 10),
        'mlp': torch.randn(1, 784),
        'conv': torch.randn(1, 1, 28, 28),
        'sequence': torch.randn(1, 10, 100),  # batch, seq_len, features
        'attention': torch.randn(1, 50, 512),  # batch, seq_len, d_model
        'image_batch': torch.randn(8, 3, 224, 224),  # batch of images
    }


@pytest.fixture
def model_configurations() -> List[Dict[str, Any]]:
    """Different model configurations for parametrized testing."""
    return [
        {
            'name': 'small_linear',
            'model': nn.Linear(5, 3),
            'input_shape': (1, 5),
            'expected_output_shape': (1, 3),
        },
        {
            'name': 'medium_mlp',
            'model': nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            ),
            'input_shape': (1, 100),
            'expected_output_shape': (1, 5),
        },
        {
            'name': 'simple_conv',
            'model': nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 2)
            ),
            'input_shape': (1, 1, 32, 32),
            'expected_output_shape': (1, 2),
        }
    ]


@pytest.fixture
def photonic_pdk_configs() -> Dict[str, Dict[str, Any]]:
    """Different PDK configurations for testing."""
    return {
        'skywater130': {
            'wavelength': 1550,
            'temperature': 25,
            'process_corner': 'tt',
            'metal_layers': 5,
            'min_feature_size': 130,  # nm
        },
        'globalfoundries45': {
            'wavelength': 1310,
            'temperature': 25,
            'process_corner': 'tt',
            'metal_layers': 7,
            'min_feature_size': 45,  # nm
        },
        'asap7': {
            'wavelength': 1550,
            'temperature': 25,
            'process_corner': 'tt',
            'metal_layers': 9,
            'min_feature_size': 7,  # nm
        }
    }


@pytest.fixture
def performance_test_cases() -> List[Dict[str, Any]]:
    """Test cases for performance benchmarking."""
    return [
        {
            'name': 'small_model_fast',
            'model_size': 'small',
            'complexity': 'low',
            'expected_time_ms': 10,
            'expected_memory_mb': 50,
        },
        {
            'name': 'medium_model_normal',
            'model_size': 'medium',
            'complexity': 'medium',
            'expected_time_ms': 100,
            'expected_memory_mb': 200,
        },
        {
            'name': 'large_model_slow',
            'model_size': 'large',
            'complexity': 'high',
            'expected_time_ms': 1000,
            'expected_memory_mb': 500,
        }
    ]


@pytest.fixture
def verilog_test_patterns() -> Dict[str, str]:
    """Expected Verilog patterns for validation."""
    return {
        'module_declaration': r'module\s+\w+\s*\(',
        'input_port': r'input\s+.*?\s+\w+',
        'output_port': r'output\s+.*?\s+\w+',
        'wire_declaration': r'wire\s+.*?\s+\w+',
        'always_block': r'always\s*@',
        'endmodule': r'endmodule',
        'photonic_mac': r'photonic_mac\s+\w+',
        'mzi_array': r'mzi_array\s+\w+',
        'photodetector': r'photodetector\s+\w+',
    }


@pytest.fixture(scope="session")
def temp_workspace(tmp_path_factory):
    """Create a temporary workspace for test files."""
    workspace = tmp_path_factory.mktemp("photonic_foundry_tests")
    
    # Create subdirectories
    (workspace / "models").mkdir()
    (workspace / "verilog").mkdir()
    (workspace / "results").mkdir()
    (workspace / "cache").mkdir()
    
    return workspace