"""Shared test configuration and fixtures."""

import pytest
import torch
import numpy as np
import tempfile
import pathlib
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_linear_model():
    """Create a simple linear model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    return torch.randn(32, 10)


@pytest.fixture
def photonic_params():
    """Standard photonic device parameters for testing."""
    return {
        'wavelength': 1550,  # nm
        'pdk': 'skywater130',
        'energy_per_op': 0.5,  # pJ
        'latency': 100  # ps
    }


@pytest.fixture
def sample_cnn_model():
    """Create a simple CNN model for testing."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((4, 4)),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )


@pytest.fixture
def sample_cnn_input():
    """Create sample CNN input data."""
    return torch.randn(8, 1, 28, 28)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def mock_transpiler():
    """Create a mock transpiler for testing."""
    mock = Mock()
    mock.transpile.return_value = "// Mock Verilog code"
    mock.get_power_consumption.return_value = 1.5  # pJ
    mock.get_latency.return_value = 200  # ps
    return mock


@pytest.fixture
def mock_photonic_core():
    """Create a mock photonic core for testing."""
    mock = Mock()
    mock.process_model.return_value = Mock()
    mock.simulate.return_value = {
        'energy_per_op': 0.8,
        'latency': 150,
        'accuracy': 0.95
    }
    return mock


@pytest.fixture
def large_model():
    """Create a large model for performance testing."""
    layers = []
    for i in range(50):
        layers.extend([
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        ])
    layers.append(torch.nn.Linear(512, 10))
    return torch.nn.Sequential(*layers)


@pytest.fixture
def verilog_testbench_template():
    """Template for Verilog testbench generation."""
    return """
module {module_name}_tb;
    // Testbench for {module_name}
    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Test vectors
    reg [{input_width-1}:0] test_input;
    wire [{output_width-1}:0] test_output;
    
    // Instantiate DUT
    {module_name} dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(test_input),
        .output_data(test_output)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        rst_n = 0;
        test_input = 0;
        
        #100 rst_n = 1;
        
        // Add test vectors here
        
        #1000 $finish;
    end
    
endmodule
"""


@pytest.fixture
def performance_metrics():
    """Standard performance metrics for comparison."""
    return {
        'cpu_baseline': {
            'latency_ms': 10.5,
            'energy_mj': 50.0,
            'throughput_ops': 1000
        },
        'gpu_baseline': {
            'latency_ms': 2.1,
            'energy_mj': 120.0,
            'throughput_ops': 5000
        },
        'photonic_target': {
            'latency_ms': 0.1,
            'energy_mj': 0.5,
            'throughput_ops': 10000
        }
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure reproducible test results by resetting random seeds."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def test_config():
    """Standard test configuration."""
    return {
        'debug': True,
        'log_level': 'DEBUG',
        'max_model_size_mb': 100,
        'timeout_seconds': 30,
        'enable_simulation': True,
        'simulation_accuracy': 'medium'
    }