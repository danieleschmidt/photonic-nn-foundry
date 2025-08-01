"""Test data fixtures for photonic-nn-foundry tests."""

import numpy as np
import torch
from typing import Tuple, List, Dict, Any
import pytest
from pathlib import Path


@pytest.fixture
def sample_verilog_code() -> str:
    """Sample Verilog code for testing."""
    return """
module photonic_mac #(
    parameter DATA_WIDTH = 8,
    parameter NUM_INPUTS = 16
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] data_in [NUM_INPUTS-1:0],
    input [DATA_WIDTH-1:0] weights [NUM_INPUTS-1:0],
    output reg [DATA_WIDTH+4-1:0] result
);

    // Photonic MAC implementation
    wire [NUM_INPUTS-1:0] mzi_outputs;
    wire [DATA_WIDTH+4-1:0] accumulator;
    
    genvar i;
    generate
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin : mzi_gen
            mzi_cell mzi_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(data_in[i]),
                .weight(weights[i]),
                .output_optical(mzi_outputs[i])
            );
        end
    endgenerate
    
    photodetector_array pd_array (
        .clk(clk),
        .rst_n(rst_n),
        .optical_inputs(mzi_outputs),
        .electrical_output(accumulator)
    );
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            result <= 0;
        else
            result <= accumulator;
    end

endmodule

module mzi_cell (
    input clk,
    input rst_n,
    input [7:0] data_in,
    input [7:0] weight,
    output reg output_optical
);
    // Simplified MZI implementation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            output_optical <= 0;
        else
            output_optical <= data_in * weight;
    end
endmodule

module photodetector_array #(
    parameter NUM_INPUTS = 16,
    parameter DATA_WIDTH = 8
) (
    input clk,
    input rst_n,
    input [NUM_INPUTS-1:0] optical_inputs,
    output reg [DATA_WIDTH+4-1:0] electrical_output
);
    integer i;
    reg [DATA_WIDTH+4-1:0] sum;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            electrical_output <= 0;
        end else begin
            sum = 0;
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                sum = sum + optical_inputs[i];
            end
            electrical_output <= sum;
        end
    end
endmodule
""".strip()


@pytest.fixture
def invalid_verilog_code() -> str:
    """Invalid Verilog code for error testing."""
    return """
module broken_module (
    input clk,
    // Missing semicolon
    input rst_n
    output result
);
    // Syntax errors for testing
    always @(posedge clk begin
        result <= 1'b0
    end
    // Missing endmodule
""".strip()


@pytest.fixture
def sample_pytorch_models() -> Dict[str, torch.nn.Module]:
    """Sample PyTorch models for transpilation testing."""
    return {
        'simple_linear': torch.nn.Linear(10, 5),
        'relu_network': torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ),
        'conv_network': torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 10)
        )
    }


@pytest.fixture
def sample_input_tensors() -> Dict[str, torch.Tensor]:
    """Sample input tensors for different test scenarios."""
    return {
        'small_vector': torch.randn(1, 10),
        'image_28x28': torch.randn(1, 1, 28, 28),
        'image_224x224': torch.randn(1, 3, 224, 224),
        'sequence_data': torch.randn(1, 50, 100),
        'batch_vectors': torch.randn(32, 784),
        'batch_images': torch.randn(16, 1, 28, 28),
        'large_tensor': torch.randn(1, 4096),
        'zero_tensor': torch.zeros(1, 10),
        'ones_tensor': torch.ones(1, 10),
        'random_sparse': torch.zeros(1, 100).scatter_(1, torch.tensor([[5, 15, 25]]), 1.0)
    }


@pytest.fixture
def pdk_test_data() -> Dict[str, Dict[str, Any]]:
    """Test data for different PDK configurations."""
    return {
        'skywater130': {
            'name': 'SkyWater 130nm',
            'wavelength_range': (1260, 1620),  # nm
            'default_wavelength': 1550,
            'temperature_range': (-40, 125),  # Celsius
            'default_temperature': 25,
            'metal_layers': 5,
            'min_feature_size': 130,  # nm
            'supported_operations': ['linear', 'relu', 'conv2d'],
            'max_power': 1000,  # mW
        },
        'globalfoundries45': {
            'name': 'GlobalFoundries 45nm SOI',
            'wavelength_range': (1270, 1580),
            'default_wavelength': 1310,
            'temperature_range': (-40, 85),
            'default_temperature': 25,
            'metal_layers': 7,
            'min_feature_size': 45,
            'supported_operations': ['linear', 'relu', 'conv2d', 'batch_norm'],
            'max_power': 500,
        },
        'asap7': {
            'name': 'ASAP7 7nm FinFET',
            'wavelength_range': (1500, 1600),
            'default_wavelength': 1550,
            'temperature_range': (0, 85),
            'default_temperature': 25,
            'metal_layers': 9,
            'min_feature_size': 7,
            'supported_operations': ['linear', 'relu', 'conv2d', 'batch_norm', 'attention'],
            'max_power': 100,
        }
    }


@pytest.fixture
def performance_benchmarks() -> Dict[str, Dict[str, float]]:
    """Expected performance benchmarks for different operations."""
    return {
        'linear_layer': {
            'transpilation_time_ms': 5.0,
            'compilation_time_ms': 100.0,
            'memory_usage_mb': 10.0,
            'energy_per_op_pj': 0.1,
            'latency_ps': 50.0,
            'area_um2': 1000.0,
        },
        'conv2d_layer': {
            'transpilation_time_ms': 15.0,
            'compilation_time_ms': 500.0,
            'memory_usage_mb': 50.0,
            'energy_per_op_pj': 0.5,
            'latency_ps': 200.0,
            'area_um2': 5000.0,
        },
        'relu_activation': {
            'transpilation_time_ms': 2.0,
            'compilation_time_ms': 20.0,
            'memory_usage_mb': 1.0,
            'energy_per_op_pj': 0.01,
            'latency_ps': 10.0,
            'area_um2': 100.0,
        }
    }


@pytest.fixture
def error_test_cases() -> List[Dict[str, Any]]:
    """Test cases that should raise specific errors."""
    return [
        {
            'name': 'unsupported_operation',
            'model': torch.nn.LSTM(10, 20, 2),
            'expected_error': 'UnsupportedOperationError',
            'error_message': 'LSTM operations not supported'
        },
        {
            'name': 'invalid_input_shape',
            'model': torch.nn.Linear(10, 5),
            'input_tensor': torch.randn(1, 20),  # Wrong input size
            'expected_error': 'ValueError',
            'error_message': 'Input tensor shape mismatch'
        },
        {
            'name': 'empty_model',
            'model': torch.nn.Sequential(),
            'expected_error': 'ValueError',
            'error_message': 'Empty model not supported'
        },
        {
            'name': 'invalid_pdk',
            'pdk_name': 'nonexistent_pdk',
            'expected_error': 'PDKNotFoundError',
            'error_message': 'PDK not found'
        }
    ]


@pytest.fixture
def regression_test_data() -> Dict[str, Any]:
    """Regression test data to ensure consistent behavior."""
    return {
        'version': '0.1.0',
        'test_models': {
            'linear_10_5': {
                'model_hash': 'a1b2c3d4e5f6',
                'expected_verilog_hash': 'f6e5d4c3b2a1',
                'expected_performance': {
                    'energy_pj': 0.15,
                    'latency_ps': 45.0,
                    'area_um2': 950.0
                }
            }
        },
        'compatibility_matrix': {
            'python_versions': ['3.8', '3.9', '3.10', '3.11'],
            'pytorch_versions': ['2.0.0', '2.0.1', '2.1.0'],
            'supported_platforms': ['linux', 'macos', 'windows']
        }
    }


@pytest.fixture
def mock_simulation_results() -> Dict[str, Any]:
    """Mock simulation results for testing without actual simulation."""
    return {
        'compilation_success': True,
        'compilation_time_ms': 125.5,
        'simulation_time_ms': 45.2,
        'functional_verification': True,
        'timing_analysis': {
            'max_frequency_mhz': 500.0,
            'critical_path_ps': 2000.0,
            'setup_time_ps': 100.0,
            'hold_time_ps': 50.0
        },
        'power_analysis': {
            'static_power_mw': 10.5,
            'dynamic_power_mw': 45.2,
            'total_power_mw': 55.7,
            'energy_per_op_pj': 0.112
        },
        'area_analysis': {
            'logic_area_um2': 2500.0,
            'memory_area_um2': 1500.0,
            'io_area_um2': 500.0,
            'total_area_um2': 4500.0
        },
        'resource_utilization': {
            'luts': 1250,
            'ffs': 800,
            'brams': 4,
            'dsps': 16
        }
    }


@pytest.fixture
def test_file_paths(tmp_path) -> Dict[str, Path]:
    """Temporary file paths for testing file operations."""
    paths = {
        'model_file': tmp_path / 'test_model.pth',
        'verilog_file': tmp_path / 'output.v',
        'config_file': tmp_path / 'config.yaml',
        'log_file': tmp_path / 'test.log',
        'results_dir': tmp_path / 'results',
        'cache_dir': tmp_path / 'cache'
    }
    
    # Create directories
    paths['results_dir'].mkdir(exist_ok=True)
    paths['cache_dir'].mkdir(exist_ok=True)
    
    return paths


@pytest.fixture
def large_model_data() -> Dict[str, Any]:
    """Large model data for stress testing."""
    return {
        'resnet50_like': {
            'total_parameters': 25_557_032,
            'conv_layers': 49,
            'linear_layers': 1,
            'batch_norm_layers': 49,
            'activation_layers': 49,
            'expected_transpilation_time_s': 30.0,
            'expected_memory_gb': 2.0
        },
        'transformer_like': {
            'total_parameters': 110_000_000,
            'attention_layers': 12,
            'linear_layers': 37,
            'layer_norm_layers': 25,
            'expected_transpilation_time_s': 120.0,
            'expected_memory_gb': 8.0
        }
    }


@pytest.fixture(scope="session")
def test_datasets():
    """Sample datasets for testing."""
    np.random.seed(42)  # For reproducible tests
    
    return {
        'classification': {
            'X': np.random.randn(1000, 784),
            'y': np.random.randint(0, 10, 1000)
        },
        'regression': {
            'X': np.random.randn(500, 20),
            'y': np.random.randn(500, 1)
        },
        'timeseries': {
            'X': np.random.randn(200, 50, 10),  # samples, timesteps, features
            'y': np.random.randint(0, 5, 200)
        },
        'images': {
            'X': np.random.randint(0, 256, (100, 1, 28, 28), dtype=np.uint8),
            'y': np.random.randint(0, 10, 100)
        }
    }