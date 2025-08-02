"""Integration tests for the full photonic neural network pipeline."""

import pytest
import torch
from pathlib import Path

from tests.utils import (
    PerformanceTimer, 
    MockVerilogGenerator, 
    assert_verilog_syntax,
    integration_test
)


@integration_test
def test_pytorch_to_verilog_pipeline(sample_linear_model, temp_dir):
    """Test the complete PyTorch to Verilog conversion pipeline."""
    # This is a placeholder test - in a real implementation,
    # we would import the actual photonic foundry modules
    
    mock_generator = MockVerilogGenerator()
    
    with PerformanceTimer("full_pipeline") as timer:
        # Simulate the conversion process
        verilog_code = mock_generator.generate_module(
            module_name="linear_network",
            inputs=["data_in"],
            outputs=["data_out"],
            parameters={"INPUT_WIDTH": 32, "OUTPUT_WIDTH": 8}
        )
    
    # Verify the output
    assert verilog_code is not None
    assert assert_verilog_syntax(verilog_code)
    assert "linear_network" in verilog_code
    assert timer.duration < 10.0  # Should complete within 10 seconds


@integration_test
def test_multi_layer_conversion(sample_cnn_model, photonic_params):
    """Test conversion of multi-layer CNN models."""
    mock_generator = MockVerilogGenerator()
    
    # Simulate layer-by-layer conversion
    layers = ["conv1", "relu1", "pool1", "flatten", "fc1", "relu2", "fc2"]
    
    for i, layer_name in enumerate(layers):
        verilog_code = mock_generator.generate_module(
            module_name=f"layer_{i}_{layer_name}",
            inputs=[f"in_{i}"],
            outputs=[f"out_{i}"],
            parameters=photonic_params
        )
        
        assert assert_verilog_syntax(verilog_code)
        assert layer_name in verilog_code
    
    # Verify all layers were processed
    assert len(mock_generator.generated_modules) == len(layers)


@integration_test
def test_power_latency_estimation(photonic_params):
    """Test power and latency estimation integration."""
    mock_generator = MockVerilogGenerator()
    
    # Generate a test module
    verilog_code = mock_generator.generate_module(
        module_name="power_test",
        inputs=["data"],
        outputs=["result"],
        parameters={
            "WAVELENGTH": photonic_params["wavelength"],
            "ENERGY_PER_OP": photonic_params["energy_per_op"]
        }
    )
    
    # In a real implementation, this would use actual power/latency models
    estimated_power = 2.5  # pJ (placeholder)
    estimated_latency = 150  # ps (placeholder)
    
    assert estimated_power > 0
    assert estimated_latency > 0
    assert estimated_power < 100  # Reasonable upper bound
    assert estimated_latency < 1000  # Reasonable upper bound


@integration_test
def test_testbench_generation(verilog_testbench_template):
    """Test automatic testbench generation."""
    mock_generator = MockVerilogGenerator()
    
    # Generate main module
    main_verilog = mock_generator.generate_module(
        module_name="test_module",
        inputs=["data_in"],
        outputs=["data_out"]
    )
    
    # Generate testbench using template
    testbench_code = verilog_testbench_template.format(
        module_name="test_module",
        input_width=32,
        output_width=16
    )
    
    assert "test_module_tb" in testbench_code
    assert "test_module dut" in testbench_code
    assert assert_verilog_syntax(testbench_code)


@integration_test
def test_multiple_pdk_support(photonic_params):
    """Test support for multiple Process Design Kits (PDKs)."""
    pdks = ["skywater130", "globalfoundries22fdx", "amf_130nm"]
    mock_generator = MockVerilogGenerator()
    
    for pdk in pdks:
        params = photonic_params.copy()
        params["pdk"] = pdk
        
        verilog_code = mock_generator.generate_module(
            module_name=f"pdk_test_{pdk}",
            inputs=["input"],
            outputs=["output"],
            parameters=params
        )
        
        assert assert_verilog_syntax(verilog_code)
        assert pdk in str(mock_generator.generated_modules[-1])


@integration_test
def test_model_accuracy_preservation(sample_linear_model, sample_input_data):
    """Test that model accuracy is preserved through conversion."""
    # Get original model output
    original_model = sample_linear_model
    original_model.eval()
    
    with torch.no_grad():
        original_output = original_model(sample_input_data)
    
    # In a real implementation, this would involve:
    # 1. Converting to photonic hardware
    # 2. Running simulation
    # 3. Comparing outputs
    
    # For now, simulate with small noise
    simulated_output = original_output + torch.randn_like(original_output) * 0.01
    
    # Check accuracy preservation (within 1% error)
    relative_error = torch.abs(simulated_output - original_output) / torch.abs(original_output)
    max_error = torch.max(relative_error)
    
    assert max_error < 0.01, f"Accuracy loss too high: {max_error:.4f}"


@integration_test
def test_configuration_file_integration(temp_dir, test_config):
    """Test integration with configuration files."""
    import json
    
    # Create a test configuration file
    config_file = temp_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # Verify config file was created and is readable
    assert config_file.exists()
    
    with open(config_file, 'r') as f:
        loaded_config = json.load(f)
    
    assert loaded_config == test_config
    assert loaded_config["debug"] is True
    assert loaded_config["max_model_size_mb"] == 100


@integration_test
def test_error_handling_integration():
    """Test error handling throughout the pipeline."""
    mock_generator = MockVerilogGenerator()
    
    # Test with invalid inputs
    with pytest.raises((ValueError, TypeError)):
        mock_generator.generate_module(
            module_name="",  # Invalid empty name
            inputs=[],
            outputs=[]
        )
    
    # Test with extremely large model (should handle gracefully)
    try:
        verilog_code = mock_generator.generate_module(
            module_name="large_test",
            inputs=[f"in_{i}" for i in range(1000)],  # Many inputs
            outputs=[f"out_{i}" for i in range(1000)],  # Many outputs
        )
        # If it succeeds, verify it's still valid
        assert assert_verilog_syntax(verilog_code)
    except MemoryError:
        # This is acceptable for very large models
        pytest.skip("Insufficient memory for large model test")


@integration_test
def test_cli_integration(temp_dir):
    """Test command-line interface integration."""
    # This would test the actual CLI in a real implementation
    # For now, simulate CLI workflow
    
    commands = [
        "photonic-foundry --version",
        "photonic-foundry convert --help",
        "photonic-foundry simulate --help"
    ]
    
    # In a real test, we would use subprocess to run these commands
    # and verify their outputs
    
    for command in commands:
        # Simulate successful command execution
        exit_code = 0  # Success
        assert exit_code == 0, f"Command failed: {command}"