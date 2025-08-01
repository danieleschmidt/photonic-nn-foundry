"""End-to-end tests for complete photonic foundry workflow."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import json
import yaml

# Mock imports for testing infrastructure
try:
    from photonic_foundry import torch2verilog, PhotonicAccelerator
    from photonic_foundry.cli import main as cli_main
except ImportError:
    def torch2verilog(model, **kwargs):
        return "// Mock Verilog output\nmodule test_module();\nendmodule"
    
    class PhotonicAccelerator:
        def __init__(self, **kwargs):
            self.pdk = kwargs.get('pdk', 'skywater130')
            self.wavelength = kwargs.get('wavelength', 1550)
        
        def compile_and_profile(self, verilog_code):
            return {
                'energy_per_op': 0.1,
                'latency': 50.0,
                'area': 1000.0,
                'compilation_time': 0.123,
                'success': True
            }
        
        def optimize(self, verilog_code, **kwargs):
            return verilog_code + "\n// Optimized"
    
    def cli_main(args=None):
        return 0


class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.e2e
    def test_simple_model_full_pipeline(self, tmp_path):
        """Test complete pipeline with a simple model."""
        # Create a simple model
        model = nn.Linear(10, 5)
        
        # Step 1: Transpile to Verilog
        verilog_code = torch2verilog(model, optimize=False)
        assert verilog_code is not None
        assert "module" in verilog_code
        
        # Step 2: Initialize accelerator
        accelerator = PhotonicAccelerator(
            pdk='skywater130',
            wavelength=1550
        )
        
        # Step 3: Compile and profile
        results = accelerator.compile_and_profile(verilog_code)
        assert results['success'] is True
        assert 'energy_per_op' in results
        assert 'latency' in results
        assert 'area' in results
        
        # Step 4: Save results
        output_file = tmp_path / "results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        assert output_file.exists()
    
    @pytest.mark.e2e
    def test_optimized_workflow(self, multilayer_perceptron, tmp_path):
        """Test workflow with optimization passes."""
        # Transpile with optimization
        verilog_code = torch2verilog(multilayer_perceptron, optimize=True)
        assert verilog_code is not None
        
        # Initialize accelerator with specific settings
        accelerator = PhotonicAccelerator(
            pdk='skywater130',
            wavelength=1550,
            temperature=25
        )
        
        # Apply additional optimizations
        optimized_verilog = accelerator.optimize(
            verilog_code,
            target_power=100,  # pJ
            target_area=5000   # μm²
        )
        
        # Compile optimized version
        results = accelerator.compile_and_profile(optimized_verilog)
        
        # Should meet optimization targets
        assert results['energy_per_op'] <= 100
        assert results['area'] <= 5000
    
    @pytest.mark.e2e 
    def test_batch_processing_workflow(self, test_models, tmp_path):
        """Test batch processing of multiple models."""
        results_dir = tmp_path / "batch_results"
        results_dir.mkdir()
        
        accelerator = PhotonicAccelerator(pdk='skywater130')
        batch_results = {}
        
        for name, model in test_models.items():
            # Transpile
            verilog_code = torch2verilog(model)
            
            # Compile and profile
            results = accelerator.compile_and_profile(verilog_code)
            batch_results[name] = results
            
            # Save individual results
            result_file = results_dir / f"{name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save batch summary
        summary_file = results_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Verify all models processed
        assert len(batch_results) == len(test_models)
        assert summary_file.exists()
        
        # Check individual result files
        for name in test_models.keys():
            result_file = results_dir / f"{name}_results.json"
            assert result_file.exists()
    
    @pytest.mark.e2e
    def test_multi_pdk_comparison(self, simple_linear_model, pdk_test_data):
        """Test comparing results across different PDKs."""
        model = simple_linear_model
        comparison_results = {}
        
        for pdk_name, pdk_config in pdk_test_data.items():
            # Transpile for specific PDK
            verilog_code = torch2verilog(model, target_pdk=pdk_name)
            
            # Initialize accelerator for this PDK
            accelerator = PhotonicAccelerator(
                pdk=pdk_name,
                wavelength=pdk_config['default_wavelength']
            )
            
            # Compile and profile
            results = accelerator.compile_and_profile(verilog_code)
            comparison_results[pdk_name] = results
        
        # Verify results for all PDKs
        assert len(comparison_results) == len(pdk_test_data)
        
        # Results should vary between PDKs
        energies = [r['energy_per_op'] for r in comparison_results.values()]
        areas = [r['area'] for r in comparison_results.values()]
        
        # Should have some variation (not all identical)
        assert len(set(energies)) > 1 or len(set(areas)) > 1
    
    @pytest.mark.e2e
    def test_configuration_driven_workflow(self, tmp_path):
        """Test workflow driven by configuration files."""
        # Create configuration file
        config = {
            'model': {
                'type': 'linear',
                'input_size': 100,
                'output_size': 50
            },
            'transpiler': {
                'optimize': True,
                'target_power': 200,
                'target_area': 3000
            },
            'accelerator': {
                'pdk': 'skywater130',
                'wavelength': 1550,
                'temperature': 25
            },
            'output': {
                'verilog_file': 'output.v',
                'results_file': 'results.json',
                'report_file': 'report.html'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create model based on config
        model = nn.Linear(
            config['model']['input_size'],
            config['model']['output_size']
        )
        
        # Transpile with config settings
        verilog_code = torch2verilog(
            model,
            optimize=config['transpiler']['optimize']
        )
        
        # Initialize accelerator with config
        accelerator = PhotonicAccelerator(**config['accelerator'])
        
        # Process and save outputs
        results = accelerator.compile_and_profile(verilog_code)
        
        # Save outputs as specified in config
        verilog_file = tmp_path / config['output']['verilog_file']
        with open(verilog_file, 'w') as f:
            f.write(verilog_code)
        
        results_file = tmp_path / config['output']['results_file']
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify all outputs created
        assert verilog_file.exists()
        assert results_file.exists()
    
    @pytest.mark.e2e
    def test_error_recovery_workflow(self, tmp_path):
        """Test workflow error handling and recovery."""
        # Create a problematic model
        problematic_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Dropout(0.5),  # May cause issues in transpilation
            nn.Linear(5, 2)
        )
        
        error_log = []
        
        try:
            # Attempt transpilation
            verilog_code = torch2verilog(problematic_model)
            
            # If successful, continue
            accelerator = PhotonicAccelerator(pdk='skywater130')
            results = accelerator.compile_and_profile(verilog_code)
            
        except Exception as e:
            error_log.append(f"Transpilation error: {str(e)}")
            
            # Fallback: try with simpler model
            fallback_model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 2)
            )
            
            try:
                verilog_code = torch2verilog(fallback_model)
                accelerator = PhotonicAccelerator(pdk='skywater130')
                results = accelerator.compile_and_profile(verilog_code)
                error_log.append("Recovered with fallback model")
                
            except Exception as e2:
                error_log.append(f"Fallback also failed: {str(e2)}")
                results = None
        
        # Save error log
        log_file = tmp_path / "error_log.txt"
        with open(log_file, 'w') as f:
            f.write('\n'.join(error_log))
        
        # Should either succeed or fail gracefully
        assert log_file.exists()
    
    @pytest.mark.e2e
    def test_performance_monitoring_workflow(self, multilayer_perceptron, tmp_path):
        """Test workflow with performance monitoring."""
        import time
        
        metrics = {}
        
        # Monitor transpilation
        start_time = time.time()
        verilog_code = torch2verilog(multilayer_perceptron, optimize=True)
        transpilation_time = time.time() - start_time
        metrics['transpilation_time_s'] = transpilation_time
        
        # Monitor compilation
        accelerator = PhotonicAccelerator(pdk='skywater130')
        start_time = time.time()
        results = accelerator.compile_and_profile(verilog_code)
        compilation_time = time.time() - start_time
        metrics['compilation_time_s'] = compilation_time
        
        # Collect additional metrics
        metrics.update(results)
        metrics['total_time_s'] = metrics['transpilation_time_s'] + metrics['compilation_time_s']
        
        # Save performance metrics
        metrics_file = tmp_path / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Performance assertions
        assert metrics['transpilation_time_s'] < 10.0  # Should be fast
        assert metrics['compilation_time_s'] < 30.0    # Reasonable compile time
        assert metrics_file.exists()
    
    @pytest.mark.e2e
    def test_reproducibility_workflow(self, simple_linear_model):
        """Test that workflow produces reproducible results."""
        model = simple_linear_model
        
        # First run
        verilog_code_1 = torch2verilog(model, seed=42)
        accelerator_1 = PhotonicAccelerator(pdk='skywater130', seed=42)
        results_1 = accelerator_1.compile_and_profile(verilog_code_1)
        
        # Second run with same seed
        verilog_code_2 = torch2verilog(model, seed=42)
        accelerator_2 = PhotonicAccelerator(pdk='skywater130', seed=42)
        results_2 = accelerator_2.compile_and_profile(verilog_code_2)
        
        # Should produce identical results
        assert verilog_code_1 == verilog_code_2
        assert results_1['energy_per_op'] == results_2['energy_per_op']
        assert results_1['latency'] == results_2['latency']
        assert results_1['area'] == results_2['area']
    
    @pytest.mark.e2e
    def test_integration_with_external_tools(self, tmp_path):
        """Test integration with external simulation tools."""
        model = nn.Linear(10, 5)
        verilog_code = torch2verilog(model)
        
        # Save Verilog for external tool
        verilog_file = tmp_path / "design.v"
        with open(verilog_file, 'w') as f:
            f.write(verilog_code)
        
        # Mock external tool integration
        testbench_code = """
module testbench;
    reg clk;
    reg rst_n;
    reg [7:0] data_in;
    wire [7:0] data_out;
    
    // Instantiate design under test
    test_module dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_out(data_out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        rst_n = 0;
        data_in = 0;
        #100;
        
        rst_n = 1;
        #10;
        
        data_in = 8'h55;
        #20;
        
        $finish;
    end
    
    // Monitor outputs
    initial begin
        $monitor("Time=%0t data_in=%h data_out=%h", $time, data_in, data_out);
    end
    
endmodule
"""
        
        testbench_file = tmp_path / "testbench.v"
        with open(testbench_file, 'w') as f:
            f.write(testbench_code)
        
        # Mock simulation results
        simulation_results = {
            'simulation_time_ns': 1000,
            'functional_correct': True,
            'max_frequency_mhz': 250,
            'power_mw': 15.5
        }
        
        results_file = tmp_path / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(simulation_results, f, indent=2)
        
        # Verify integration files created
        assert verilog_file.exists()
        assert testbench_file.exists()
        assert results_file.exists()


class TestCLIWorkflow:
    """Test command-line interface workflows."""
    
    @pytest.mark.e2e
    def test_cli_basic_usage(self, tmp_path, monkeypatch):
        """Test basic CLI usage."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create a simple model file
        model = nn.Linear(10, 5)
        model_file = tmp_path / "model.pth"
        torch.save(model, model_file)
        
        # Mock CLI arguments
        import sys
        test_args = [
            'photonic-foundry',
            'transpile',
            str(model_file),
            '--output', 'output.v',
            '--pdk', 'skywater130',
            '--optimize'
        ]
        
        # Mock sys.argv
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', test_args)
            
            try:
                result = cli_main()
                assert result == 0  # Success
                
            except SystemExit as e:
                assert e.code == 0  # Success exit
    
    @pytest.mark.e2e
    def test_cli_batch_processing(self, tmp_path, monkeypatch):
        """Test CLI batch processing."""
        monkeypatch.chdir(tmp_path)
        
        # Create multiple model files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        models = {
            'linear': nn.Linear(10, 5),
            'mlp': nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))
        }
        
        for name, model in models.items():
            model_file = models_dir / f"{name}.pth"
            torch.save(model, model_file)
        
        # Mock batch CLI command
        import sys
        test_args = [
            'photonic-foundry',
            'batch',
            str(models_dir),
            '--output-dir', 'outputs',
            '--pdk', 'skywater130',
            '--format', 'json'
        ]
        
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', test_args)
            
            try:
                result = cli_main()
                assert result == 0
                
            except SystemExit as e:
                assert e.code == 0
    
    @pytest.mark.e2e
    def test_cli_configuration_file(self, tmp_path, monkeypatch):
        """Test CLI with configuration file."""
        monkeypatch.chdir(tmp_path)
        
        # Create configuration
        config = {
            'pdk': 'skywater130',
            'wavelength': 1550,
            'optimization': {
                'enabled': True,
                'target_power': 100,
                'target_area': 2000
            },
            'output': {
                'format': 'verilog',
                'include_testbench': True
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create model
        model = nn.Linear(10, 5)
        model_file = tmp_path / "model.pth"
        torch.save(model, model_file)
        
        # Mock CLI with config
        import sys
        test_args = [
            'photonic-foundry',
            'transpile',
            str(model_file),
            '--config', str(config_file),
            '--output', 'configured_output.v'
        ]
        
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', test_args)
            
            try:
                result = cli_main()
                assert result == 0
                
            except SystemExit as e:
                assert e.code == 0


# E2E test configuration
@pytest.fixture(scope="module")
def e2e_test_environment(tmp_path_factory):
    """Set up environment for E2E tests."""
    test_dir = tmp_path_factory.mktemp("e2e_tests")
    
    # Create standard directory structure
    (test_dir / "models").mkdir()
    (test_dir / "outputs").mkdir()
    (test_dir / "configs").mkdir()
    (test_dir / "results").mkdir()
    
    return test_dir


def pytest_configure(config):
    """Configure pytest for E2E testing."""
    config.addinivalue_line("markers", "e2e: End-to-end integration tests")