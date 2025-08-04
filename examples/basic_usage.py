#!/usr/bin/env python3
"""
Basic usage example for PhotonicFoundry
Demonstrates core functionality without external dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_neural_network():
    """Create a simple neural network representation for testing."""
    # Simple 2-layer MLP: 784 -> 128 -> 10
    network = {
        'name': 'simple_mlp_example',
        'layers': [
            {
                'type': 'linear',
                'input_size': 784,
                'output_size': 128,
                'weights': np.random.randn(128, 784).tolist(),
                'component_count': 784 * 128,  # One MZI per weight
                'components': [
                    {
                        'type': 'mach_zehnder_interferometer',
                        'params': {
                            'phase_shifter_bits': 8,
                            'insertion_loss': 0.1,
                            'crosstalk': -30
                        }
                    } for _ in range(784 * 128)
                ]
            },
            {
                'type': 'activation',
                'input_size': 128,
                'output_size': 128,
                'component_count': 128,
                'components': [
                    {
                        'type': 'electro_optic_modulator',
                        'params': {
                            'modulation_depth': 0.9,
                            'bandwidth': 10e9  # 10 GHz
                        }
                    } for _ in range(128)
                ]
            },
            {
                'type': 'linear',
                'input_size': 128,
                'output_size': 10,
                'weights': np.random.randn(10, 128).tolist(),
                'component_count': 128 * 10,
                'components': [
                    {
                        'type': 'mach_zehnder_interferometer',
                        'params': {
                            'phase_shifter_bits': 8,
                            'insertion_loss': 0.1,
                            'crosstalk': -30
                        }
                    } for _ in range(128 * 10)
                ]
            }
        ],
        'connections': [(0, 1), (1, 2)],
        'total_components': 784 * 128 + 128 + 128 * 10,
        'pdk': 'skywater130',
        'wavelength': 1550
    }
    
    return network

def simulate_photonic_inference(network: Dict[str, Any], input_data: np.ndarray) -> Dict[str, Any]:
    """
    Simulate inference on a photonic neural network.
    
    This is a simplified simulation without external dependencies.
    """
    logger.info(f"Simulating inference on {network['name']}")
    
    current_data = input_data.copy()
    layer_outputs = []
    
    for i, layer in enumerate(network['layers']):
        logger.info(f"Processing layer {i+1}: {layer['type']}")
        
        if layer['type'] == 'linear':
            # Simulate matrix multiplication with quantization
            weights = np.array(layer['weights'])
            
            # Apply photonic non-idealities
            noise_factor = 0.02  # 2% noise
            crosstalk_factor = 0.001  # 0.1% crosstalk
            
            # Matrix multiplication
            result = np.dot(current_data, weights.T)
            
            # Add noise and crosstalk
            noise = np.random.normal(0, noise_factor * np.std(result), result.shape)
            crosstalk = np.random.normal(0, crosstalk_factor * np.std(result), result.shape)
            
            current_data = result + noise + crosstalk
            
        elif layer['type'] == 'activation':
            # Simulate electro-optic ReLU activation
            current_data = np.maximum(0, current_data)
            
            # Add modulator non-linearity
            saturation_level = 5.0
            current_data = np.tanh(current_data / saturation_level) * saturation_level
            
        layer_outputs.append(current_data.copy())
        
    # Calculate performance metrics
    total_mzis = sum(len([c for c in layer.get('components', []) 
                         if c.get('type') == 'mach_zehnder_interferometer']) 
                    for layer in network['layers'])
    
    total_modulators = sum(len([c for c in layer.get('components', []) 
                              if c.get('type') == 'electro_optic_modulator']) 
                          for layer in network['layers'])
    
    # Energy calculation (simplified)
    mzi_energy = total_mzis * 0.5  # 0.5 pJ per MZI
    modulator_energy = total_modulators * 1.0  # 1.0 pJ per modulator
    total_energy = mzi_energy + modulator_energy
    
    # Latency calculation
    num_layers = len(network['layers'])
    total_latency = num_layers * 50  # 50 ps per layer
    
    # Area calculation
    mzi_area = total_mzis * 0.001  # 0.001 mm² per MZI
    modulator_area = total_modulators * 0.0005  # 0.0005 mm² per modulator
    total_area = mzi_area + modulator_area
    
    results = {
        'output': current_data,
        'layer_outputs': layer_outputs,
        'performance_metrics': {
            'energy_per_inference_pj': total_energy,
            'latency_ps': total_latency,
            'area_mm2': total_area,
            'throughput_gops': 1e12 / total_latency,  # GOPS
            'total_mzis': total_mzis,
            'total_modulators': total_modulators
        }
    }
    
    return results

def generate_simple_verilog(network: Dict[str, Any]) -> str:
    """Generate basic Verilog code for the photonic network."""
    
    module_name = network['name'].replace('-', '_')
    num_layers = len(network['layers'])
    
    verilog_code = f"""
// Generated Photonic Neural Network: {network['name']}
// Target PDK: {network['pdk']}
// Operating Wavelength: {network['wavelength']}nm
// Total Layers: {num_layers}
// Total Components: {network['total_components']}

module {module_name} (
    input clk,
    input rst_n,
    input [31:0] data_in,
    input valid_in,
    output [31:0] data_out,
    output valid_out
);

// Parameters
parameter INPUT_WIDTH = 32;
parameter OUTPUT_WIDTH = 32;
parameter PRECISION = 8;

// Internal signals
wire [PRECISION-1:0] layer_interconnect [{num_layers}:0];
wire layer_valid [{num_layers}:0];

// Input assignment
assign layer_interconnect[0] = data_in[PRECISION-1:0];
assign layer_valid[0] = valid_in;

// Layer instantiations
genvar i;
generate
    for (i = 0; i < {num_layers}; i = i + 1) begin: layer_gen
        photonic_layer #(
            .LAYER_TYPE(i),
            .INPUT_SIZE({network['layers'][0]['input_size']}),
            .OUTPUT_SIZE({network['layers'][0]['output_size']}),
            .PRECISION(PRECISION)
        ) layer_inst (
            .clk(clk),
            .rst_n(rst_n),
            .data_in(layer_interconnect[i]),
            .valid_in(layer_valid[i]),
            .data_out(layer_interconnect[i+1]),
            .valid_out(layer_valid[i+1])
        );
    end
endgenerate

// Output assignment
assign data_out = layer_interconnect[{num_layers}];
assign valid_out = layer_valid[{num_layers}];

endmodule

// Basic photonic layer module
module photonic_layer #(
    parameter LAYER_TYPE = 0,
    parameter INPUT_SIZE = 128,
    parameter OUTPUT_SIZE = 128,
    parameter PRECISION = 8
) (
    input clk,
    input rst_n,
    input [PRECISION-1:0] data_in,
    input valid_in,
    output [PRECISION-1:0] data_out,
    output valid_out
);

// Simplified photonic processing
reg [PRECISION-1:0] processed_data;
reg valid_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        processed_data <= 0;
        valid_reg <= 1'b0;
    end else begin
        if (valid_in) begin
            // Simulate photonic processing with delay
            processed_data <= data_in + 1; // Simplified transformation
            valid_reg <= 1'b1;
        end else begin
            valid_reg <= 1'b0;
        end
    end
end

assign data_out = processed_data;
assign valid_out = valid_reg;

endmodule
"""
    
    return verilog_code

def main():
    """Main demonstration of PhotonicFoundry capabilities."""
    
    print("=== PhotonicFoundry Basic Usage Demo ===")
    print()
    
    # Step 1: Create a simple neural network
    print("1. Creating simple neural network...")
    network = create_simple_neural_network()
    print(f"   Network: {network['name']}")
    print(f"   Layers: {len(network['layers'])}")
    print(f"   Total components: {network['total_components']:,}")
    print()
    
    # Step 2: Generate random input data
    print("2. Generating test input data...")
    input_data = np.random.randn(784) * 0.5  # Normalized input
    print(f"   Input shape: {input_data.shape}")
    print(f"   Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    print()
    
    # Step 3: Simulate photonic inference
    print("3. Running photonic inference simulation...")
    results = simulate_photonic_inference(network, input_data)
    print(f"   Output shape: {results['output'].shape}")
    print(f"   Output range: [{results['output'].min():.3f}, {results['output'].max():.3f}]")
    print()
    
    # Step 4: Display performance metrics
    print("4. Performance Analysis:")
    metrics = results['performance_metrics']
    print(f"   Energy per inference: {metrics['energy_per_inference_pj']:.1f} pJ")
    print(f"   Latency: {metrics['latency_ps']:.0f} ps")
    print(f"   Area: {metrics['area_mm2']:.3f} mm²")
    print(f"   Throughput: {metrics['throughput_gops']:.1f} GOPS")
    print(f"   Total MZIs: {metrics['total_mzis']:,}")
    print(f"   Total modulators: {metrics['total_modulators']:,}")
    print()
    
    # Step 5: Generate Verilog code
    print("5. Generating Verilog code...")
    verilog_code = generate_simple_verilog(network)
    verilog_lines = len(verilog_code.splitlines())
    print(f"   Generated {verilog_lines} lines of Verilog")
    print(f"   Module name: {network['name'].replace('-', '_')}")
    print()
    
    # Step 6: Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Verilog
    verilog_path = os.path.join(output_dir, f"{network['name']}.v")
    with open(verilog_path, 'w') as f:
        f.write(verilog_code)
    
    # Save network configuration
    config_path = os.path.join(output_dir, f"{network['name']}_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Photonic Neural Network Configuration\n")
        f.write(f"=====================================\n")
        f.write(f"Name: {network['name']}\n")
        f.write(f"PDK: {network['pdk']}\n")
        f.write(f"Wavelength: {network['wavelength']}nm\n")
        f.write(f"Total Components: {network['total_components']:,}\n")
        f.write(f"Energy per inference: {metrics['energy_per_inference_pj']:.1f} pJ\n")
        f.write(f"Latency: {metrics['latency_ps']:.0f} ps\n")
        f.write(f"Area: {metrics['area_mm2']:.3f} mm²\n")
        f.write(f"Throughput: {metrics['throughput_gops']:.1f} GOPS\n")
    
    print("6. Results saved:")
    print(f"   Verilog: {verilog_path}")
    print(f"   Configuration: {config_path}")
    print()
    
    print("=== Demo Complete ===")
    print("PhotonicFoundry successfully demonstrated basic functionality!")
    print(f"Generated photonic neural network with {metrics['total_mzis']:,} MZIs")
    print(f"Estimated performance: {metrics['throughput_gops']:.1f} GOPS at {metrics['energy_per_inference_pj']:.1f} pJ per inference")

if __name__ == "__main__":
    main()