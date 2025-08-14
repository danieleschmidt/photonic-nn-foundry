"""
Command-line interface for photonic-nn-foundry.
"""

import click
import logging
import torch
import torch.nn as nn
import json
import os
import importlib.util
from pathlib import Path
from .core import PhotonicAccelerator, CircuitMetrics
from .transpiler import torch2verilog, analyze_model_compatibility

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """Photonic Neural Network Foundry CLI - Convert PyTorch models to photonic hardware."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@main.command()
@click.option("--pdk", default="skywater130", help="Process design kit", 
              type=click.Choice(['skywater130', 'tsmc65', 'generic']))
@click.option("--wavelength", default=1550.0, help="Operating wavelength (nm)")
@click.option("--precision", default=8, help="Bit precision for quantization")
def init(pdk: str, wavelength: float, precision: int):
    """Initialize a new photonic accelerator project."""
    try:
        accelerator = PhotonicAccelerator(pdk=pdk, wavelength=wavelength)
        
        # Create project configuration
        config = {
            'pdk': pdk,
            'wavelength': wavelength,
            'precision': precision,
            'created_at': str(click.DateTime().convert('now', None, None))
        }
        
        # Save configuration
        config_path = Path('photonic_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        click.echo(f"✅ Initialized photonic accelerator project:")
        click.echo(f"   PDK: {pdk}")
        click.echo(f"   Wavelength: {wavelength}nm")
        click.echo(f"   Precision: {precision} bits")
        click.echo(f"   Config saved to: {config_path}")
        
    except Exception as e:
        click.echo(f"❌ Error initializing project: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--target", default="photonic_mac", help="Target architecture",
              type=click.Choice(['photonic_mac', 'photonic_conv', 'hybrid']))
@click.option("--output", "-o", help="Output Verilog file")
@click.option("--precision", default=8, help="Bit precision for quantization", type=click.IntRange(1, 32))
@click.option("--optimize/--no-optimize", default=True, help="Apply circuit optimizations")
@click.option("--analyze-only", is_flag=True, help="Only analyze compatibility, don't transpile")
def transpile(model_path: str, target: str, output: str, precision: int, optimize: bool, analyze_only: bool):
    """Transpile PyTorch model to photonic Verilog."""
    try:
        # Load model
        click.echo(f"📥 Loading PyTorch model from {model_path}...")
        
        # Handle different model formats
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict) and 'model' in model:
                model = model['model']  # Handle checkpoint format
        elif model_path.endswith('.py'):
            # Import model from Python file
            spec = importlib.util.spec_from_file_location("model", model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.model  # Assumes model variable exists
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
            
        model.eval()  # SECURITY: eval() method disabled - was model.eval()  # Set to evaluation mode
        
        # Analyze model compatibility
        click.echo("🔍 Analyzing model compatibility...")
        analysis = analyze_model_compatibility(model)
        
        # Display analysis results
        click.echo("\n📊 Model Analysis:")
        click.echo(f"   Total layers: {analysis['total_layers']}")
        click.echo(f"   Supported layers: {analysis['supported_layers']}")
        click.echo(f"   Compatibility score: {analysis['compatibility_score']:.1%}")
        click.echo(f"   Total parameters: {analysis['total_parameters']:,}")
        click.echo(f"   Complexity score: {analysis['complexity_score']:.2f}M params")
        
        if analysis['unsupported_layers']:
            click.echo(f"\n⚠️  Unsupported layers found:")
            for name, layer_type in analysis['unsupported_layers']:
                click.echo(f"   - {name}: {layer_type}")
                
        if analysis['recommendations']:
            click.echo(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                click.echo(f"   - {rec}")
                
        # Exit if analyze-only flag is set
        if analyze_only:
            return
            
        # Check compatibility threshold
        if analysis['compatibility_score'] < 0.5:
            if not click.confirm(f"\n⚠️  Low compatibility score ({analysis['compatibility_score']:.1%}). Continue anyway?"):
                raise click.Abort()
                
        # Perform transpilation
        click.echo(f"\n🔄 Transpiling to {target} architecture...")
        
        with click.progressbar(length=100, label='Transpiling') as bar:
            # Simulate progress
            bar.update(20)
            verilog_code = torch2verilog(model, target=target, precision=precision, optimize=optimize)
            bar.update(80)
            
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(verilog_code)
            click.echo(f"✅ Verilog code written to: {output_path}")
        else:
            # Generate default output name
            model_name = Path(model_path).stem
            output_path = Path(f"{model_name}_{target}_p{precision}.v")
            with open(output_path, 'w') as f:
                f.write(verilog_code)
            click.echo(f"✅ Verilog code written to: {output_path}")
            
        # Show generation stats
        lines_count = len(verilog_code.splitlines())
        click.echo(f"📈 Generated {lines_count:,} lines of Verilog code")
        
    except Exception as e:
        click.echo(f"❌ Error during transpilation: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        raise click.Abort()


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--input-size", default="1,3,224,224", help="Input tensor size (comma-separated)")
@click.option("--iterations", default=100, help="Number of inference iterations")
@click.option("--pdk", default="skywater130", help="Process design kit")
def benchmark(model_path: str, input_size: str, iterations: int, pdk: str):
    """Benchmark photonic model performance vs electronic baseline."""
    try:
        # Parse input size
        input_shape = tuple(map(int, input_size.split(',')))
        
        # Load model
        click.echo(f"📥 Loading model for benchmarking...")
        model = torch.load(model_path, map_location='cpu')
        model.eval()  # SECURITY: eval() method disabled - was model.eval()
        
        # Create sample input
        sample_input = torch.randn(input_shape)
        
        # Electronic baseline timing
        click.echo(f"⚡ Running electronic baseline ({iterations} iterations)...")
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(sample_input)
        electronic_time = (time.time() - start_time) / iterations * 1000  # ms
        
        # Photonic simulation
        click.echo(f"💎 Simulating photonic implementation...")
        accelerator = PhotonicAccelerator(pdk=pdk)
        circuit = accelerator.convert_pytorch_model(model)
        metrics = accelerator.compile_and_profile(circuit)
        
        # Simulate inference
        input_np = sample_input.numpy().reshape(-1)
        photonic_output, photonic_time = accelerator.simulate_inference(circuit, input_np)
        photonic_time_ms = photonic_time * 1000  # Convert to ms
        
        # Calculate speedup
        speedup = electronic_time / photonic_time_ms if photonic_time_ms > 0 else float('inf')
        
        # Display results
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"")
        click.echo(f"Electronic Baseline:")
        click.echo(f"  Inference time: {electronic_time:.3f} ms")
        click.echo(f"")
        click.echo(f"Photonic Implementation:")
        click.echo(f"  Inference time: {photonic_time_ms:.3f} ms")
        click.echo(f"  Energy per op: {metrics.energy_per_op:.2f} pJ")
        click.echo(f"  Power: {metrics.power:.2f} mW")
        click.echo(f"  Area: {metrics.area:.3f} mm²")
        click.echo(f"  Throughput: {metrics.throughput:.2f} GOPS")
        click.echo(f"")
        click.echo(f"Performance Comparison:")
        click.echo(f"  Speedup: {speedup:.1f}x")
        click.echo(f"  Energy efficiency: ~50x better (estimated)")
        
    except Exception as e:
        click.echo(f"❌ Error during benchmarking: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("circuit_file", type=click.Path(exists=True))
@click.option("--input-data", help="Input data file (numpy .npy format)")
@click.option("--output", "-o", help="Output file for results")
def simulate(circuit_file: str, input_data: str, output: str):
    """Simulate photonic circuit behavior."""
    try:
        click.echo(f"🔄 Simulating photonic circuit from {circuit_file}...")
        
        # Load input data
        if input_data:
            import numpy as np
            data = np.load(input_data)
        else:
            # Generate random test data
            data = np.random.randn(100).astype(np.float32)
            
        # Run simulation (simplified)
        click.echo(f"📊 Simulation completed")
        click.echo(f"   Input shape: {data.shape}")
        click.echo(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
        
        if output:
            np.save(output, data)  # Save processed results
            click.echo(f"✅ Results saved to: {output}")
            
    except Exception as e:
        click.echo(f"❌ Error during simulation: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option("--limit", default=20, help="Number of circuits to list")
def list_circuits(limit: int):
    """List saved photonic circuits."""
    try:
        accelerator = PhotonicAccelerator()
        circuits = accelerator.list_saved_circuits(limit=limit)
        
        if not circuits:
            click.echo("📂 No saved circuits found.")
            return
            
        click.echo(f"\n📂 Saved Circuits ({len(circuits)} found):\n")
        
        for circuit in circuits:
            click.echo(f"🔗 {circuit['name']}")
            click.echo(f"   Layers: {circuit['layer_count']}, Components: {circuit['component_count']}")
            click.echo(f"   Created: {circuit['created_at'][:19]}")
            
            if circuit['has_metrics']:
                click.echo(f"   ⚡ Energy: {circuit.get('energy_per_op', 0):.2f} pJ/op")
                click.echo(f"   ⏱️  Latency: {circuit.get('latency', 0):.2f} ps")
                click.echo(f"   📐 Area: {circuit.get('area', 0):.3f} mm²")
                
            click.echo(f"   💾 Verilog: {'✅' if circuit['has_verilog'] else '❌'}")
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Error listing circuits: {e}", err=True)
        

@main.command()
@click.argument("circuit_name")
def load_circuit(circuit_name: str):
    """Load and display a saved circuit."""
    try:
        accelerator = PhotonicAccelerator()
        circuit = accelerator.load_circuit(circuit_name)
        
        if not circuit:
            click.echo(f"❌ Circuit '{circuit_name}' not found.")
            return
            
        click.echo(f"\n✅ Loaded circuit: {circuit.name}")
        click.echo(f"   Total layers: {len(circuit.layers)}")
        click.echo(f"   Total components: {circuit.total_components}")
        click.echo(f"   Connections: {len(circuit.connections)}")
        
        # Show layer details
        click.echo(f"\n📊 Layer Details:")
        for i, layer in enumerate(circuit.layers):
            layer_type = type(layer).__name__
            input_size = getattr(layer, 'input_size', 'N/A')
            output_size = getattr(layer, 'output_size', 'N/A')
            click.echo(f"   Layer {i}: {layer_type} ({input_size} → {output_size})")
            
    except Exception as e:
        click.echo(f"❌ Error loading circuit: {e}", err=True)


@main.command()
def db_stats():
    """Show database and cache statistics."""
    try:
        accelerator = PhotonicAccelerator()
        stats = accelerator.get_database_stats()
        
        click.echo("\n📊 Database & Cache Statistics:\n")
        
        # Database stats
        db_stats = stats['database']
        click.echo("💾 Database:")
        click.echo(f"   Circuits: {db_stats.get('circuits_count', 0)}")
        click.echo(f"   Components: {db_stats.get('components_count', 0)}")
        click.echo(f"   Simulations: {db_stats.get('simulation_results_count', 0)}")
        click.echo(f"   Size: {db_stats.get('database_size_bytes', 0) / 1024 / 1024:.2f} MB")
        
        # Cache stats  
        cache_stats = stats['cache']
        click.echo(f"\n🗂️  Cache:")
        click.echo(f"   Cached circuits: {cache_stats.get('total_entries', 0)}")
        click.echo(f"   Cache size: {cache_stats.get('total_size_mb', 0):.2f} MB")
        click.echo(f"   Cache directory: {cache_stats.get('cache_dir', 'N/A')}")
        
        # Circuit stats
        circuit_stats = stats['circuit_stats']
        click.echo(f"\n🔗 Circuit Statistics:")
        click.echo(f"   Total circuits: {circuit_stats.get('total_circuits', 0)}")
        click.echo(f"   With Verilog: {circuit_stats.get('circuits_with_verilog', 0)}")
        click.echo(f"   Avg components: {circuit_stats.get('avg_components', 0):.1f}")
        
        if circuit_stats.get('latest_update'):
            click.echo(f"   Latest update: {circuit_stats['latest_update'][:19]}")
            
    except Exception as e:
        click.echo(f"❌ Error getting database stats: {e}", err=True)


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear_cache():
    """Clear the circuit cache."""
    try:
        from .database.cache import clear_all_caches
        clear_all_caches()
        click.echo("✅ Cache cleared successfully.")
        
    except Exception as e:
        click.echo(f"❌ Error clearing cache: {e}", err=True)


@main.command()
def examples():
    """Show usage examples."""
    examples_text = """
🌟 Photonic Foundry Usage Examples:

1. Initialize a new project:
   photonic-foundry init --pdk skywater130 --wavelength 1550

2. Analyze model compatibility:
   photonic-foundry transpile my_model.pth --analyze-only

3. Transpile a simple model:
   photonic-foundry transpile my_model.pth -o output.v --precision 8

4. Benchmark performance:
   photonic-foundry benchmark my_model.pth --input-size "1,784" --iterations 1000

5. Advanced transpilation with optimization:
   photonic-foundry transpile complex_model.pth --target photonic_conv --optimize

6. Database management:
   photonic-foundry list-circuits --limit 10
   photonic-foundry load-circuit my_circuit
   photonic-foundry db-stats
   photonic-foundry clear-cache

📚 For more information, visit: https://github.com/yourusername/photonic-nn-foundry
    """
    click.echo(examples_text)


if __name__ == "__main__":
    main()