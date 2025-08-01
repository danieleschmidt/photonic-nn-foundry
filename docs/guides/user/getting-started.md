# Getting Started with Photonic Neural Network Foundry

Welcome to the Photonic Neural Network Foundry! This guide will help you get up and running with converting your PyTorch models to photonic implementations.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker Desktop** (recommended) or Docker Engine 20.10+
- **Python 3.8+** for local development
- **Git** for version control
- **4GB+ available RAM** for simulation workloads

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/photonic-nn-foundry.git
cd photonic-nn-foundry
```

### 2. Launch Docker Environment

```bash
# Build the container (first time only)
docker build -t photonic-foundry:latest .

# Start interactive development environment
docker run -it -v $(pwd):/workspace photonic-foundry:latest
```

### 3. Your First Transpilation

Create a simple neural network and convert it to photonic Verilog:

```python
# example.py
from photonic_foundry import PhotonicAccelerator, torch2verilog
import torch.nn as nn

# Define a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Convert to photonic Verilog
verilog_code = torch2verilog(model, target='photonic_mac')

# Initialize accelerator
accelerator = PhotonicAccelerator(
    pdk='skywater130',
    wavelength=1550  # nm
)

# Compile and get performance metrics
results = accelerator.compile_and_profile(verilog_code)

print(f"Energy per operation: {results['energy_per_op']} pJ")
print(f"Latency: {results['latency']} ps")
print(f"Area: {results['area']} μm²")
```

### 4. Run the Example

```bash
# Inside the Docker container
python example.py
```

## Understanding the Output

The transpiler generates several outputs:

### Verilog Code
- **Module definitions**: Photonic MAC units, modulators, detectors
- **Interconnects**: Waveguide routing and optical switches  
- **Control logic**: Electronic interfaces and timing

### Performance Metrics
- **Energy per operation**: Power consumption in picojoules
- **Latency**: Processing delay in picoseconds
- **Area**: Silicon footprint in square micrometers
- **Throughput**: Operations per second

### Simulation Files
- **Testbenches**: For functional verification
- **Power analysis**: Detailed energy breakdown
- **Timing reports**: Critical path analysis

## Common Workflows

### Model Optimization for Photonics

```python
from photonic_foundry import optimize_for_photonics

# Original PyTorch model
original_model = torch.load('my_model.pth')

# Optimize for photonic implementation
optimized_model = optimize_for_photonics(
    original_model,
    target_wavelength=1550,
    power_budget=1000,  # pJ
    area_constraint=500  # μm²
)

# Compare performance
original_results = accelerator.profile(torch2verilog(original_model))
optimized_results = accelerator.profile(torch2verilog(optimized_model))

print(f"Energy improvement: {original_results['energy']/optimized_results['energy']:.1f}x")
```

### Batch Processing

```python
# Process multiple models
models = {
    'resnet18': torchvision.models.resnet18(),
    'mobilenet': torchvision.models.mobilenet_v2(),
    'efficientnet': torchvision.models.efficientnet_b0()
}

results = {}
for name, model in models.items():
    verilog = torch2verilog(model, optimize=True)
    results[name] = accelerator.compile_and_profile(verilog)
    
# Generate comparison report
accelerator.generate_report(results, output='comparison.html')
```

### Custom PDK Configuration

```python
# Use a different foundry process
accelerator = PhotonicAccelerator(
    pdk='globalfoundries45',
    wavelength=1310,  # Different wavelength
    temperature=85,   # Operating temperature (°C)
    process_corners='tt'  # Typical-typical process
)

# PDK-specific optimizations
verilog = torch2verilog(
    model, 
    target='photonic_mac',
    pdk_optimizations=['thermal_stability', 'crosstalk_reduction']
)
```

## Supported Operations

### Currently Supported
- ✅ **Linear layers** (nn.Linear): Matrix multiplication via Mach-Zehnder arrays
- ✅ **ReLU activation**: Electro-absorption modulators
- ✅ **Element-wise operations**: Amplitude modulation
- ✅ **Bias addition**: Electronic offset compensation

### Coming Soon
- 🔄 **Convolutional layers**: Weight-stationary dataflow
- 🔄 **Batch normalization**: Statistical moment computation
- 🔄 **Attention mechanisms**: Parallel correlation computation
- 🔄 **LSTM/GRU**: Temporal processing circuits

### Experimental
- 🧪 **Transformer blocks**: Full attention implementation
- 🧪 **ResNet blocks**: Skip connection optimization
- 🧪 **Custom activations**: Parametric nonlinearities

## Performance Tuning

### Memory Optimization

```python
# Reduce memory usage for large models
verilog = torch2verilog(
    model,
    memory_optimization=True,
    max_buffer_size=1024,  # MB
    streaming_mode=True
)
```

### Parallel Processing

```python
# Utilize multi-wavelength parallelism
accelerator = PhotonicAccelerator(
    pdk='skywater130',
    wavelengths=[1540, 1545, 1550, 1555],  # WDM channels
    parallel_processing=True
)
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'photonic_foundry'`
**Solution**: Ensure you're running inside the Docker container or have installed dependencies locally:
```bash
pip install -r requirements.txt
```

**Issue**: Transpilation fails with "Unsupported operation"
**Solution**: Check the supported operations list above. For custom layers, see the [Advanced Usage Guide](advanced-usage.md).

**Issue**: Poor performance metrics
**Solution**: Try enabling optimizations:
```python
verilog = torch2verilog(model, optimize=True, target_power=100)  # pJ budget
```

### Getting Help

- 📖 **Documentation**: Full API reference at [docs.photonic-foundry.org](https://docs.photonic-foundry.org)
- 💬 **Community**: Join discussions at [GitHub Discussions](https://github.com/yourusername/photonic-nn-foundry/discussions)
- 🐛 **Bug Reports**: Submit issues at [GitHub Issues](https://github.com/yourusername/photonic-nn-foundry/issues)
- ✉️ **Contact**: Email support@photonic-foundry.org

## Next Steps

Once you're comfortable with the basics:

1. **[Advanced Usage Guide](advanced-usage.md)**: Custom operations and optimizations
2. **[Developer Guide](../developer/contributing.md)**: Contributing to the project
3. **[API Reference](../api/index.md)**: Complete function documentation
4. **[Examples Repository](https://github.com/yourusername/photonic-foundry-examples)**: Real-world use cases

Happy photonic computing! 🌟