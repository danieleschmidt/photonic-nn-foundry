# Architecture Documentation

## Overview

Photonic-nn-foundry is a Python-based framework for developing and deploying silicon-photonic neural network accelerators. The architecture follows a modular design with clear separation of concerns.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CLI Interface (Click)                  │
├─────────────────────────────────────────────────────────┤
│                 Core Framework Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Model     │  │  Photonic   │  │ Transpiler  │    │
│  │   Parser    │  │    Core     │  │   Engine    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────┤
│                Hardware Abstraction Layer               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Photonic   │  │   Memory    │  │ Compute     │    │
│  │  Circuits   │  │ Management  │  │  Units      │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────┤
│                   Output Generation                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Verilog   │  │ Test Bench  │  │   FPGA      │    │
│  │ Generation  │  │ Generation  │  │  Bitstream  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. CLI Interface (`cli.py`)

**Purpose**: Command-line interface for user interactions

**Responsibilities**:
- Command parsing and validation
- User input/output handling
- Error reporting and logging
- Configuration management

**Key Components**:
```python
@click.group()
def cli():
    """Main CLI entry point"""

@cli.command()
def convert(model_path: str, output_dir: str):
    """Convert PyTorch model to photonic hardware"""

@cli.command()
def simulate(config_path: str):
    """Simulate photonic neural network"""
```

### 2. Core Framework (`core.py`)

**Purpose**: Central orchestration and business logic

**Responsibilities**:
- Model loading and validation
- Photonic circuit design
- Resource allocation and optimization
- Integration between components

**Key Classes**:
```python
class PhotonicCore:
    """Central coordinator for photonic neural networks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.transpiler = PhotonicTranspiler()
        self.optimizer = CircuitOptimizer()
    
    def process_model(self, model: torch.nn.Module) -> PhotonicCircuit:
        """Convert PyTorch model to photonic circuit"""
        
    def simulate(self, circuit: PhotonicCircuit) -> SimulationResults:
        """Simulate photonic circuit behavior"""
```

### 3. Transpiler Engine (`transpiler.py`)

**Purpose**: Convert PyTorch models to photonic hardware descriptions

**Responsibilities**:
- Neural network layer analysis
- Photonic component mapping
- Hardware description language generation
- Optimization pass management

**Key Classes**:
```python
class PhotonicTranspiler:
    """Converts neural networks to photonic circuits"""
    
    def transpile(self, model: torch.nn.Module) -> str:
        """Generate Verilog from PyTorch model"""
    
    def optimize_circuit(self, circuit: Circuit) -> Circuit:
        """Apply circuit-level optimizations"""
```

## Data Flow Architecture

### 1. Model Processing Pipeline

```
PyTorch Model → Model Parser → Layer Analysis → Photonic Mapping → Circuit Generation → Verilog Output
      ↓              ↓              ↓               ↓                ↓               ↓
   Validation → Type Checking → Resource → Component → Optimization → Testing
                                Planning    Selection
```

### 2. Configuration Flow

```
CLI Args → Config Validation → Parameter Resolution → Component Configuration → Execution
    ↓           ↓                    ↓                      ↓                  ↓
User Input → Schema Check → Default Merging → Object Creation → Processing
```

## Module Dependencies

### Internal Dependencies

```
cli.py
├── core.py
│   ├── transpiler.py
│   └── utils/
│       ├── validators.py
│       ├── optimizers.py
│       └── generators.py
└── config/
    ├── settings.py
    └── schemas.py
```

### External Dependencies

```
Core Framework:
├── PyTorch (>= 2.0.0)    # Neural network framework
├── NumPy (>= 1.21.0)     # Numerical computing
├── Pydantic (>= 2.0.0)   # Data validation
└── Click (>= 8.0.0)      # CLI framework

Development:
├── pytest (>= 7.0.0)     # Testing framework
├── black (>= 23.0.0)     # Code formatting
├── mypy (>= 1.0.0)       # Type checking
└── pre-commit            # Git hooks
```

## Design Patterns

### 1. Strategy Pattern

Used for different photonic component implementations:

```python
class PhotonicLayer(ABC):
    @abstractmethod
    def generate_verilog(self) -> str:
        pass

class MZILayer(PhotonicLayer):
    def generate_verilog(self) -> str:
        return "// MZI implementation"

class RingResonatorLayer(PhotonicLayer):
    def generate_verilog(self) -> str:
        return "// Ring resonator implementation"
```

### 2. Factory Pattern

For creating photonic components based on neural network layers:

```python
class PhotonicComponentFactory:
    @staticmethod
    def create_component(layer_type: str) -> PhotonicLayer:
        if layer_type == "Linear":
            return MZILayer()
        elif layer_type == "Conv2d":
            return PhotonicConvLayer()
        else:
            raise ValueError(f"Unsupported layer: {layer_type}")
```

### 3. Builder Pattern

For complex photonic circuit construction:

```python
class CircuitBuilder:
    def __init__(self):
        self.circuit = PhotonicCircuit()
    
    def add_input_layer(self, size: int) -> 'CircuitBuilder':
        self.circuit.add_layer(InputLayer(size))
        return self
    
    def add_mzi_layer(self, config: Dict) -> 'CircuitBuilder':
        self.circuit.add_layer(MZILayer(config))
        return self
    
    def build(self) -> PhotonicCircuit:
        return self.circuit
```

## Error Handling Architecture

### 1. Exception Hierarchy

```python
class PhotonicFoundryError(Exception):
    """Base exception for all photonic foundry errors"""

class ModelValidationError(PhotonicFoundryError):
    """Raised when model validation fails"""

class TranspilerError(PhotonicFoundryError):
    """Raised during transpilation process"""

class CircuitOptimizationError(PhotonicFoundryError):
    """Raised during circuit optimization"""

class HardwareConstraintError(PhotonicFoundryError):
    """Raised when hardware constraints are violated"""
```

### 2. Error Propagation

```
User Input → Validation → Processing → Output Generation
     ↓           ↓           ↓            ↓
Error Catch → Log Error → Cleanup → User Feedback
```

## Performance Architecture

### 1. Optimization Strategies

- **Lazy Loading**: Load models and configurations on-demand
- **Caching**: Cache transpilation results for repeated operations
- **Parallel Processing**: Utilize multi-core processing for large models
- **Memory Management**: Efficient memory usage for large neural networks

### 2. Profiling Points

```python
import time
from functools import wraps

def profile_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## Security Architecture

### 1. Input Validation

- **Schema Validation**: Use Pydantic for strict input validation
- **Path Sanitization**: Validate file paths to prevent directory traversal
- **Resource Limits**: Implement limits on model size and complexity

### 2. Secure Defaults

```python
DEFAULT_CONFIG = {
    "max_model_size_mb": 100,
    "max_layers": 1000,
    "allowed_layer_types": ["Linear", "Conv2d", "ReLU", "BatchNorm2d"],
    "output_sanitization": True,
}
```

## Testing Architecture

### 1. Test Structure

```
tests/
├── unit/                 # Component-level tests
│   ├── test_core.py
│   ├── test_cli.py
│   └── test_transpiler.py
├── integration/          # System-level tests
│   ├── test_workflows.py
│   └── test_end_to_end.py
└── fixtures/            # Test data and utilities
    ├── models/
    └── configs/
```

### 2. Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **Security Tests**: Validate input handling and security measures

## Deployment Architecture

### 1. Container Structure

```dockerfile
# Multi-stage build for different deployment targets
FROM python:3.10-slim as base
# Base dependencies and common setup

FROM base as development
# Development tools and debugging capabilities

FROM base as production
# Minimal production environment

FROM base as testing
# Testing tools and test data
```

### 2. Configuration Management

- **Environment Variables**: Runtime configuration
- **Config Files**: Static configuration and defaults
- **CLI Arguments**: User-specific overrides

## Extension Points

### 1. Plugin Architecture

```python
class PhotonicPlugin(ABC):
    @abstractmethod
    def register_components(self) -> List[PhotonicComponent]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
```

### 2. Custom Layer Support

```python
def register_custom_layer(layer_name: str, implementation: Type[PhotonicLayer]):
    """Register custom photonic layer implementation"""
    PhotonicComponentFactory.register(layer_name, implementation)
```

## Monitoring and Observability

### 1. Logging Architecture

```python
import logging

# Structured logging with context
logger = logging.getLogger(__name__)

def log_operation(operation: str, context: Dict):
    logger.info(f"Operation: {operation}", extra={"context": context})
```

### 2. Metrics Collection

- **Performance Metrics**: Execution time, memory usage
- **Usage Metrics**: Command frequency, model types
- **Error Metrics**: Error rates, error types

## Data Flow Diagrams

### Model Processing Data Flow

```mermaid
graph TB
    A[PyTorch Model] --> B[Model Analyzer]
    B --> C[Layer Decomposer]
    C --> D[Photonic Mapper]
    D --> E[Circuit Generator]
    E --> F[Verilog Synthesizer]
    F --> G[Hardware Description]
    
    H[Configuration] --> D
    I[PDK Library] --> E
    J[Optimization Rules] --> E
```

### System Component Interactions

```mermaid
graph LR
    CLI[CLI Interface] --> Core[Photonic Core]
    Core --> Trans[Transpiler Engine]
    Core --> Sim[Simulator]
    Core --> Prof[Profiler]
    
    Trans --> MZI[MZI Components]
    Trans --> Ring[Ring Resonators]
    Trans --> WG[Waveguides]
    
    Sim --> OpSim[Optical Simulation]
    Sim --> ESim[Electrical Simulation]
    
    Prof --> Energy[Energy Analysis]
    Prof --> Timing[Timing Analysis]
```

### Advanced Photonic Algorithm Pipeline

```mermaid
graph TD
    A[Neural Network Model] --> B[Photonic Compatibility Analysis]
    B --> C{Compatible?}
    C -->|Yes| D[Photonic Component Mapping]
    C -->|No| E[Model Transformation]
    E --> D
    
    D --> F[MZI Mesh Synthesis]
    D --> G[Ring Resonator Design]
    D --> H[Waveguide Routing]
    
    F --> I[Phase Optimization]
    G --> J[Resonance Tuning]
    H --> K[Loss Minimization]
    
    I --> L[Circuit Integration]
    J --> L
    K --> L
    
    L --> M[Performance Validation]
    M --> N[Verilog Generation]
```

### Database and Caching Architecture

```mermaid
graph LR
    subgraph "Application Layer"
        A[PhotonicAccelerator]
        B[API Server]
        C[CLI Interface]
    end
    
    subgraph "Data Access Layer"
        D[Circuit Repository]
        E[Model Repository]
        F[Metrics Repository]
    end
    
    subgraph "Caching Layer"
        G[Circuit Cache]
        H[Model Cache]
        I[Computation Cache]
    end
    
    subgraph "Storage Layer"
        J[(SQLite/PostgreSQL)]
        K[(Redis/Memory)]
    end
    
    A --> D
    B --> E
    C --> F
    
    D --> G
    E --> H
    F --> I
    
    G --> J
    H --> J
    I --> K
```

## Advanced Photonic Computing Algorithms

### 1. Mach-Zehnder Interferometer (MZI) Mesh Optimization

The core of photonic neural networks relies on MZI meshes for matrix multiplication:

```python
def optimize_mzi_mesh(weight_matrix: np.ndarray, precision: int = 8) -> Dict[str, Any]:
    """
    Optimize MZI mesh configuration for given weight matrix.
    
    Key optimizations:
    - Phase shifter quantization
    - Insertion loss minimization  
    - Crosstalk mitigation
    - Power consumption optimization
    """
```

### 2. Ring Resonator Weight Storage

Advanced weight storage using ring resonator thermal tuning:

```python
def design_resonator_bank(weights: np.ndarray, q_factor: float = 10000) -> List[Dict]:
    """
    Design ring resonator bank for weight storage.
    
    Features:
    - Thermal tuning for weight updates
    - High Q-factor for sharp resonances
    - Wavelength division multiplexing
    """
```

### 3. Photonic Activation Functions

Non-linear activation functions using electro-optic effects:

```python
def photonic_activation_synthesis(activation_type: str) -> Dict[str, Any]:
    """
    Synthesize photonic activation function circuits.
    
    Supported activations:
    - ReLU via amplitude clipping
    - Sigmoid via Mach-Zehnder nonlinearity
    - Tanh via dual-rail encoding
    """
```

## System Performance Modeling

### Energy Consumption Model

```python
def calculate_photonic_energy(circuit: PhotonicCircuit) -> float:
    """
    Energy consumption model for photonic circuits.
    
    Components:
    - Laser power (dominant factor)
    - Thermal tuning power
    - Electronic control power
    - Photodetector power
    """
    base_laser_power = 10e-3  # 10 mW baseline
    mzi_count = sum(len([c for c in layer.components 
                        if c['type'] == PhotonicComponent.MZI]) 
                   for layer in circuit.layers)
    
    # Laser power scales with MZI count
    laser_power = base_laser_power * (1 + 0.1 * mzi_count)
    
    # Thermal tuning power (per MZI)
    thermal_power = mzi_count * 0.5e-3  # 0.5 mW per MZI
    
    # Electronic control overhead
    control_power = 1e-3  # 1 mW baseline
    
    return laser_power + thermal_power + control_power
```

### Latency Analysis Model

```python
def analyze_photonic_latency(circuit: PhotonicCircuit) -> Dict[str, float]:
    """
    Comprehensive latency analysis for photonic circuits.
    
    Latency components:
    - Optical propagation delay
    - Electronic processing delay
    - Thermal settling time
    - Photodetection time
    """
    layer_count = len(circuit.layers)
    
    # Speed of light in silicon: ~c/3.5
    propagation_delay = layer_count * 10e-12  # 10 ps per layer
    
    # Electronic ADC/DAC delays
    conversion_delay = 50e-12  # 50 ps
    
    # Thermal settling (if dynamic reconfiguration)
    thermal_delay = 1e-6 if circuit.requires_thermal_tuning else 0
    
    return {
        'optical_propagation': propagation_delay,
        'electronic_conversion': conversion_delay,
        'thermal_settling': thermal_delay,
        'total_latency': propagation_delay + conversion_delay + thermal_delay
    }
```

## Scalability Architecture

### Multi-Chip Photonic Systems

```mermaid
graph TB
    subgraph "Photonic Chip 1"
        A1[Input Layer]
        B1[Hidden Layer 1]
        C1[Output Buffer]
    end
    
    subgraph "Photonic Chip 2"
        A2[Input Buffer]
        B2[Hidden Layer 2]
        C2[Output Layer]
    end
    
    subgraph "Electronic Controller"
        D[Centralized Control]
        E[Data Orchestration]
        F[Thermal Management]
    end
    
    C1 --> A2
    D --> A1
    D --> A2
    E --> B1
    E --> B2
    F --> C1
    F --> C2
```

This enhanced architecture provides a comprehensive foundation for scalable, high-performance photonic neural network systems with enterprise-grade reliability and maintainability.