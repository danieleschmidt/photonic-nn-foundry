# photonic-nn-foundry

A pure-NumPy software stack for silicon-photonic AI accelerators.

Model optical neural networks built from **Mach-Zehnder Interferometer (MZI)** meshes —
the same hardware used by MIT, Stanford, and Lightmatter for optical matrix multiplication.

## What's Inside

| Module | Description |
|--------|-------------|
| `mzi.py` | `MachZehnderInterferometer` — 2×2 unitary optical element |
| `mesh.py` | `PhotonicMeshLayer` — N×N unitary mesh (Clements topology) |
| `linear.py` | `PhotonicLinear` — optical matrix multiply via SVD decomposition |
| `network.py` | `PhotonicNet` — 2-layer optical neural network with SPSA training |

## Quick Start

```bash
pip install -e .
python examples/demo.py
```

## Core Concepts

**MZI:** A beamsplitter with two phase shifters. Parameterized by `theta` (splitting ratio) and `phi` (output phase). Implements a 2×2 unitary:

```
T = [[e^(iφ)·cos(θ/2),  -sin(θ/2)],
     [e^(iφ)·sin(θ/2),   cos(θ/2)]]
```

**Mesh:** A rectangular array of MZIs (Clements 2016) implementing any N×N unitary. Used as the optical weight matrix in neural network layers.

**PhotonicLinear:** Implements W = U·Σ·V† (optical SVD). Two unitary meshes + amplitude weights ≈ arbitrary matrix multiply.

**PhotonicNet:** Stacks two `PhotonicLinear` layers with an electronic ReLU between them. Trains using SPSA (hardware-friendly finite-difference gradient estimation).

## Usage

```python
from photonic_nn_foundry import MachZehnderInterferometer, PhotonicNet
import numpy as np

# Single MZI
mzi = MachZehnderInterferometer(theta=np.pi/4, phi=0.0)
T = mzi.transmission_matrix()  # 2×2 unitary

# Full photonic network
net = PhotonicNet(input_size=784, hidden_size=128, output_size=10)
logits = net.forward(np.random.randn(32, 784))  # batch forward pass
loss = net.train_step(X, y, lr=0.01)
```

## Tests

```bash
pytest tests/ -v
```

## Requirements

- Python 3.9+
- NumPy

No PyTorch, no CUDA. Just optics.
