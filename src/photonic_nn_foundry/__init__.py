"""
photonic-nn-foundry: Silicon-photonic AI accelerator software stack.

A pure-NumPy library for modeling optical neural networks built from
Mach-Zehnder Interferometer (MZI) meshes — the core building block of
silicon-photonic matrix multiply hardware.
"""

from .mzi import MachZehnderInterferometer
from .mesh import PhotonicMeshLayer
from .linear import PhotonicLinear
from .network import PhotonicNet, relu, softmax, cross_entropy_loss

__all__ = [
    "MachZehnderInterferometer",
    "PhotonicMeshLayer",
    "PhotonicLinear",
    "PhotonicNet",
    "relu",
    "softmax",
    "cross_entropy_loss",
]

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
