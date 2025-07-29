"""
Photonic Neural Network Foundry

A comprehensive software stack for silicon-photonic AI accelerators.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@photonic-foundry.com"

from .core import PhotonicAccelerator
from .transpiler import torch2verilog

__all__ = ["PhotonicAccelerator", "torch2verilog"]