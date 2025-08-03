"""
Simple Multi-Layer Perceptron models optimized for photonic implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SimpleMLP(nn.Module):
    """
    Simple MLP with only linear layers and ReLU activations.
    Optimized for photonic implementation using MZI arrays.
    """
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [256, 128], 
                 num_classes: int = 10, dropout: float = 0.0):
        """
        Initialize SimpleMLP.
        
        Args:
            input_size: Input feature dimension
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability (0.0 = no dropout)
        """
        super(SimpleMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for better photonic compatibility
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with small values suitable for photonic implementation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier uniform initialization with smaller scale
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                    
    def forward(self, x):
        """Forward pass through the network."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        return self.network(x)
        
    def get_photonic_complexity(self) -> dict:
        """Calculate photonic implementation complexity."""
        total_mzis = 0
        total_params = 0
        layer_info = []
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                mzis = module.in_features * module.out_features
                params = mzis + (module.out_features if module.bias is not None else 0)
                
                total_mzis += mzis
                total_params += params
                
                layer_info.append({
                    'name': name,
                    'type': 'Linear',
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'mzis_required': mzis,
                    'parameters': params
                })
                
        return {
            'total_mzis': total_mzis,
            'total_parameters': total_params,
            'layer_details': layer_info,
            'estimated_area_mm2': total_mzis * 0.001,  # 1μm² per MZI
            'estimated_power_mw': total_mzis * 0.1,    # 0.1mW per MZI
        }


class PhotonicOptimizedMLP(SimpleMLP):
    """
    MLP specifically optimized for photonic implementation.
    Uses specific layer sizes that map efficiently to photonic hardware.
    """
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        """
        Initialize with photonic-friendly dimensions.
        
        Args:
            input_size: Input feature dimension
            num_classes: Number of output classes
        """
        # Use power-of-2 hidden sizes for efficient photonic implementation
        hidden_sizes = [512, 256] if input_size >= 512 else [256, 128]
        
        super(PhotonicOptimizedMLP, self).__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
            dropout=0.0  # No dropout for hardware implementation
        )
        
    def _initialize_weights(self):
        """Initialize weights with photonic-specific constraints."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Constrain weights to [-1, 1] range for better photonic mapping
                nn.init.uniform_(module.weight, -0.8, 0.8)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)


class BinaryMLP(nn.Module):
    """
    Binary MLP using sign activation for ultra-efficient photonic implementation.
    """
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [256, 128], 
                 num_classes: int = 10):
        super(BinaryMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build layers with binary activations
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        
        self._initialize_binary_weights()
        
    def _initialize_binary_weights(self):
        """Initialize weights to binary values {-1, +1}."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize to binary values
                weight_data = torch.sign(torch.randn_like(module.weight))
                module.weight.data = weight_data
                if module.bias is not None:
                    module.bias.data.zero_()
                    
    def binary_activation(self, x):
        """Binary activation function."""
        return torch.sign(x)
        
    def forward(self, x):
        """Forward pass with binary activations."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        for layer in self.layers:
            x = layer(x)
            x = self.binary_activation(x)
            
        x = self.output_layer(x)
        return x
        
    def get_photonic_complexity(self) -> dict:
        """Calculate complexity for binary photonic implementation."""
        complexity = super(SimpleMLP, self).get_photonic_complexity() if hasattr(super(), 'get_photonic_complexity') else {}
        
        # Binary implementation reduces complexity
        if 'total_mzis' in complexity:
            complexity['total_mzis'] = complexity['total_mzis'] // 2  # Binary weights reduce MZI count
            complexity['estimated_power_mw'] = complexity['estimated_power_mw'] * 0.3  # Much lower power
            
        return complexity


def create_mnist_mlp() -> SimpleMLP:
    """Create MLP optimized for MNIST classification."""
    return PhotonicOptimizedMLP(input_size=784, num_classes=10)


def create_cifar10_mlp() -> SimpleMLP:
    """Create MLP for CIFAR-10 classification."""
    return PhotonicOptimizedMLP(input_size=3072, num_classes=10)  # 32*32*3


def create_binary_mnist_mlp() -> BinaryMLP:
    """Create binary MLP for MNIST classification."""
    return BinaryMLP(input_size=784, hidden_sizes=[512, 256], num_classes=10)


if __name__ == "__main__":
    # Example usage
    model = create_mnist_mlp()
    print(f"Model: {model}")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 784)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Show photonic complexity
    complexity = model.get_photonic_complexity()
    print(f"\nPhotonic Implementation Complexity:")
    for key, value in complexity.items():
        if key != 'layer_details':
            print(f"  {key}: {value}")
            
    print(f"\nLayer Details:")
    for layer in complexity['layer_details']:
        print(f"  {layer['name']}: {layer['input_size']} -> {layer['output_size']} ({layer['mzis_required']} MZIs)")