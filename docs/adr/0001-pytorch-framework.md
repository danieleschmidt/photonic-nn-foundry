# ADR-0001: Use PyTorch as Primary ML Framework

## Status
Accepted

## Context
The photonic-nn-foundry needs a deep learning framework for neural network model definition and transpilation to photonic circuits. Key requirements:
- Mature ecosystem with extensive community support
- Dynamic computation graphs for flexible model design
- Strong support for custom operators and extensions
- Python-first API for research productivity
- Active development and long-term viability

Alternatives considered:
- TensorFlow: More complex deployment, less research-friendly API
- JAX: Smaller ecosystem, less mature for production
- Custom framework: Significant development overhead, limited community

## Decision
Use PyTorch 2.0+ as the primary machine learning framework for:
- Neural network model definitions
- Pre-trained model loading and manipulation
- Gradient computation and training (when applicable)
- Integration with the transpiler engine

## Consequences

### Positive
- Rich ecosystem of pre-trained models via torch.hub and Hugging Face
- Dynamic graphs enable complex photonic circuit generation patterns
- Strong community support and extensive documentation
- Native CUDA support for GPU acceleration during transpilation
- TorchScript compilation for optimized model serialization

### Negative
- Additional dependency complexity compared to pure Python implementation
- Some overhead for purely inference-based photonic applications
- Potential version compatibility issues with other deep learning libraries

### Neutral
- Requires team familiarity with PyTorch API patterns
- Standard dependency management practices apply