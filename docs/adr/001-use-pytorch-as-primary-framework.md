# ADR 001: Use PyTorch as Primary Deep Learning Framework

## Status

Accepted

## Context

The photonic-nn-foundry needs a robust deep learning framework to:
- Parse neural network architectures for transpilation
- Provide familiar APIs for researchers and engineers
- Support efficient tensor operations for validation
- Enable seamless integration with existing ML workflows

Key considerations:
- Framework maturity and community support
- Compatibility with photonic computing concepts
- Ease of model parsing and analysis
- Performance for validation workloads

## Decision

We will use PyTorch as the primary deep learning framework for the photonic-nn-foundry.

### Rationale

1. **Dynamic Computation Graphs**: PyTorch's dynamic nature makes it easier to analyze and transpile models at runtime
2. **Research Community Adoption**: Widely used in academic research, our primary target audience
3. **Model Introspection**: Excellent capabilities for examining model structure and parameters
4. **Ecosystem**: Rich ecosystem of tools and libraries for model analysis
5. **TorchScript**: Provides static graph representation when needed for optimization

## Consequences

### Positive

- Familiar API for most ML researchers and engineers
- Rich ecosystem of pre-trained models and utilities
- Strong community support and documentation
- Good performance for validation and testing workflows
- Excellent debugging and profiling tools

### Negative

- Additional dependency in the containerized environment
- Need to maintain compatibility with PyTorch version updates
- Some overhead for users primarily using other frameworks (TensorFlow, JAX)

### Mitigation

- Provide clear documentation for users coming from other frameworks
- Implement framework-agnostic intermediate representations where possible
- Consider adding support for ONNX models in future versions