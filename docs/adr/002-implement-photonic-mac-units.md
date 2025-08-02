# ADR 002: Implement Photonic MAC Units as Core Computing Primitive

## Status

Accepted

## Context

The photonic neural network foundry needs to define its core computing primitives. Traditional electronic processors use arithmetic logic units (ALUs), while photonic processors can leverage optical interference for multiply-accumulate (MAC) operations.

Key considerations:
- Energy efficiency of photonic operations
- Compatibility with neural network operations
- Implementation complexity in silicon photonics
- Scalability and routing constraints

Research context:
- MIT demonstrations showing sub-pJ/Op energy consumption
- Mach-Zehnder interferometer-based MAC units
- Wavelength division multiplexing for parallel operations

## Decision

We will implement photonic Multiply-Accumulate (MAC) units as the core computing primitive, based on Mach-Zehnder interferometer arrays.

### Architecture

1. **Optical MAC Units**: Utilize optical interference for multiplication
2. **Wavelength Parallelism**: Multiple wavelengths for parallel operations
3. **Balanced Detection**: Convert optical signals to electrical domain
4. **Modular Design**: Hierarchical composition of MAC units

## Consequences

### Positive

- Ultra-low energy consumption (sub-pJ/Op demonstrated)
- Inherent parallelism through wavelength division multiplexing
- High bandwidth potential (THz optical frequencies)
- Natural fit for linear algebra operations in neural networks
- Reduced heat generation compared to electronic computation

### Negative

- Complexity in optical-to-electrical conversion
- Limited precision compared to digital electronics
- Sensitivity to temperature and fabrication variations
- Requirement for sophisticated control electronics
- Challenges in implementing non-linear operations

### Technical Requirements

1. **Process Design Kit (PDK)**: Support for standard silicon photonics processes
2. **Wavelength Management**: Stable laser sources and wavelength control
3. **Modulation**: High-speed electro-optic modulators
4. **Detection**: Balanced photodetectors with low noise
5. **Routing**: Optical waveguide routing with minimal loss

### Implementation Strategy

1. Start with simple MAC operations (dot products)
2. Validate against electronic implementations
3. Characterize noise and precision limitations
4. Develop calibration and compensation techniques
5. Scale to larger arrays and more complex operations