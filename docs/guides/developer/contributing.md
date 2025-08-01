# Developer Contributing Guide

Thank you for your interest in contributing to the Photonic Neural Network Foundry! This guide will help you get started with development and understand our contribution process.

## Development Environment Setup

### Prerequisites

- **Git** with SSH keys configured for GitHub
- **Docker Desktop** or Docker Engine 20.10+
- **Python 3.8+** for local development
- **VS Code** (recommended) with Python extension

### Local Development Setup

1. **Fork and Clone**
   ```bash
   git clone git@github.com:YOUR_USERNAME/photonic-nn-foundry.git
   cd photonic-nn-foundry
   git remote add upstream git@github.com:yourusername/photonic-nn-foundry.git
   ```

2. **Development Container** (Recommended)
   ```bash
   # VS Code with Dev Containers extension
   code .
   # When prompted, select "Reopen in Container"
   ```

3. **Local Python Environment** (Alternative)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Verify Setup**
   ```bash
   pytest tests/
   python -m photonic_foundry --version
   ```

## Code Architecture

### Package Structure
```
src/photonic_foundry/
├── __init__.py          # Public API exports
├── core.py              # PhotonicAccelerator class
├── transpiler.py        # PyTorch to Verilog conversion
├── cli.py               # Command-line interface
├── optimization/        # Performance optimization passes
├── pdk/                 # Process Design Kit interfaces
├── simulation/          # Photonic simulation engines
└── utils/               # Shared utilities
```

### Key Components

#### Transpiler Engine (`transpiler.py`)
- **Purpose**: Convert PyTorch operations to photonic Verilog
- **Key Classes**: `VerilogTranspiler`, `OperationMapper`, `OptimizationPass`
- **Extension Points**: Custom operation handlers, PDK-specific optimizations

#### Core Accelerator (`core.py`)
- **Purpose**: Main user interface for photonic acceleration
- **Key Classes**: `PhotonicAccelerator`, `PerformanceProfiler`
- **Extension Points**: Custom PDKs, simulation backends

#### CLI Interface (`cli.py`)
- **Purpose**: Command-line tools for batch processing
- **Key Functions**: Model transpilation, batch profiling, report generation
- **Extension Points**: New commands, output formats

## Contribution Workflow

### 1. Issue First Approach

Before starting significant work:
1. Check existing [issues](https://github.com/yourusername/photonic-nn-foundry/issues)
2. Create a new issue if none exists
3. Discuss approach with maintainers
4. Get assignment or approval to proceed

### 2. Development Process

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes with commits following conventional commits
git add .
git commit -m "feat(transpiler): add support for conv2d layers"

# Push and create pull request
git push origin feature/your-feature-name
```

### 3. Code Quality Standards

#### Formatting and Linting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run all quality checks
make lint
```

#### Testing Requirements
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/photonic_foundry --cov-report=html

# Performance tests
pytest tests/performance/ --benchmark-only
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks (done automatically in dev container)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Development Guidelines

### Code Style

1. **Python Style**: Follow PEP 8, enforced by Black and Flake8
2. **Type Hints**: Required for all public APIs
3. **Docstrings**: Google-style docstrings for all public functions
4. **Error Handling**: Use specific exceptions with descriptive messages

Example:
```python
def torch2verilog(
    model: torch.nn.Module,
    target: str = 'photonic_mac',
    optimize: bool = False
) -> str:
    """Convert PyTorch model to photonic Verilog implementation.
    
    Args:
        model: PyTorch neural network model to convert
        target: Target photonic architecture ('photonic_mac', 'mzi_array')
        optimize: Whether to apply optimization passes
        
    Returns:
        Generated Verilog code as string
        
    Raises:
        UnsupportedOperationError: If model contains unsupported operations
        PDKCompatibilityError: If target incompatible with current PDK
    """
```

### Testing Strategy

#### Test Categories
1. **Unit Tests**: Test individual functions and classes
   - Location: `tests/unit/`
   - Coverage: > 90% for new code
   - Speed: < 1s per test

2. **Integration Tests**: Test component interactions
   - Location: `tests/integration/`
   - Coverage: Key user workflows
   - Speed: < 30s per test

3. **Performance Tests**: Benchmark critical paths
   - Location: `tests/performance/`
   - Coverage: Transpilation speed, memory usage
   - Speed: < 5min per test

#### Writing Tests
```python
import pytest
from photonic_foundry import torch2verilog
import torch.nn as nn

def test_linear_layer_transpilation():
    """Test transpilation of simple linear layer."""
    model = nn.Linear(10, 5)
    verilog = torch2verilog(model)
    
    assert 'module photonic_mac' in verilog
    assert 'input [9:0] data_in' in verilog
    assert 'output [4:0] data_out' in verilog

@pytest.mark.performance
def test_transpilation_speed():
    """Test transpilation performance for ResNet-18."""
    model = torchvision.models.resnet18()
    
    start_time = time.time()
    verilog = torch2verilog(model)
    duration = time.time() - start_time
    
    assert duration < 10.0  # Should complete in < 10 seconds
```

### Documentation

#### Updating Documentation
- **API Changes**: Update docstrings and regenerate API docs
- **New Features**: Add examples to user guides
- **Architecture Changes**: Update ADRs and architecture docs

#### Building Documentation Locally
```bash
# Install docs dependencies
pip install -r docs/requirements.txt

# Build and serve locally
cd docs/
make html
python -m http.server 8000 -d _build/html/
```

## Contribution Areas

### High-Priority Areas
1. **Operation Support**: Add transpilation for new PyTorch operations
2. **PDK Integration**: Support for additional foundry processes
3. **Optimization Passes**: Performance and area optimization algorithms
4. **Documentation**: User guides, tutorials, and examples

### Medium-Priority Areas
1. **Testing**: Expand test coverage and add performance benchmarks
2. **CLI Enhancements**: New commands and output formats
3. **Simulation**: Integration with additional photonic simulators
4. **Tooling**: Development and build process improvements

### Specialized Areas
1. **Hardware Validation**: Test with real photonic hardware
2. **Standards**: Contribute to IEEE photonic computing standards
3. **Research**: Novel algorithms and architectural innovations
4. **Community**: Workshops, tutorials, and outreach programs

## Pull Request Process

### PR Checklist
- [ ] Code follows style guidelines and passes linting
- [ ] Tests added/updated and all tests pass
- [ ] Documentation updated for API changes
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Commit messages follow conventional commits format

### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Performance Impact
Describe any performance implications.

## Related Issues
Closes #123
```

### Review Process
1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Manual testing for significant features
4. **Documentation**: Verify docs are complete and accurate

## Release Process

### Versioning
- **Semantic Versioning**: MAJOR.MINOR.PATCH
- **Pre-releases**: alpha, beta, rc suffixes
- **Release Cadence**: Monthly minor releases, patches as needed

### Release Preparation
1. Update CHANGELOG.md with release notes
2. Update version in pyproject.toml
3. Create release PR and get approval
4. Tag release and create GitHub release
5. Automated PyPI publication via GitHub Actions

## Getting Help

### Communication Channels
- **GitHub Discussions**: General questions and design discussions
- **GitHub Issues**: Bug reports and feature requests
- **Monthly Community Calls**: Live discussion of roadmap and contributions
- **Developer Slack**: Real-time development discussion (invite link in main README)

### Maintainer Contact
- **Technical Questions**: Create GitHub issue with `question` label
- **Contribution Planning**: Comment on existing issues or create discussion
- **Urgent Issues**: Email developers@photonic-foundry.org

### Recognition
Contributors are recognized through:
- GitHub contributors page
- Quarterly contributor highlights in newsletter
- Conference presentation opportunities
- Co-authorship on academic publications (where appropriate)

## Code of Conduct

All contributors must adhere to our [Code of Conduct](../../CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for all contributors regardless of background, experience level, or identity.

Thank you for contributing to the future of photonic computing! 🚀