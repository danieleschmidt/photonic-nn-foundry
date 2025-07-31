# Testing Guide

This document outlines the testing strategy and practices for photonic-nn-foundry.

## Testing Philosophy

Our testing approach follows the testing pyramid:
- **Unit Tests (70%)**: Fast, isolated tests for individual functions and classes
- **Integration Tests (20%)**: Tests for component interactions and workflows
- **End-to-End Tests (10%)**: Full system tests with real-world scenarios

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_core.py        # Core functionality tests
│   ├── test_cli.py         # CLI interface tests
│   └── test_transpiler.py  # Transpiler tests
├── integration/             # Integration tests
│   ├── test_workflows.py   # End-to-end workflows
│   └── test_containers.py  # Docker integration tests
├── performance/             # Performance benchmarks
│   └── test_benchmarks.py  # Performance tests
├── security/               # Security tests
│   └── test_security.py    # Security validation tests
└── conftest.py             # Shared test fixtures
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/photonic_foundry

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m slow             # Slow tests
pytest -m "not slow"       # Exclude slow tests
```

### Advanced Test Options

```bash
# Run tests in parallel
pytest -n auto

# Run with detailed output
pytest -v --tb=long

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::test_photonic_layer_creation

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

### Performance Testing

```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Compare benchmark results
pytest tests/performance/ --benchmark-compare=0001

# Save benchmark results
pytest tests/performance/ --benchmark-save=baseline
```

## Test Categories and Markers

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_basic_function():
    pass

@pytest.mark.integration
def test_component_interaction():
    pass

@pytest.mark.slow
def test_long_running_process():
    pass

@pytest.mark.network
def test_api_call():
    pass

@pytest.mark.gpu
def test_gpu_computation():
    pass

@pytest.mark.security
def test_input_validation():
    pass
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_size,expected", [
    (10, 100),
    (20, 400),
    (30, 900),
])
def test_layer_scaling(input_size, expected):
    result = calculate_layer_size(input_size)
    assert result == expected
```

## Test Fixtures

### Common Fixtures (conftest.py)

```python
import pytest
from photonic_foundry import PhotonicCore

@pytest.fixture
def photonic_core():
    """Provides a configured PhotonicCore instance."""
    return PhotonicCore(config={'precision': 'float32'})

@pytest.fixture
def sample_model():
    """Provides a sample PyTorch model for testing."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

@pytest.fixture(scope="session")
def docker_compose_up():
    """Start Docker services for integration tests."""
    # Implementation for Docker setup
    pass
```

## Mock and Patch Strategies

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch
import pytest

def test_cli_with_mocked_core():
    with patch('photonic_foundry.cli.PhotonicCore') as mock_core:
        mock_core.return_value.process.return_value = "success"
        result = run_cli_command(['convert', 'model.pth'])
        assert result.exit_code == 0
        mock_core.assert_called_once()

@pytest.fixture
def mock_torch_device():
    with patch('torch.cuda.is_available', return_value=False):
        yield
```

## Integration Testing

### Docker Integration Tests

```python
import docker
import pytest

@pytest.fixture(scope="module")
def docker_client():
    return docker.from_env()

def test_container_startup(docker_client):
    container = docker_client.containers.run(
        "photonic-foundry:latest",
        "photonic-foundry --version",
        remove=True,
        capture_output=True
    )
    assert "photonic-foundry" in container.logs().decode()
```

### API Integration Tests

```python
import requests
import pytest

@pytest.mark.integration
def test_api_endpoint():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Test Data Management

### Test Data Structure

```
tests/
├── data/
│   ├── models/              # Sample PyTorch models
│   │   ├── simple_nn.pth
│   │   └── complex_cnn.pth
│   ├── configs/             # Test configurations
│   │   └── test_config.yml
│   └── expected/            # Expected outputs
│       └── verilog_outputs/
```

### Loading Test Data

```python
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_model_path(test_data_dir):
    return test_data_dir / "models" / "simple_nn.pth"
```

## Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Modules**: 90% coverage required
- **New Code**: 100% coverage for new features
- **Documentation**: All public APIs must be documented and tested

### Coverage Exclusions

```python
def debug_function():  # pragma: no cover
    """This function is excluded from coverage."""
    pass

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional
```

## Continuous Integration

### Test Pipeline Stages

1. **Lint and Format**: Code quality checks
2. **Unit Tests**: Fast, isolated tests
3. **Integration Tests**: Component interaction tests
4. **Security Tests**: Security validation
5. **Performance Tests**: Benchmark regression tests
6. **Coverage Report**: Generate and upload coverage

### Test Matrix

- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **Operating Systems**: Ubuntu, macOS, Windows
- **Dependencies**: Minimum and latest versions

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with pdb on failures
pytest --pdb

# Run with verbose output
pytest -vvv

# Run with logging
pytest --log-cli-level=DEBUG

# Run single test with maximum detail
pytest tests/unit/test_core.py::test_function -vvv --tb=long --log-cli-level=DEBUG
```

### Common Issues and Solutions

1. **Import Errors**: Check PYTHONPATH and package installation
2. **Fixture Not Found**: Verify fixture scope and location
3. **Mock Not Working**: Ensure correct import path in patch
4. **Test Isolation**: Use proper setup/teardown or fixture scopes
5. **Flaky Tests**: Add proper waits and deterministic test data

## Performance Testing Guidelines

### Benchmark Test Structure

```python
import pytest

def test_performance_baseline(benchmark):
    result = benchmark(expensive_function, large_input)
    assert result is not None
    
def test_memory_usage():
    import tracemalloc
    tracemalloc.start()
    
    # Test code here
    expensive_operation()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 100 * 1024 * 1024  # Less than 100MB
```

### Performance Thresholds

- **Unit Tests**: < 1 second per test
- **Integration Tests**: < 30 seconds per test
- **Performance Tests**: Baseline + 10% regression threshold
- **Memory Usage**: Monitor for memory leaks and excessive usage

## Security Testing

### Input Validation Tests

```python
@pytest.mark.security
def test_input_validation():
    with pytest.raises(ValueError):
        process_user_input("../../../etc/passwd")
    
    with pytest.raises(ValueError):
        process_user_input("'; DROP TABLE users; --")
```

### Dependency Security Tests

```bash
# Run security scans as part of test suite
pytest tests/security/ --security-scan
```

## Test Reporting

### Coverage Reports

- **HTML Report**: `htmlcov/index.html`
- **XML Report**: `coverage.xml` (for CI integration)
- **Terminal Report**: Displayed during test runs

### Test Results

- **JUnit XML**: For CI/CD integration
- **JSON Report**: For detailed analysis
- **Benchmark Results**: Performance trend tracking

This testing guide ensures comprehensive validation of photonic-nn-foundry across all development phases and deployment environments.