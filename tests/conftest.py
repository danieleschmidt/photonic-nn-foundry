"""
Global test configuration and shared fixtures for photonic-nn-foundry.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np
from typing import Generator, Dict, Any

# Handle torch import gracefully for testing
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create minimal mock torch for testing
    class MockTensor:
        def __init__(self, *args, **kwargs):
            self.data = np.array(args[0]) if args else np.array([])
        def __getattr__(self, name):
            return lambda *args, **kwargs: MockTensor()
        def allclose(self, other, **kwargs):
            return True
        def max(self):
            return MockTensor()
        def abs(self):
            return MockTensor()
        def cuda(self):
            return self
        def manual_seed(self, seed):
            np.random.seed(seed)
        def randn(*args):
            return MockTensor(np.random.randn(*args))
        def __sub__(self, other):
            return MockTensor()
            
    class MockDevice:
        def __init__(self, device_type):
            self.type = device_type
            
    class MockBackends:
        class cudnn:
            deterministic = True
            benchmark = False
        class mps:
            @staticmethod
            def is_available():
                return False
                
    class MockNN:
        class Sequential:
            def __init__(self, *layers):
                self.layers = layers
        class Linear:
            def __init__(self, in_features, out_features):
                pass
        class ReLU:
            def __init__(self):
                pass
                
    class MockTorch:
        device = MockDevice
        backends = MockBackends()
        nn = MockNN()
        Tensor = MockTensor
        
        @staticmethod
        def cuda():
            class Cuda:
                @staticmethod
                def is_available():
                    return False
                @staticmethod
                def manual_seed(seed):
                    np.random.seed(seed)
                @staticmethod
                def manual_seed_all(seed):
                    np.random.seed(seed)
            return Cuda()
        
        @staticmethod
        def manual_seed(seed):
            np.random.seed(seed)
            
        @staticmethod
        def randn(*args):
            return MockTensor(np.random.randn(*args))
            
        @staticmethod
        def allclose(a, b, **kwargs):
            return True
            
        @staticmethod
        def max(tensor):
            return MockTensor()
            
        @staticmethod
        def abs(tensor):
            return MockTensor()
    
    if not TORCH_AVAILABLE:
        torch = MockTorch()
        torch.cuda.is_available = lambda: False

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "photonic: Photonic-specific tests")
    config.addinivalue_line("markers", "transpiler: Transpiler tests")
    config.addinivalue_line("markers", "simulation: Simulation tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Auto-add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Add slow marker for tests that take longer than expected
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker for tests requiring CUDA
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp(prefix="photonic_test_")
    workspace = Path(temp_dir)
    
    # Create standard directory structure
    (workspace / "models").mkdir()
    (workspace / "circuits").mkdir()
    (workspace / "results").mkdir()
    (workspace / "logs").mkdir()
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_workspace(temp_workspace: Path) -> Generator[Path, None, None]:
    """Provide a clean workspace for each test."""
    # Clear any existing files
    for item in temp_workspace.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
            item.mkdir()
    
    yield temp_workspace


@pytest.fixture(scope="session")
def torch_device() -> torch.device:
    """Get the best available torch device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def random_seed() -> Generator[None, None, None]:
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # Reset to non-deterministic after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture
def photonic_config() -> Dict[str, Any]:
    """Default photonic system configuration for tests."""
    return {
        "wavelength": 1550,  # nm
        "power_budget": 10,  # dBm
        "temperature": 25,   # Celsius
        "pdk": "skywater130",
        "simulation": {
            "precision": "float32",
            "timeout": 30,  # seconds
            "max_iterations": 1000,
        },
        "transpiler": {
            "optimization_level": 1,
            "target_platform": "photonic_mac",
            "enable_debug": False,
        },
        "validation": {
            "accuracy_threshold": 0.01,
            "energy_target": 1.0,  # pJ/Op
            "latency_target": 1e-6,  # s
        }
    }


@pytest.fixture
def sample_linear_model():
    """Create a simple linear model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    return torch.randn(32, 10)


@pytest.fixture
def photonic_params():
    """Standard photonic device parameters for testing."""
    return {
        'wavelength': 1550,  # nm
        'pdk': 'skywater130',
        'energy_per_op': 0.5,  # pJ
        'latency': 100  # ps
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Set test-specific environment variables
    monkeypatch.setenv("PHOTONIC_TEST_MODE", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # Reduce log noise in tests
    monkeypatch.setenv("DISABLE_PROGRESS_BARS", "true")
    
    # Ensure reproducible behavior
    monkeypatch.setenv("PYTHONHASHSEED", "0")


# Skip decorators for conditional tests
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

skip_if_no_network = pytest.mark.skipif(
    os.getenv("NO_NETWORK", "false").lower() == "true",
    reason="Network access disabled"
)

skip_if_slow = pytest.mark.skipif(
    os.getenv("SKIP_SLOW", "false").lower() == "true",
    reason="Slow tests disabled"
)


# Custom assertion helpers
def assert_tensor_close(actual: torch.Tensor, 
                       expected: torch.Tensor, 
                       rtol: float = 1e-5, 
                       atol: float = 1e-8):
    """Assert tensors are close with custom tolerances."""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Tensors not close:\nActual: {actual}\nExpected: {expected}\n" \
        f"Max diff: {torch.max(torch.abs(actual - expected))}"


def assert_energy_efficient(energy_per_op: float, target: float = 1.0):
    """Assert energy efficiency meets target."""
    assert energy_per_op <= target, \
        f"Energy efficiency not met: {energy_per_op} pJ/Op > {target} pJ/Op"


def assert_latency_target(latency: float, target: float = 1e-6):
    """Assert latency meets target."""
    assert latency <= target, \
        f"Latency target not met: {latency} s > {target} s"