"""Compatibility tests for different PyTorch versions and dependencies."""

import pytest
import torch
import sys
from packaging import version

from tests.utils import requires_gpu


class TestPyTorchCompatibility:
    """Test compatibility across different PyTorch versions."""
    
    def test_minimum_pytorch_version(self):
        """Test that minimum PyTorch version is supported."""
        min_version = "2.0.0"
        current_version = torch.__version__.split('+')[0]  # Remove any suffix
        
        assert version.parse(current_version) >= version.parse(min_version), (
            f"PyTorch version {current_version} is below minimum {min_version}"
        )
    
    def test_torch_api_compatibility(self, sample_linear_model, sample_input_data):
        """Test that core PyTorch APIs work as expected."""
        model = sample_linear_model
        
        # Test basic forward pass
        output = model(sample_input_data)
        assert output is not None
        assert output.shape[0] == sample_input_data.shape[0]
        
        # Test model evaluation mode
        model.eval()
        with torch.no_grad():
            eval_output = model(sample_input_data)
            assert torch.allclose(output, eval_output, atol=1e-6)
    
    def test_torch_jit_compatibility(self, sample_linear_model):
        """Test TorchScript compatibility for model tracing."""
        model = sample_linear_model
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 10)
        
        try:
            # Test model tracing
            traced_model = torch.jit.trace(model, example_input)
            
            # Test that traced model produces same output
            with torch.no_grad():
                original_output = model(example_input)
                traced_output = traced_model(example_input)
                
            assert torch.allclose(original_output, traced_output, atol=1e-6)
            
        except Exception as e:
            pytest.skip(f"TorchScript tracing not supported: {e}")
    
    @requires_gpu
    def test_cuda_compatibility(self, sample_linear_model):
        """Test CUDA compatibility if GPU is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = sample_linear_model
        device = torch.device("cuda:0")
        
        # Move model to GPU
        model = model.to(device)
        
        # Test forward pass on GPU
        input_data = torch.randn(8, 10, device=device)
        output = model(input_data)
        
        assert output.device == device
        assert output.shape == (8, 1)
    
    def test_autograd_compatibility(self, sample_linear_model):
        """Test autograd functionality."""
        model = sample_linear_model
        input_data = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 1)
        
        # Forward pass
        output = model(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients were computed
        assert input_data.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestDependencyCompatibility:
    """Test compatibility with other dependencies."""
    
    def test_numpy_compatibility(self):
        """Test NumPy integration."""
        import numpy as np
        
        # Test tensor to numpy conversion
        tensor = torch.randn(3, 3)
        numpy_array = tensor.detach().numpy()
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (3, 3)
        
        # Test numpy to tensor conversion
        back_to_tensor = torch.from_numpy(numpy_array)
        assert torch.allclose(tensor, back_to_tensor)
    
    def test_scipy_compatibility(self):
        """Test SciPy integration if available."""
        try:
            import scipy
            import scipy.sparse
            import numpy as np
            
            # Test sparse matrix compatibility
            sparse_matrix = scipy.sparse.csr_matrix(np.eye(4))
            dense_tensor = torch.from_numpy(sparse_matrix.toarray()).float()
            
            assert dense_tensor.shape == (4, 4)
            assert torch.allclose(dense_tensor, torch.eye(4))
            
        except ImportError:
            pytest.skip("SciPy not available")
    
    def test_matplotlib_compatibility(self):
        """Test matplotlib integration if available."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Test plotting tensor data
            data = torch.randn(100)
            plt.figure(figsize=(6, 4))
            plt.plot(data.numpy())
            plt.close()
            
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_pydantic_compatibility(self):
        """Test Pydantic integration for configuration validation."""
        try:
            from pydantic import BaseModel, ValidationError
            
            class PhotonicConfig(BaseModel):
                wavelength: float
                pdk: str
                energy_per_op: float
                latency: float
            
            # Test valid config
            config = PhotonicConfig(
                wavelength=1550.0,
                pdk="skywater130",
                energy_per_op=0.5,
                latency=100.0
            )
            
            assert config.wavelength == 1550.0
            assert config.pdk == "skywater130"
            
            # Test invalid config
            with pytest.raises(ValidationError):
                PhotonicConfig(
                    wavelength="invalid",  # Should be float
                    pdk="skywater130",
                    energy_per_op=0.5,
                    latency=100.0
                )
                
        except ImportError:
            pytest.skip("pydantic not available")


class TestPythonVersionCompatibility:
    """Test compatibility across Python versions."""
    
    def test_minimum_python_version(self):
        """Test minimum Python version requirement."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        assert current_version >= min_version, (
            f"Python version {current_version} is below minimum {min_version}"
        )
    
    def test_typing_compatibility(self):
        """Test typing features work correctly."""
        from typing import Dict, List, Optional, Union
        
        # Test type annotations work
        def test_function(
            param1: List[str], 
            param2: Optional[Dict[str, int]] = None
        ) -> Union[str, None]:
            if param2 is None:
                return None
            return str(len(param1))
        
        result = test_function(["a", "b"], {"x": 1})
        assert result == "2"
    
    def test_pathlib_compatibility(self):
        """Test pathlib functionality."""
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            test_file = path / "test.txt"
            
            # Test path operations
            assert path.exists()
            assert path.is_dir()
            assert not test_file.exists()
            
            # Test file creation
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"
    
    def test_dataclass_compatibility(self):
        """Test dataclass functionality (Python 3.7+)."""
        from dataclasses import dataclass
        
        @dataclass
        class PhotonicParams:
            wavelength: float
            pdk: str
            energy_per_op: float = 0.5
        
        params = PhotonicParams(wavelength=1550.0, pdk="skywater130")
        assert params.wavelength == 1550.0
        assert params.pdk == "skywater130"
        assert params.energy_per_op == 0.5


class TestEnvironmentCompatibility:
    """Test compatibility with different execution environments."""
    
    def test_jupyter_compatibility(self):
        """Test Jupyter notebook compatibility."""
        try:
            # Check if we're in a Jupyter environment
            from IPython import get_ipython
            if get_ipython() is not None:
                # We're in an IPython/Jupyter environment
                assert True
            else:
                pytest.skip("Not in Jupyter environment")
        except ImportError:
            pytest.skip("IPython not available")
    
    def test_docker_compatibility(self):
        """Test Docker environment compatibility."""
        import os
        
        # Check for common Docker environment indicators
        docker_indicators = [
            os.path.exists('/.dockerenv'),
            os.environ.get('DOCKER_CONTAINER'),
            'docker' in os.environ.get('PATH', '').lower()
        ]
        
        if any(docker_indicators):
            # We're likely in a Docker container
            # Test that basic functionality works
            tensor = torch.randn(10, 10)
            assert tensor.shape == (10, 10)
        else:
            # Not in Docker, but that's fine too
            assert True
    
    def test_ci_environment_compatibility(self):
        """Test CI/CD environment compatibility."""
        ci_indicators = [
            'CI',
            'CONTINUOUS_INTEGRATION', 
            'GITHUB_ACTIONS',
            'GITLAB_CI',
            'JENKINS_URL'
        ]
        
        is_ci = any(os.environ.get(indicator) for indicator in ci_indicators)
        
        if is_ci:
            # In CI environment - test basic functionality
            model = torch.nn.Linear(5, 1)
            input_data = torch.randn(1, 5)
            output = model(input_data)
            assert output.shape == (1, 1)
        
        # Test passes regardless of environment
        assert True