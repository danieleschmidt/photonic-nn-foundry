"""Shared test configuration and fixtures."""

import pytest
import torch
import numpy as np


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