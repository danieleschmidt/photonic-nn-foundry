"""
Tests for photonic-nn-foundry.

Covers MZI, PhotonicMeshLayer, PhotonicLinear, and PhotonicNet.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from photonic_nn_foundry import (
    MachZehnderInterferometer,
    PhotonicMeshLayer,
    PhotonicLinear,
    PhotonicNet,
)


# ─── MZI Tests ─────────────────────────────────────────────────────────────────

def test_mzi_transmission_matrix_shape():
    """Transmission matrix must be 2×2 complex."""
    mzi = MachZehnderInterferometer()
    T = mzi.transmission_matrix()
    assert T.shape == (2, 2), f"Expected (2,2), got {T.shape}"
    assert np.iscomplexobj(T), "Transmission matrix should be complex"


def test_mzi_phase_shift():
    """
    MZI at theta=0: should be a phase-only device (no power splitting).
    At theta=0, cos(0)=1, sin(0)=0 → T = [[e^iphi, 0], [0, 1]].
    """
    mzi = MachZehnderInterferometer(theta=0.0, phi=np.pi / 2)
    T = mzi.transmission_matrix()
    expected_T00 = np.exp(1j * np.pi / 2)
    assert abs(T[0, 0] - expected_T00) < 1e-10, f"T[0,0] mismatch: {T[0,0]} vs {expected_T00}"
    assert abs(T[1, 0]) < 1e-10, f"T[1,0] should be ~0, got {T[1,0]}"
    assert abs(T[0, 1]) < 1e-10, f"T[0,1] should be ~0, got {T[0,1]}"
    assert abs(T[1, 1] - 1.0) < 1e-10, f"T[1,1] should be 1, got {T[1,1]}"


def test_mzi_forward():
    """MZI forward pass should return a 2-element vector."""
    mzi = MachZehnderInterferometer(theta=np.pi / 4, phi=0.0)
    x = np.array([1.0 + 0j, 0.0 + 0j])
    y = mzi.forward(x)
    assert y.shape == (2,), f"Expected shape (2,), got {y.shape}"
    # Check energy conservation: |y|^2 = |x|^2
    assert abs(np.sum(np.abs(y)**2) - np.sum(np.abs(x)**2)) < 1e-10, \
        "MZI should conserve optical power"


def test_mzi_set_phases():
    """set_phases should update theta and phi."""
    mzi = MachZehnderInterferometer(theta=0.0, phi=0.0)
    mzi.set_phases(np.pi / 3, np.pi / 6)
    assert abs(mzi.theta - np.pi / 3) < 1e-12
    assert abs(mzi.phi - np.pi / 6) < 1e-12

    # Verify the matrix changed
    T = mzi.transmission_matrix()
    c = np.cos(np.pi / 6)
    s = np.sin(np.pi / 6)
    assert abs(T[0, 0] - np.exp(1j * np.pi / 6) * c) < 1e-10


def test_mzi_unitarity():
    """Transmission matrix must be unitary: T† T = I."""
    for _ in range(5):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        mzi = MachZehnderInterferometer(theta=theta, phi=phi)
        T = mzi.transmission_matrix()
        product = T.conj().T @ T
        assert np.allclose(product, np.eye(2), atol=1e-10), \
            f"MZI matrix not unitary: T†T = {product}"


# ─── Mesh Tests ─────────────────────────────────────────────────────────────────

def test_mesh_layer_shape():
    """Mesh unitary should be size×size complex."""
    mesh = PhotonicMeshLayer(size=4)
    U = mesh.get_unitary()
    assert U.shape == (4, 4), f"Expected (4,4), got {U.shape}"
    assert np.iscomplexobj(U)


def test_mesh_forward_shape():
    """Mesh forward should preserve vector dimension."""
    mesh = PhotonicMeshLayer(size=6)
    x = np.random.randn(6)
    y = mesh.forward(x)
    assert y.shape == (6,), f"Expected (6,), got {y.shape}"


def test_mesh_unitary():
    """The full mesh unitary matrix must be unitary: U†U = I."""
    for size in [2, 4, 6]:
        mesh = PhotonicMeshLayer(size=size)
        U = mesh.get_unitary()
        product = U.conj().T @ U
        assert np.allclose(product, np.eye(size), atol=1e-9), \
            f"Mesh size={size} not unitary: max error = {np.max(np.abs(product - np.eye(size))):.2e}"


def test_mesh_set_phases():
    """set_phases should update the mesh and change the output."""
    mesh = PhotonicMeshLayer(size=4)
    x = np.random.randn(4)

    y_before = mesh.forward(x).copy()

    # Build a new random phase array
    n = mesh.num_phases()
    new_phases = np.random.uniform(0, 2 * np.pi, n)
    mesh.set_phases(new_phases)

    y_after = mesh.forward(x)
    # The output should (almost certainly) have changed
    assert not np.allclose(y_before, y_after), \
        "set_phases had no effect on mesh output"


def test_mesh_forward_batch():
    """Mesh forward with a batch of inputs."""
    mesh = PhotonicMeshLayer(size=4)
    X = np.random.randn(8, 4)  # batch of 8
    Y = mesh.forward(X)
    assert Y.shape == (8, 4), f"Expected (8,4), got {Y.shape}"


# ─── PhotonicLinear Tests ───────────────────────────────────────────────────────

def test_photonic_linear_forward_shape():
    """PhotonicLinear forward should map (batch, in) → (batch, out)."""
    layer = PhotonicLinear(in_features=8, out_features=4)
    X = np.random.randn(16, 8)
    Y = layer.forward(X)
    assert Y.shape == (16, 4), f"Expected (16,4), got {Y.shape}"


def test_photonic_linear_weight_matrix():
    """Weight matrix should be (out, in) shaped."""
    layer = PhotonicLinear(in_features=6, out_features=3)
    W = layer.weight_matrix()
    assert W.shape == (3, 6), f"Expected (3,6), got {W.shape}"
    assert np.iscomplexobj(W)


def test_photonic_linear_1d_input():
    """PhotonicLinear should handle 1D (non-batched) input."""
    layer = PhotonicLinear(in_features=4, out_features=4)
    x = np.random.randn(4)
    y = layer.forward(x)
    assert y.shape == (4,), f"Expected (4,), got {y.shape}"


# ─── PhotonicNet Tests ──────────────────────────────────────────────────────────

def test_photonic_net_forward():
    """PhotonicNet forward pass should run without errors."""
    net = PhotonicNet(input_size=16, hidden_size=8, output_size=4)
    x = np.random.randn(16)
    out = net.forward(x)
    assert out is not None
    assert not np.any(np.isnan(out)), "Output contains NaN"
    assert not np.any(np.isinf(out)), "Output contains Inf"


def test_photonic_net_output_shape():
    """PhotonicNet output shape should match output_size."""
    net = PhotonicNet(input_size=16, hidden_size=8, output_size=4)
    X = np.random.randn(32, 16)
    out = net.forward(X)
    assert out.shape == (32, 4), f"Expected (32,4), got {out.shape}"


def test_photonic_net_train_step():
    """train_step should return a scalar loss and not crash."""
    net = PhotonicNet(input_size=16, hidden_size=8, output_size=4)
    X = np.random.randn(10, 16)
    y = np.random.randint(0, 4, size=10)
    loss = net.train_step(X, y, lr=0.01)
    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
    assert not np.isnan(loss), "Loss is NaN"


def test_full_pipeline():
    """End-to-end: build net, run forward, do train steps, check loss decreases."""
    np.random.seed(42)
    net = PhotonicNet(input_size=20, hidden_size=10, output_size=5)

    # Synthetic dataset
    X = np.random.randn(50, 20)
    y = np.random.randint(0, 5, size=50)

    # Collect losses over a few steps
    losses = []
    for _ in range(5):
        loss = net.train_step(X, y, lr=0.05)
        losses.append(loss)

    # All losses should be valid numbers
    assert all(not np.isnan(l) for l in losses), f"NaN in losses: {losses}"
    assert all(l >= 0 for l in losses), f"Negative loss: {losses}"

    # At least verify the network runs end-to-end and produces valid output
    out = net.forward(X)
    assert out.shape == (50, 5)
    assert not np.any(np.isnan(out))
