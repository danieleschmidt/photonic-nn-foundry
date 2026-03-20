"""
photonic-nn-foundry demo: MNIST-style classification with an optical neural network.

Creates synthetic 784-dimensional data (like flattened 28×28 images),
builds a 2-layer photonic network, and runs a short training loop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from photonic_nn_foundry import PhotonicNet, cross_entropy_loss, softmax

# ─── Synthetic MNIST-style dataset ─────────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 200
N_CLASSES = 10
INPUT_DIM = 784  # 28×28

print("=" * 60)
print("  photonic-nn-foundry: Silicon-Photonic Neural Network Demo")
print("=" * 60)
print()
print(f"Dataset: {N_SAMPLES} synthetic MNIST-style samples")
print(f"Input:   {INPUT_DIM} dimensions (28×28 pixels)")
print(f"Classes: {N_CLASSES}")
print()

# Random Gaussian data normalized to [0,1] — like pixel intensities
X = np.random.randn(N_SAMPLES, INPUT_DIM) * 0.1 + 0.5
X = np.clip(X, 0, 1)
y = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

# ─── Build photonic network ─────────────────────────────────────────────────────
print("Building PhotonicNet(784, 128, 10)...")
net = PhotonicNet(input_size=INPUT_DIM, hidden_size=128, output_size=N_CLASSES)
print(f"  {net}")
print()

# ─── Forward pass ───────────────────────────────────────────────────────────────
print("Running forward pass...")
logits = net.forward(X[:4])
print(f"  Input shape:  {X[:4].shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Sample logits (first example): {logits[0].round(4)}")
probs = softmax(logits)
print(f"  Sample probs  (first example): {probs[0].round(4)}")
print(f"  Predicted class: {np.argmax(probs[0])}  |  True class: {y[0]}")
print()

# ─── Training loop ──────────────────────────────────────────────────────────────
print("Training (5 steps, SPSA gradient estimation)...")
print("-" * 40)
batch_size = 32

for step in range(5):
    idx = np.random.choice(N_SAMPLES, batch_size, replace=False)
    X_batch = X[idx]
    y_batch = y[idx]

    loss = net.train_step(X_batch, y_batch, lr=0.02)

    # Compute accuracy on the batch
    logits_batch = net.forward(X_batch)
    preds = np.argmax(logits_batch, axis=1)
    acc = np.mean(preds == y_batch)

    print(f"  Step {step+1}/5  |  loss={loss:.4f}  |  batch_acc={acc:.2%}")

print()
print("Done! The photonic network ran entirely on NumPy — no GPUs, no PyTorch.")
print("In real silicon-photonic hardware, the matrix multiplies happen at the speed of light. ⚡")
