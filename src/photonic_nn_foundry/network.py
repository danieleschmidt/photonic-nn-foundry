"""
PhotonicNet - a 2-layer optical neural network.

Stacks two PhotonicLinear layers with a nonlinearity between them.
Since photonic hardware is linear by nature, nonlinearities are applied
electronically between optical layers (optoelectronic nonlinearity).

Training uses finite-difference gradient estimation (SPSA or perturbation)
since backprop through optical hardware is not directly available.
"""

import numpy as np
from .linear import PhotonicLinear


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU nonlinearity (applied electronically between photonic layers)."""
    return np.maximum(0, x.real).astype(np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax for classification output."""
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
        return e / e.sum()
    else:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss.

    Args:
        logits: (batch, num_classes) — raw output scores.
        labels: (batch,) — integer class indices.

    Returns:
        Scalar loss value.
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    eps = 1e-9
    correct_log_probs = -np.log(probs[np.arange(batch_size), labels] + eps)
    return float(np.mean(correct_log_probs))


class PhotonicNet:
    """
    2-layer photonic neural network.

    Architecture:
        input → PhotonicLinear(input_size, hidden_size) → ReLU
              → PhotonicLinear(hidden_size, output_size) → Softmax (at eval)

    The network operates on real-valued inputs (e.g., pixel intensities),
    and produces real-valued class scores as output.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Args:
            input_size:  Dimensionality of input (e.g., 784 for 28×28 images).
            hidden_size: Number of hidden optical modes.
            output_size: Number of output classes.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = PhotonicLinear(input_size, hidden_size)
        self.layer2 = PhotonicLinear(hidden_size, output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Full forward pass through the photonic network.

        Args:
            x: Input of shape (input_size,) or (batch, input_size).
               Real-valued (e.g., normalized pixel intensities in [0, 1]).

        Returns:
            Raw logits of shape (output_size,) or (batch, output_size).
        """
        x = np.asarray(x, dtype=np.float64)
        # Layer 1: optical matrix multiply → electronic ReLU
        h = self.layer1.forward(x)
        h = relu(h)
        # Layer 2: optical matrix multiply → logits
        out = self.layer2.forward(h)
        return out.real

    def _get_all_params(self) -> list[np.ndarray]:
        """Collect all trainable parameter arrays."""
        return [
            self.layer1.singular_values,
            self.layer2.singular_values,
        ]

    def _set_all_params(self, params: list[np.ndarray]):
        """Set trainable parameter arrays."""
        self.layer1.singular_values = np.clip(params[0], 0, 1)
        self.layer2.singular_values = np.clip(params[1], 0, 1)

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        One training step using simultaneous perturbation stochastic approximation (SPSA).

        SPSA estimates the gradient using two forward passes with random ±delta
        perturbations, making it hardware-friendly for optical systems.

        Args:
            X:  Input batch, shape (batch, input_size).
            y:  Integer class labels, shape (batch,).
            lr: Learning rate.

        Returns:
            Loss value before the update.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        # Current loss
        logits = self.forward(X)
        loss = cross_entropy_loss(logits, y)

        # SPSA: perturb singular values only (most impactful parameters)
        delta = 0.01
        params = self._get_all_params()
        new_params = []

        for idx, p in enumerate(params):
            # Random ±1 perturbation vector
            perturbation = np.random.choice([-1.0, 1.0], size=p.shape)

            # Forward pass with +delta
            p_plus = p + delta * perturbation
            self._set_all_params([p_plus if i == idx else params[i]
                                   for i in range(len(params))])
            loss_plus = cross_entropy_loss(self.forward(X), y)

            # Forward pass with -delta
            p_minus = p - delta * perturbation
            self._set_all_params([p_minus if i == idx else params[i]
                                   for i in range(len(params))])
            loss_minus = cross_entropy_loss(self.forward(X), y)

            # SPSA gradient estimate
            grad = (loss_plus - loss_minus) / (2 * delta) * perturbation
            new_params.append(p - lr * grad)

            # Restore original
            self._set_all_params(params)

        self._set_all_params(new_params)
        return loss

    def __repr__(self) -> str:
        return (f"PhotonicNet(input={self.input_size}, "
                f"hidden={self.hidden_size}, output={self.output_size})")
