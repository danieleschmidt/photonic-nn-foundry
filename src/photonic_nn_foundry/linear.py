"""
PhotonicLinear - optical matrix multiplication layer.

Implements a rectangular (in_features → out_features) linear transformation
using photonic hardware. The weight matrix is implemented via two unitary meshes
and a diagonal singular value layer (SVD decomposition: W = U @ Sigma @ V†).

This is the photonic equivalent of torch.nn.Linear (without bias).
"""

import numpy as np
from .mesh import PhotonicMeshLayer


class PhotonicLinear:
    """
    Photonic linear layer: y = x @ W.T

    Implements W = U @ Sigma @ V† where:
      - U: out_features × out_features unitary (output mesh)
      - Sigma: min(in, out) × min(in, out) diagonal (amplitude weights)
      - V†: in_features × in_features unitary (input mesh, conjugate transpose)

    This corresponds to the physical SVD decomposition used in optical matrix
    multiply units (MZI mesh + attenuators + MZI mesh).
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features:  Number of input optical modes.
            out_features: Number of output optical modes.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)

        # Two unitary meshes (V and U in the SVD W = U Σ V†)
        self.mesh_v = PhotonicMeshLayer(in_features)   # input transform
        self.mesh_u = PhotonicMeshLayer(out_features)  # output transform

        # Singular values (non-negative amplitudes), initialized randomly
        self.singular_values = np.abs(np.random.randn(self.rank)) * 0.1 + 0.5
        self.singular_values = np.clip(self.singular_values, 0, 1)

    def weight_matrix(self) -> np.ndarray:
        """
        Compute the effective weight matrix W = U @ Sigma_rect @ V†.

        Returns:
            np.ndarray of shape (out_features, in_features), complex128.
        """
        V = self.mesh_v.get_unitary()   # (in, in)
        U = self.mesh_u.get_unitary()   # (out, out)

        # Build rectangular diagonal singular value matrix
        Sigma = np.zeros((self.out_features, self.in_features), dtype=np.complex128)
        for i in range(self.rank):
            Sigma[i, i] = self.singular_values[i]

        return U @ Sigma @ V.conj().T

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the photonic linear transform.

        Performs: y = x @ W.T  (i.e., W @ x for column vectors)

        Args:
            x: Input of shape (in_features,) or (batch, in_features).
               If complex, operates on the complex amplitudes directly.
               If real, treated as real-valued optical intensities.

        Returns:
            Output of shape (out_features,) or (batch, out_features).
        """
        W = self.weight_matrix()
        if x.ndim == 1:
            return W @ x
        else:
            return x @ W.conj().T

    def __repr__(self) -> str:
        return f"PhotonicLinear(in_features={self.in_features}, out_features={self.out_features})"
