"""
PhotonicMeshLayer - an N×N unitary optical mesh built from MZI units.

Uses the Clements (2016) butterfly decomposition: a rectangular mesh that can
implement any N×N unitary matrix with N(N-1)/2 MZIs and N phase shifters.

Reference: Clements et al., "Optimal design for universal multiport interferometers",
           Optica 3, 1460 (2016).
"""

import numpy as np
from .mzi import MachZehnderInterferometer


class PhotonicMeshLayer:
    """
    An N×N unitary photonic mesh layer.

    Implements an arbitrary N×N unitary transformation using a rectangular
    array of MZIs in the Clements topology. This is the core building block
    of photonic neural networks — equivalent to a complex-valued weight matrix
    that is constrained to be unitary.

    The mesh is organized as 'depth' columns of MZIs. Each column covers
    alternating pairs of modes. The number of MZI columns = N.
    """

    def __init__(self, size: int, topology: str = 'clements'):
        """
        Args:
            size:     Number of optical modes (mesh is size × size).
            topology: Mesh topology. Currently supports 'clements'.
        """
        if topology != 'clements':
            raise ValueError(f"Unsupported topology '{topology}'. Use 'clements'.")
        self.size = size
        self.topology = topology

        # Build the MZI grid: N columns of floor(N/2) MZIs each
        # Column c uses even offsets if c is even, odd offsets if c is odd
        self.columns: list[list[MachZehnderInterferometer]] = []
        for col in range(size):
            offset = col % 2  # alternating even/odd
            mzis = []
            for row in range(offset, size - 1, 2):
                mzis.append(MachZehnderInterferometer(
                    theta=np.random.uniform(0, np.pi),
                    phi=np.random.uniform(0, 2 * np.pi),
                ))
            self.columns.append(mzis)

        # Output phase shifters (diagonal phases)
        self.output_phases = np.random.uniform(0, 2 * np.pi, size)

    def _column_unitary(self, col: int) -> np.ndarray:
        """Build the full N×N unitary for a single column of MZIs."""
        U = np.eye(self.size, dtype=np.complex128)
        offset = col % 2
        mzis = self.columns[col]
        mzi_idx = 0
        for row in range(offset, self.size - 1, 2):
            T = mzis[mzi_idx].transmission_matrix()
            # Embed 2x2 T into the N×N identity at position (row, row)
            U[row:row+2, row:row+2] = T
            mzi_idx += 1
        return U

    def get_unitary(self) -> np.ndarray:
        """
        Compute and return the full N×N unitary matrix of the mesh.

        Returns:
            np.ndarray of shape (size, size), complex128.
        """
        U = np.eye(self.size, dtype=np.complex128)
        for col in range(self.size):
            U = self._column_unitary(col) @ U
        # Apply output diagonal phase shifters
        D = np.diag(np.exp(1j * self.output_phases))
        return D @ U

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the photonic mesh transform to an input vector or batch.

        Args:
            x: Input array of shape (size,) or (batch, size).

        Returns:
            Transformed array, same shape as x.
        """
        U = self.get_unitary()
        if x.ndim == 1:
            return U @ x
        else:
            # batch: x is (batch, size), return (batch, size)
            return x @ U.T

    def set_phases(self, phases: np.ndarray):
        """
        Set all MZI phases from a flat array.

        The array is ordered as: [theta0, phi0, theta1, phi1, ...] for each MZI
        across all columns (left to right), followed by N output phases.

        Args:
            phases: 1D array of length 2 * total_mzis + size.
        """
        idx = 0
        for col in self.columns:
            for mzi in col:
                mzi.set_phases(phases[idx], phases[idx + 1])
                idx += 2
        # Remaining values are output phases
        if idx < len(phases):
            self.output_phases = np.array(phases[idx:idx + self.size], dtype=float)

    def num_phases(self) -> int:
        """Total number of phase parameters in this mesh."""
        total_mzis = sum(len(col) for col in self.columns)
        return 2 * total_mzis + self.size

    def get_phases(self) -> np.ndarray:
        """Return all phase parameters as a flat array."""
        phases = []
        for col in self.columns:
            for mzi in col:
                phases.extend([mzi.theta, mzi.phi])
        phases.extend(self.output_phases.tolist())
        return np.array(phases)

    def __repr__(self) -> str:
        n_mzis = sum(len(col) for col in self.columns)
        return f"PhotonicMeshLayer(size={self.size}, topology='{self.topology}', n_mzis={n_mzis})"
