"""
Mach-Zehnder Interferometer (MZI) - fundamental building block of photonic neural networks.

An MZI splits light, applies phase shifts, then recombines — implementing a 2x2 unitary transform.
The two phase parameters theta (internal) and phi (external) define the transmission matrix.
"""

import numpy as np


class MachZehnderInterferometer:
    """
    A single Mach-Zehnder Interferometer unit.

    Implements a 2x2 unitary transformation parameterized by two phase angles:
      - theta: internal phase shift (controls splitting ratio)
      - phi: external phase shift (controls output phase)

    The transmission matrix follows the Clements convention:
        T = [[e^(i*phi) * cos(theta/2),  -sin(theta/2)],
             [e^(i*phi) * sin(theta/2),   cos(theta/2)]]
    """

    def __init__(self, theta: float = 0.0, phi: float = 0.0):
        """
        Args:
            theta: Internal phase shift (beamsplitter angle), in radians.
            phi:   External phase shift (output phase), in radians.
        """
        self.theta = float(theta)
        self.phi = float(phi)

    def set_phases(self, theta: float, phi: float):
        """Update the MZI phase angles."""
        self.theta = float(theta)
        self.phi = float(phi)

    def transmission_matrix(self) -> np.ndarray:
        """
        Compute the 2x2 complex unitary transmission matrix.

        Returns:
            np.ndarray of shape (2, 2), complex128.
        """
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        phase = np.exp(1j * self.phi)
        return np.array([
            [phase * c, -s],
            [phase * s,  c],
        ], dtype=np.complex128)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the MZI transformation to a 2-element input vector.

        Args:
            x: Input array of shape (2,) or (2, batch).

        Returns:
            Transformed array, same shape as x.
        """
        T = self.transmission_matrix()
        return T @ x

    def __repr__(self) -> str:
        return f"MachZehnderInterferometer(theta={self.theta:.4f}, phi={self.phi:.4f})"
