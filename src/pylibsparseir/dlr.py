"""
Discrete Lehmann Representation (DLR) functionality for SparseIR.

This module implements DLR basis with poles at IR extrema, providing
an alternative representation that can be more efficient for certain calculations.
"""

import numpy as np
from .abstract import AbstractBasis
from .basis import FiniteTempBasis


class DiscreteLehmannRepresentation(AbstractBasis):
    """
    Discrete Lehmann Representation basis.

    The DLR provides an alternative representation of Green's functions
    using poles at the IR sampling points. This can be more efficient
    than the standard IR basis for certain applications.
    """

    def __init__(self, basis_or_kernel, eps=None):
        """
        Initialize DLR from IR basis or kernel.

        Parameters
        ----------
        basis_or_kernel : FiniteTempBasis or Kernel
            Either a FiniteTempBasis or a kernel object
        eps : float, optional
            Accuracy threshold for the DLR construction
        """
        if isinstance(basis_or_kernel, FiniteTempBasis):
            self._ir_basis = basis_or_kernel
        else:
            # Assume it's a kernel - would need to construct IR basis
            raise NotImplementedError("DLR construction from kernel not yet implemented")

        self._eps = eps or 1e-12

        # Get the poles from the IR basis sampling points
        self._tau_poles = self._ir_basis.default_tau_sampling_points()
        self._matsubara_poles = self._ir_basis.default_matsubara_sampling_points()

        # Store properties
        self._size = len(self._tau_poles)

        # Precompute transformation matrices
        self._compute_transformation_matrices()

    def _compute_transformation_matrices(self):
        """Compute transformation matrices between IR and DLR."""
        # Transformation matrix from IR to DLR in tau
        self._tau_ir_to_dlr = self._ir_basis.u(self._tau_poles)  # Shape: (n_poles, ir_size)

        # Transformation matrix from IR to DLR in Matsubara
        self._matsu_ir_to_dlr = self._ir_basis.uhat(self._matsubara_poles)  # Shape: (n_poles, ir_size)

        # Store inverses for DLR to IR transformation
        try:
            self._tau_dlr_to_ir = np.linalg.pinv(self._tau_ir_to_dlr)
            self._matsu_dlr_to_ir = np.linalg.pinv(self._matsu_ir_to_dlr)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            self._tau_dlr_to_ir = np.linalg.lstsq(self._tau_ir_to_dlr, np.eye(self._size), rcond=None)[0]
            self._matsu_dlr_to_ir = np.linalg.lstsq(self._matsu_ir_to_dlr, np.eye(self._size), rcond=None)[0]

    @property
    def statistics(self):
        """Quantum statistic."""
        return self._ir_basis.statistics

    @property
    def size(self):
        """Number of DLR poles."""
        return self._size

    @property
    def significance(self):
        """Significances of DLR poles."""
        # Use IR basis significances (reordered according to poles)
        return self._ir_basis.significance[:self._size]

    @property
    def lambda_(self):
        """Basis cutoff parameter."""
        return self._ir_basis.lambda_

    @property
    def beta(self):
        """Inverse temperature."""
        return self._ir_basis.beta

    @property
    def tau_poles(self):
        """DLR poles in imaginary time."""
        return self._tau_poles

    @property
    def matsubara_poles(self):
        """DLR poles in Matsubara frequency."""
        return self._matsubara_poles

    @property
    def u(self):
        """Imaginary-time basis functions."""
        return DLRTauBasisFunctions(self)

    @property
    def uhat(self):
        """Matsubara frequency basis functions."""
        return DLRMatsubaraBasisFunctions(self)

    def ir_to_dlr(self, ir_coeffs, domain='tau'):
        """
        Transform IR coefficients to DLR coefficients.

        Parameters
        ----------
        ir_coeffs : array_like
            IR expansion coefficients
        domain : str
            Transformation domain ('tau' or 'matsubara')

        Returns
        -------
        ndarray
            DLR coefficients
        """
        ir_coeffs = np.asarray(ir_coeffs)

        if domain == 'tau':
            return self._tau_ir_to_dlr @ ir_coeffs
        elif domain == 'matsubara':
            return self._matsu_ir_to_dlr @ ir_coeffs
        else:
            raise ValueError("domain must be 'tau' or 'matsubara'")

    def dlr_to_ir(self, dlr_coeffs, domain='tau'):
        """
        Transform DLR coefficients to IR coefficients.

        Parameters
        ----------
        dlr_coeffs : array_like
            DLR coefficients
        domain : str
            Transformation domain ('tau' or 'matsubara')

        Returns
        -------
        ndarray
            IR expansion coefficients
        """
        dlr_coeffs = np.asarray(dlr_coeffs)

        if domain == 'tau':
            return self._tau_dlr_to_ir @ dlr_coeffs
        elif domain == 'matsubara':
            return self._matsu_dlr_to_ir @ dlr_coeffs
        else:
            raise ValueError("domain must be 'tau' or 'matsubara'")

    def default_tau_sampling_points(self, *, npoints=None):
        """Default sampling points on imaginary time axis."""
        return self._tau_poles

    def default_matsubara_sampling_points(self, *, npoints=None, positive_only=False):
        """Default sampling points on Matsubara frequency axis."""
        if positive_only:
            return self._matsubara_poles[self._matsubara_poles >= 0]
        return self._matsubara_poles


class DLRTauBasisFunctions:
    """DLR basis functions for imaginary time evaluation."""

    def __init__(self, dlr):
        self.dlr = dlr

    def __call__(self, tau):
        """
        Evaluate DLR basis functions at tau points.

        The DLR basis functions are constructed using Lagrange interpolation
        at the DLR poles.
        """
        tau = np.asarray(tau)
        tau_flat = tau.flatten()

        # Lagrange interpolation basis functions
        result = np.zeros((len(tau_flat), self.dlr.size))

        for i, tau_val in enumerate(tau_flat):
            for j, pole in enumerate(self.dlr.tau_poles):
                # Lagrange basis function j at tau_val
                lagrange_val = 1.0
                for k, other_pole in enumerate(self.dlr.tau_poles):
                    if k != j:
                        lagrange_val *= (tau_val - other_pole) / (pole - other_pole)
                result[i, j] = lagrange_val

        return result.reshape(tau.shape + (self.dlr.size,))


class DLRMatsubaraBasisFunctions:
    """DLR basis functions for Matsubara frequency evaluation."""

    def __init__(self, dlr):
        self.dlr = dlr

    def __call__(self, n):
        """
        Evaluate DLR basis functions at Matsubara frequencies.

        The DLR basis functions are constructed using rational interpolation
        at the DLR poles.
        """
        n = np.asarray(n)
        n_flat = n.flatten()

        # Convert Matsubara indices to frequencies
        if self.dlr.statistics == 'F':
            # Fermionic: ω_n = (2n+1)πi/β
            omega = (2 * n_flat + 1) * np.pi * 1j / self.dlr.beta
        else:
            # Bosonic: ω_n = 2nπi/β
            omega = 2 * n_flat * np.pi * 1j / self.dlr.beta

        # Convert pole indices to frequencies
        if self.dlr.statistics == 'F':
            pole_omega = (2 * self.dlr.matsubara_poles + 1) * np.pi * 1j / self.dlr.beta
        else:
            pole_omega = 2 * self.dlr.matsubara_poles * np.pi * 1j / self.dlr.beta

        # Lagrange interpolation in frequency domain
        result = np.zeros((len(n_flat), self.dlr.size), dtype=complex)

        for i, omega_val in enumerate(omega):
            for j, pole_omega_val in enumerate(pole_omega):
                # Lagrange basis function j at omega_val
                lagrange_val = 1.0 + 0j
                for k, other_pole_omega in enumerate(pole_omega):
                    if k != j:
                        lagrange_val *= (omega_val - other_pole_omega) / (pole_omega_val - other_pole_omega)
                result[i, j] = lagrange_val

        return result.reshape(n.shape + (self.dlr.size,))


class TauPoles:
    """
    Imaginary time representation using DLR poles.

    This class represents a function in imaginary time using DLR coefficients
    at the pole locations.
    """

    def __init__(self, dlr, coeffs):
        """
        Initialize tau poles representation.

        Parameters
        ----------
        dlr : DiscreteLehmannRepresentation
            DLR basis
        coeffs : array_like
            DLR coefficients
        """
        self.dlr = dlr
        self.coeffs = np.asarray(coeffs)

        if len(self.coeffs) != self.dlr.size:
            raise ValueError(f"Number of coefficients ({len(self.coeffs)}) must match DLR size ({self.dlr.size})")

    def __call__(self, tau):
        """Evaluate function at tau points."""
        # Use DLR basis functions for evaluation
        basis_vals = self.dlr.u(tau)  # Shape: (..., dlr_size)
        return np.sum(basis_vals * self.coeffs, axis=-1)

    def to_ir(self):
        """Convert to IR representation."""
        ir_coeffs = self.dlr.dlr_to_ir(self.coeffs, domain='tau')
        from .basis import FiniteTempBasis  # Avoid circular import
        # Would need to create a proper IR function representation
        return ir_coeffs


class MatsubaraPoles:
    """
    Matsubara frequency representation using DLR poles.

    This class represents a function in Matsubara frequency using DLR coefficients
    at the pole locations.
    """

    def __init__(self, dlr, coeffs):
        """
        Initialize Matsubara poles representation.

        Parameters
        ----------
        dlr : DiscreteLehmannRepresentation
            DLR basis
        coeffs : array_like
            DLR coefficients
        """
        self.dlr = dlr
        self.coeffs = np.asarray(coeffs, dtype=complex)

        if len(self.coeffs) != self.dlr.size:
            raise ValueError(f"Number of coefficients ({len(self.coeffs)}) must match DLR size ({self.dlr.size})")

    def __call__(self, n):
        """Evaluate function at Matsubara frequencies."""
        # Use DLR basis functions for evaluation
        basis_vals = self.dlr.uhat(n)  # Shape: (..., dlr_size)
        return np.sum(basis_vals * self.coeffs, axis=-1)

    def to_ir(self):
        """Convert to IR representation."""
        ir_coeffs = self.dlr.dlr_to_ir(self.coeffs, domain='matsubara')
        # Would need to create a proper IR function representation
        return ir_coeffs