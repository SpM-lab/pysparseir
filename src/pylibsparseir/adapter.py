"""
Adapter module for backward compatibility with irbasis.

This module provides drop-in replacement functionality for existing irbasis code,
allowing users to migrate from irbasis to pysparseir with minimal code changes.
"""

import numpy as np
from .basis import FiniteTempBasis


class Basis:
    """
    Compatibility wrapper for irbasis.

    This class provides the same interface as the original irbasis.Basis class,
    allowing existing code to work with minimal modifications.
    """

    def __init__(self, statistics, beta, wmax, eps=None):
        """
        Initialize basis with irbasis-compatible interface.

        Parameters
        ----------
        statistics : str
            'F' for fermions, 'B' for bosons
        beta : float
            Inverse temperature
        wmax : float
            Frequency cutoff
        eps : float, optional
            Accuracy threshold
        """
        self._basis = FiniteTempBasis(statistics, beta, wmax, eps)
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._eps = eps or 1e-12

    @property
    def statistics(self):
        """Quantum statistic."""
        return self._statistics

    @property
    def beta(self):
        """Inverse temperature."""
        return self._beta

    @property
    def wmax(self):
        """Frequency cutoff."""
        return self._wmax

    @property
    def Lambda(self):
        """Cutoff parameter (irbasis naming convention)."""
        return self._basis.lambda_

    @property
    def size(self):
        """Number of basis functions."""
        return self._basis.size

    @property
    def dim(self):
        """Dimension of basis (alias for size)."""
        return self.size

    def u(self, tau):
        """
        Imaginary-time basis functions.

        Parameters
        ----------
        tau : float or array_like
            Imaginary time points

        Returns
        -------
        ndarray
            Basis function values
        """
        return self._basis.u(tau).T  # Transpose for irbasis compatibility

    def uhat(self, n):
        """
        Matsubara frequency basis functions.

        Parameters
        ----------
        n : int or array_like
            Matsubara frequency indices

        Returns
        -------
        ndarray
            Basis function values
        """
        return self._basis.uhat(n).T  # Transpose for irbasis compatibility

    def v(self, omega):
        """
        Real frequency basis functions.

        Parameters
        ----------
        omega : float or array_like
            Real frequency points

        Returns
        -------
        ndarray
            Basis function values
        """
        return self._basis.v(omega).T  # Transpose for irbasis compatibility

    @property
    def s(self):
        """Singular values."""
        return self._basis.s

    def accuracy(self):
        """Overall accuracy of the basis."""
        return self._basis.accuracy


def load(statistics, Lambda, h5file=None):
    """
    Load basis with irbasis interface.

    This function provides compatibility with the irbasis.load() function.
    The h5file parameter is ignored as pysparseir computes bases on-the-fly.

    Parameters
    ----------
    statistics : str
        'F' for fermions, 'B' for bosons
    Lambda : float
        Cutoff parameter (β * ωmax)
    h5file : str, optional
        HDF5 file path (ignored in pysparseir)

    Returns
    -------
    Basis
        Basis object compatible with irbasis interface
    """
    # For compatibility, we need to determine beta and wmax from Lambda
    # Since we don't have both, we'll use a default beta = 1.0
    beta = 1.0
    wmax = Lambda / beta

    return Basis(statistics, beta, wmax)


def sampling_points_x(basis, whichl):
    """
    Optimal sampling points in x space.

    Parameters
    ----------
    basis : Basis
        Basis object
    whichl : int
        Which basis function to use for sampling points

    Returns
    -------
    ndarray
        Sampling points in x space (tau)
    """
    # Return default tau sampling points
    return basis._basis.default_tau_sampling_points()


def sampling_points_y(basis, whichl):
    """
    Optimal sampling points in y space.

    Parameters
    ----------
    basis : Basis
        Basis object
    whichl : int
        Which basis function to use for sampling points

    Returns
    -------
    ndarray
        Sampling points in y space (omega)
    """
    # Return default omega sampling points
    return basis._basis.default_omega_sampling_points()


def sampling_points_matsubara(basis, whichl):
    """
    Sampling points in Matsubara domain.

    Parameters
    ----------
    basis : Basis
        Basis object
    whichl : int
        Which basis function to use for sampling points

    Returns
    -------
    ndarray
        Sampling points in Matsubara frequency domain
    """
    # Return default Matsubara sampling points
    return basis._basis.default_matsubara_sampling_points()


# Additional helper functions for compatibility
def create_basis(statistics, beta, wmax, eps=None):
    """
    Create a basis object (alternative constructor).

    Parameters
    ----------
    statistics : str
        'F' for fermions, 'B' for bosons
    beta : float
        Inverse temperature
    wmax : float
        Frequency cutoff
    eps : float, optional
        Accuracy threshold

    Returns
    -------
    Basis
        Basis object
    """
    return Basis(statistics, beta, wmax, eps)


def get_sampling_points(basis, domain='tau'):
    """
    Get default sampling points for a given domain.

    Parameters
    ----------
    basis : Basis
        Basis object
    domain : str
        Domain for sampling points ('tau', 'omega', or 'matsubara')

    Returns
    -------
    ndarray
        Default sampling points
    """
    if domain == 'tau':
        return basis._basis.default_tau_sampling_points()
    elif domain == 'omega':
        return basis._basis.default_omega_sampling_points()
    elif domain == 'matsubara':
        return basis._basis.default_matsubara_sampling_points()
    else:
        raise ValueError(f"Unknown domain: {domain}")


# For even more compatibility, provide some aliases
def basis(statistics, beta, wmax, eps=None):
    """Alias for create_basis."""
    return create_basis(statistics, beta, wmax, eps)


class IrBasisCompatibility:
    """
    Additional compatibility layer for specific irbasis patterns.

    This class can be used when more specific compatibility is needed
    beyond the basic Basis class.
    """

    def __init__(self, basis_obj):
        self.basis = basis_obj

    def get_basis_functions(self, domain='tau'):
        """Get basis functions for a specific domain."""
        if domain == 'tau':
            return self.basis.u
        elif domain == 'matsubara':
            return self.basis.uhat
        elif domain == 'omega':
            return self.basis.v
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def transform_to_ir(self, data, domain_from='tau'):
        """
        Transform data to IR coefficients.

        This is a placeholder - actual implementation would depend on
        the specific transformation being performed.
        """
        raise NotImplementedError("transform_to_ir not yet implemented")

    def transform_from_ir(self, coeffs, domain_to='tau'):
        """
        Transform IR coefficients to data.

        This is a placeholder - actual implementation would depend on
        the specific transformation being performed.
        """
        raise NotImplementedError("transform_from_ir not yet implemented")