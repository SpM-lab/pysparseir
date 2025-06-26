"""
Kernel classes for SparseIR.

This module provides Python wrappers for kernel objects from the C library.
"""

import ctypes
from ctypes import c_int, c_double, byref
import numpy as np

from .ctypes_wrapper import spir_kernel
from .core import _lib
from .constants import COMPUTATION_SUCCESS


class LogisticKernel:
    """
    Fermionic/logistic imaginary-time kernel.
    
    This kernel treats a fermionic spectral function at finite temperature.
    The definition is:
    
        K(τ, ω) = exp(-τ ω) / (1 + exp(-β ω))
    
    with τ ∈ [0, β] and ω ∈ [-ωmax, ωmax].
    
    Parameters
    ----------
    lambda_ : float
        Kernel cutoff Λ = β * ωmax
    """
    
    def __init__(self, lambda_):
        """Initialize logistic kernel with cutoff lambda."""
        self._lambda = float(lambda_)
        status = c_int()
        self._kernel = _lib.spir_logistic_kernel_new(self._lambda, byref(status))
        if status.value != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to create logistic kernel: {status.value}")
    
    @property
    def lambda_(self):
        """Kernel cutoff."""
        return self._lambda
    
    @property
    def beta(self):
        """Inverse temperature (always 1 for scaled kernels)."""
        return 1.0
    
    @property
    def wmax(self):
        """Frequency cutoff."""
        return self._lambda
    
    def __call__(self, tau, omega=None):
        """
        Evaluate kernel at given tau and omega values.
        
        Parameters
        ----------
        tau : array_like
            Imaginary time values
        omega : array_like, optional
            Frequency values. If None, returns kernel matrix for default omega grid.
            
        Returns
        -------
        ndarray
            Kernel values
        """
        # TODO: Implement kernel evaluation
        raise NotImplementedError("Kernel evaluation not yet implemented")
    
    def __del__(self):
        """Clean up kernel resources."""
        if hasattr(self, '_kernel') and self._kernel:
            # TODO: Add kernel cleanup function when available in C API
            pass


class RegularizedBoseKernel:
    """
    Bosonic imaginary-time kernel.
    
    This kernel treats a bosonic spectral function at finite temperature.
    The definition is:
    
        K(τ, ω) = ω exp(-τ ω) / (1 - exp(-β ω))
    
    with τ ∈ [0, β] and ω ∈ [-ωmax, ωmax]. The kernel is regularized
    at ω = 0 to avoid the singularity.
    
    Parameters
    ----------
    lambda_ : float
        Kernel cutoff Λ = β * ωmax
    """
    
    def __init__(self, lambda_):
        """Initialize regularized bosonic kernel with cutoff lambda."""
        self._lambda = float(lambda_)
        status = c_int()
        self._kernel = _lib.spir_reg_bose_kernel_new(self._lambda, byref(status))
        if status.value != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to create regularized bosonic kernel: {status.value}")
    
    @property
    def lambda_(self):
        """Kernel cutoff."""
        return self._lambda
    
    @property
    def beta(self):
        """Inverse temperature (always 1 for scaled kernels)."""
        return 1.0
    
    @property
    def wmax(self):
        """Frequency cutoff."""
        return self._lambda
    
    def __call__(self, tau, omega=None):
        """
        Evaluate kernel at given tau and omega values.
        
        Parameters
        ----------
        tau : array_like
            Imaginary time values
        omega : array_like, optional
            Frequency values. If None, returns kernel matrix for default omega grid.
            
        Returns
        -------
        ndarray
            Kernel values
        """
        # TODO: Implement kernel evaluation
        raise NotImplementedError("Kernel evaluation not yet implemented")
    
    def __del__(self):
        """Clean up kernel resources."""
        if hasattr(self, '_kernel') and self._kernel:
            # TODO: Add kernel cleanup function when available in C API
            pass