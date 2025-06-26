"""
SVE (Singular Value Expansion) functionality for SparseIR.

This module provides Python wrappers for SVE computation and results.
"""

import ctypes
from ctypes import c_int, c_double, byref, POINTER
import numpy as np

from .ctypes_wrapper import spir_sve_result, DOUBLE_DTYPE
from .core import _lib
from .constants import COMPUTATION_SUCCESS
from .kernel import LogisticKernel, RegularizedBoseKernel


class SVEResult:
    """
    Result of a singular value expansion (SVE).
    
    Contains the singular values and basis functions resulting from
    the SVE of an integral kernel.
    
    Attributes
    ----------
    s : ndarray
        Singular values in descending order
    u : object
        Left singular functions (tau basis)
    v : object
        Right singular functions (omega basis)
    """
    
    def __init__(self, kernel, epsilon):
        """
        Compute SVE of the given kernel.
        
        Parameters
        ----------
        kernel : LogisticKernel or RegularizedBoseKernel
            Kernel to compute SVE for
        epsilon : float
            Desired accuracy of the expansion
        """
        if not isinstance(kernel, (LogisticKernel, RegularizedBoseKernel)):
            raise TypeError("kernel must be LogisticKernel or RegularizedBoseKernel")
        
        self._epsilon = float(epsilon)
        
        # Validate epsilon
        if self._epsilon <= 0:
            raise RuntimeError(f"Failed to create SVE result: epsilon must be positive, got {self._epsilon}")
        
        # Get the C kernel object
        c_kernel = kernel._kernel
        
        # Create SVE result
        status = c_int()
        self._sve = _lib.spir_sve_result_new(c_kernel, self._epsilon, byref(status))
        if status.value != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to create SVE result: {status.value}")
        
        # Get size
        self._size = self._get_size()
        
        # Cache singular values
        self._s = None
    
    def _get_size(self):
        """Get the number of singular values."""
        size = c_int()
        status = _lib.spir_sve_result_get_size(self._sve, byref(size))
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to get SVE result size: {status}")
        return size.value
    
    @property
    def s(self):
        """Singular values in descending order."""
        if self._s is None:
            self._s = np.zeros(self._size, dtype=DOUBLE_DTYPE)
            status = _lib.spir_sve_result_get_svals(
                self._sve, self._s.ctypes.data_as(POINTER(c_double))
            )
            if status != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to get singular values: {status}")
        return self._s
    
    @property
    def u(self):
        """Left singular functions (tau basis)."""
        # TODO: Implement when C API provides access to u functions
        raise NotImplementedError("Access to u functions not yet implemented")
    
    @property
    def v(self):
        """Right singular functions (omega basis)."""
        # TODO: Implement when C API provides access to v functions
        raise NotImplementedError("Access to v functions not yet implemented")
    
    def __len__(self):
        """Number of terms in the expansion."""
        return self._size
    
    def __del__(self):
        """Clean up SVE resources."""
        if hasattr(self, '_sve') and self._sve:
            # TODO: Add SVE cleanup function when available in C API
            pass


def compute_sve(kernel, epsilon):
    """
    Compute singular value expansion of a kernel.
    
    Parameters
    ----------
    kernel : LogisticKernel or RegularizedBoseKernel
        Kernel to compute SVE for
    epsilon : float
        Desired accuracy of the expansion
        
    Returns
    -------
    SVEResult
        Result of the singular value expansion
    """
    return SVEResult(kernel, epsilon)