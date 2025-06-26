"""
SVE (Singular Value Expansion) functionality for SparseIR.

This module provides Python wrappers for SVE computation and results.
"""

import ctypes
from ctypes import c_int, c_double, byref, POINTER
import numpy as np

from .ctypes_wrapper import spir_sve_result, spir_funcs, DOUBLE_DTYPE
from .core import _lib
from .constants import COMPUTATION_SUCCESS, ORDER_ROW_MAJOR
from .kernel import LogisticKernel, RegularizedBoseKernel


class FunctionSet:
    """
    Represents a set of functions (u, v, or uhat basis functions).
    
    This class provides methods to evaluate basis functions at single
    or multiple points.
    """
    
    def __init__(self, funcs_ptr):
        """
        Initialize with a C funcs pointer.
        
        Parameters
        ----------
        funcs_ptr : spir_funcs
            Pointer to C funcs object
        """
        self._funcs = funcs_ptr
        
        # Get size
        size = c_int()
        status = _lib.spir_funcs_get_size(self._funcs, byref(size))
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to get function set size: {status}")
        self._size = size.value
    
    def __call__(self, x):
        """
        Evaluate functions at point(s) x.
        
        Parameters
        ----------
        x : float or array_like
            Point(s) at which to evaluate
            
        Returns
        -------
        ndarray
            Function values. If x is scalar, returns 1D array of length size.
            If x is array, returns 2D array of shape (len(x), size).
        """
        x = np.asarray(x)
        if x.ndim == 0:
            # Single point evaluation
            out = np.zeros(self._size, dtype=DOUBLE_DTYPE)
            status = _lib.spir_funcs_eval(
                self._funcs, float(x), out.ctypes.data_as(POINTER(c_double))
            )
            if status != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to evaluate functions: {status}")
            return out
        else:
            # Multiple points evaluation
            x = x.flatten()
            out = np.zeros((len(x), self._size), dtype=DOUBLE_DTYPE)
            status = _lib.spir_funcs_batch_eval(
                self._funcs, ORDER_ROW_MAJOR, len(x),
                x.ctypes.data_as(POINTER(c_double)),
                out.ctypes.data_as(POINTER(c_double))
            )
            if status != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to evaluate functions: {status}")
            return out.reshape(x.shape + (self._size,))
    
    @property
    def size(self):
        """Number of functions in this set."""
        return self._size
    
    def __len__(self):
        """Number of functions in this set."""
        return self._size
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_funcs') and self._funcs:
            # TODO: Add cleanup when available in C API
            pass


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
        
        self._kernel = kernel  # Store kernel for later use
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
        
        # Cache singular values and functions
        self._s = None
        self._u = None
        self._v = None
    
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
        if self._u is None:
            # Note: The C API doesn't provide direct access to u/v from SVEResult.
            # In the pure Python implementation, these are computed during SVE.
            # For now, we raise NotImplementedError with an informative message.
            raise NotImplementedError(
                "Direct access to u functions from SVEResult is not available in the C API. "
                "Please create a FiniteTempBasis object to access u functions:\n"
                "  basis = FiniteTempBasis(statistics, beta, wmax, kernel=kernel, sve=sve_result)\n"
                "  u_functions = basis.u"
            )
        return self._u
    
    @property
    def v(self):
        """Right singular functions (omega basis)."""
        if self._v is None:
            # Note: The C API doesn't provide direct access to u/v from SVEResult.
            # In the pure Python implementation, these are computed during SVE.
            # For now, we raise NotImplementedError with an informative message.
            raise NotImplementedError(
                "Direct access to v functions from SVEResult is not available in the C API. "
                "Please create a FiniteTempBasis object to access v functions:\n"
                "  basis = FiniteTempBasis(statistics, beta, wmax, kernel=kernel, sve=sve_result)\n"
                "  v_functions = basis.v"
            )
        return self._v
    
    def part(self):
        """
        Return the (u, s, v) tuple.
        
        Returns
        -------
        tuple
            (u, s, v) where u and v are function objects and s are singular values
            
        Raises
        ------
        NotImplementedError
            The C API doesn't provide direct access to u/v from SVEResult
        """
        return (self.u, self.s, self.v)
    
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