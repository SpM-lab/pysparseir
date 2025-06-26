"""
High-level Python classes for sparse sampling
"""

import numpy as np
from ctypes import *
from .core import *
from .constants import *
from .core import _lib


class TauSampling:
    """Sparse sampling in imaginary time."""
    
    def __init__(self, basis, sampling_points=None):
        """
        Initialize tau sampling.
        
        Parameters:
        -----------
        basis : FiniteTempBasis
            Finite temperature basis
        sampling_points : array_like, optional
            Tau sampling points. If None, use default.
        """
        self.basis = basis
        
        if sampling_points is None:
            self.sampling_points = basis.default_tau_sampling_points()
        else:
            self.sampling_points = np.asarray(sampling_points, dtype=np.float64)
        
        # Create sampling object
        self._sampling = tau_sampling_new(basis._basis, self.sampling_points)
    
    @property
    def tau(self):
        """Tau sampling points."""
        return self.sampling_points
    
    def evaluate(self, al, axis=None):
        """
        Transform basis coefficients to sampling points.
        
        Parameters:
        -----------
        al : array_like
            Basis coefficients
        axis : int, optional
            Axis along which to transform
            
        Returns:
        --------
        ndarray
            Values at sampling points
        """
        al = np.asarray(al, dtype=np.float64)
        
        # Simple case: 1D array
        if al.ndim == 1:
            if len(al) != self.basis.size:
                raise ValueError(f"Expected {self.basis.size} coefficients, got {len(al)}")
            
            result = np.zeros(len(self.sampling_points), dtype=np.float64)
            
            # Call C function for evaluation
            dims = np.array([self.basis.size], dtype=np.int32)
            status = _lib.spir_sampling_eval_dd(
                self._sampling, 0, 1, dims.ctypes.data_as(POINTER(c_int)), 0,
                al.ctypes.data_as(POINTER(c_double)),
                result.ctypes.data_as(POINTER(c_double))
            )
            if status != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to evaluate sampling: {status}")
            
            return result
        else:
            raise NotImplementedError("Multi-dimensional arrays not yet implemented")
    
    def fit(self, ax, axis=None):
        """
        Fit basis coefficients from sampling point values.
        
        Parameters:
        -----------
        ax : array_like
            Values at sampling points
        axis : int, optional
            Axis along which to fit
            
        Returns:
        --------
        ndarray
            Basis coefficients
        """
        ax = np.asarray(ax, dtype=np.float64)
        
        # Simple case: 1D array
        if ax.ndim == 1:
            if len(ax) != len(self.sampling_points):
                raise ValueError(f"Expected {len(self.sampling_points)} values, got {len(ax)}")
            
            result = np.zeros(self.basis.size, dtype=np.float64)
            
            # Call C function for fitting
            dims = np.array([len(self.sampling_points)], dtype=np.int32)
            status = _lib.spir_sampling_fit_dd(
                self._sampling, 0, 1, dims.ctypes.data_as(POINTER(c_int)), 0,
                ax.ctypes.data_as(POINTER(c_double)),
                result.ctypes.data_as(POINTER(c_double))
            )
            if status != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to fit sampling: {status}")
            
            return result
        else:
            raise NotImplementedError("Multi-dimensional arrays not yet implemented")
    
    def __repr__(self):
        return f"TauSampling(n_points={len(self.sampling_points)})"


class MatsubaraSampling:
    """Sparse sampling in Matsubara frequencies."""
    
    def __init__(self, basis, sampling_points=None, positive_only=False):
        """
        Initialize Matsubara sampling.
        
        Parameters:
        -----------
        basis : FiniteTempBasis
            Finite temperature basis
        sampling_points : array_like, optional
            Matsubara frequency indices. If None, use default.
        positive_only : bool, optional
            If True, use only positive frequencies
        """
        self.basis = basis
        self.positive_only = positive_only
        
        if sampling_points is None:
            self.sampling_points = basis.default_matsubara_sampling_points(positive_only=positive_only)
        else:
            self.sampling_points = np.asarray(sampling_points, dtype=np.int64)
        
        # Create sampling object
        self._sampling = matsubara_sampling_new(basis._basis, positive_only, self.sampling_points)
    
    @property 
    def wn(self):
        """Matsubara frequency indices."""
        return self.sampling_points
    
    def evaluate(self, al, axis=None):
        """
        Transform basis coefficients to sampling points.
        
        Parameters:
        -----------
        al : array_like
            Basis coefficients
        axis : int, optional
            Axis along which to transform
            
        Returns:
        --------
        ndarray
            Values at Matsubara frequencies (complex)
        """
        al = np.asarray(al, dtype=np.complex128)
        
        # Simple case: 1D array
        if al.ndim == 1:
            if len(al) != self.basis.size:
                raise ValueError(f"Expected {self.basis.size} coefficients, got {len(al)}")
            
            result = np.zeros(len(self.sampling_points), dtype=np.complex128)
            
            # For complex data, we need to use the complex version
            # This is a simplified implementation - may need adjustment based on actual C API
            raise NotImplementedError("Complex Matsubara evaluation not yet implemented")
        else:
            raise NotImplementedError("Multi-dimensional arrays not yet implemented")
    
    def fit(self, ax, axis=None):
        """
        Fit basis coefficients from Matsubara frequency values.
        
        Parameters:
        -----------
        ax : array_like
            Values at Matsubara frequencies
        axis : int, optional
            Axis along which to fit
            
        Returns:
        --------
        ndarray
            Basis coefficients
        """
        ax = np.asarray(ax, dtype=np.complex128)
        
        # Simple case: 1D array
        if ax.ndim == 1:
            if len(ax) != len(self.sampling_points):
                raise ValueError(f"Expected {len(self.sampling_points)} values, got {len(ax)}")
            
            # For complex data, we need to use the complex version
            # This is a simplified implementation - may need adjustment based on actual C API
            raise NotImplementedError("Complex Matsubara fitting not yet implemented")
        else:
            raise NotImplementedError("Multi-dimensional arrays not yet implemented")
    
    def __repr__(self):
        return f"MatsubaraSampling(n_points={len(self.sampling_points)}, positive_only={self.positive_only})"