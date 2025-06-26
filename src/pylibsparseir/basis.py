"""
High-level Python classes for FiniteTempBasis
"""

import numpy as np
from ctypes import *
from .core import *
from .constants import *
from .core import _lib
from .abstract import AbstractBasis


class BasisFunctions:
    """Wrapper for basis function evaluation."""
    
    def __init__(self, funcs_ptr, func_type='u'):
        self._funcs = funcs_ptr
        self.func_type = func_type
    
    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if self.func_type == 'uhat':
            # For Matsubara frequencies, use integer indices
            x = np.asarray(x, dtype=np.int32)
            return self._evaluate_matsubara(x)
        else:
            # For tau and omega, use float values
            x = np.asarray(x, dtype=np.float64)
            return funcs_evaluate(self._funcs, x)
    
    def _evaluate_matsubara(self, n):
        """Evaluate at Matsubara frequencies."""
        from .core import funcs_evaluate_matsubara
        return funcs_evaluate_matsubara(self._funcs, n).T  # Transpose to match expected shape


class FiniteTempBasis(AbstractBasis):
    """Finite temperature basis for intermediate representation."""
    
    def __init__(self, statistics, beta, wmax, eps=None, **kwargs):
        """
        Initialize finite temperature basis.
        
        Parameters:
        -----------
        statistics : str
            'F' for fermions, 'B' for bosons
        beta : float
            Inverse temperature
        wmax : float
            Frequency cutoff
        eps : float, optional
            Accuracy threshold
        """
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._lambda = beta * wmax
        
        if eps is None:
            eps = 1e-12
        
        # Create kernel
        if statistics == 'F' or statistics == 'B':
            self._kernel = logistic_kernel_new(self._lambda)
        else:
            raise ValueError(f"Invalid statistics: {statistics}")
        
        # Compute SVE
        self._sve = sve_result_new(self._kernel, eps)
        
        # Create basis
        stats_int = 1 if statistics == 'F' else 0  # 1=fermion, 0=boson
        self._basis = basis_new(stats_int, self._beta, self._wmax, self._kernel, self._sve)
        
        # Cache properties
        self._size = None
        self._s = None
        self._u = None
        self._v = None
        self._uhat = None
    
    @property
    def statistics(self):
        """Quantum statistic ('F' for fermionic, 'B' for bosonic)"""
        return self._statistics
    
    @property
    def beta(self):
        """Inverse temperature"""
        return self._beta
    
    @property
    def wmax(self):
        """Real frequency cutoff"""
        return self._wmax
    
    @property
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax"""
        return self._lambda
    
    @property
    def size(self):
        """Number of basis functions."""
        if self._size is None:
            self._size = basis_get_size(self._basis)
        return self._size
    
    @property
    def s(self):
        """Singular values."""
        if self._s is None:
            self._s = basis_get_svals(self._basis)
        return self._s
    
    @property
    def u(self):
        """Imaginary-time basis functions."""
        if self._u is None:
            u_funcs = basis_get_u(self._basis)
            self._u = BasisFunctions(u_funcs, 'u')
        return self._u
    
    @property
    def v(self):
        """Real-frequency basis functions."""
        if self._v is None:
            v_funcs = basis_get_v(self._basis)
            self._v = BasisFunctions(v_funcs, 'v')
        return self._v
    
    @property
    def uhat(self):
        """Matsubara frequency basis functions."""
        if self._uhat is None:
            uhat_funcs = basis_get_uhat(self._basis)
            self._uhat = BasisFunctions(uhat_funcs, 'uhat')
        return self._uhat
    
    @property
    def significance(self):
        """Relative significance of basis functions."""
        return self.s / self.s[0]
    
    @property
    def accuracy(self):
        """Overall accuracy bound."""
        return self.s[-1] / self.s[0]
    
    def default_tau_sampling_points(self, npoints=None):
        """Get default tau sampling points."""
        return basis_get_default_tau_sampling_points(self._basis)
    
    def default_omega_sampling_points(self, npoints=None):
        """
        Get default omega (real frequency) sampling points.
        
        Returns the extrema of the highest-order basis function in real frequency.
        These points provide near-optimal conditioning for the basis.
        
        Parameters
        ----------
        npoints : int, optional
            Ignored (for compatibility with sparse-ir API)
            
        Returns
        -------
        ndarray
            Default omega sampling points
        """
        from .core import basis_get_default_omega_sampling_points
        return basis_get_default_omega_sampling_points(self._basis)
    
    def default_matsubara_sampling_points(self, npoints=None, positive_only=False):
        """Get default Matsubara sampling points."""
        return basis_get_default_matsubara_sampling_points(self._basis, positive_only)
    
    def __repr__(self):
        return (f"FiniteTempBasis(statistics='{self.statistics}', "
                f"beta={self.beta}, wmax={self.wmax}, size={self.size})")
    
    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.
        
        This can be used to truncate the basis to the n most significant
        singular values: basis[:3].
        """
        # TODO: Implement basis truncation when C API supports it
        raise NotImplementedError("Basis truncation not yet implemented in C API")


def finite_temp_bases(beta, wmax, eps=None, **kwargs):
    """
    Construct both fermion and boson bases.
    
    Returns:
    --------
    tuple
        (fermion_basis, boson_basis)
    """
    fermion_basis = FiniteTempBasis('F', beta, wmax, eps, **kwargs)
    boson_basis = FiniteTempBasis('B', beta, wmax, eps, **kwargs)
    return fermion_basis, boson_basis