"""
Piecewise polynomial functionality for SparseIR.

This module provides piecewise Legendre polynomial representation and
their Fourier transforms, which serve as core mathematical infrastructure
for IR basis functions.
"""

from ctypes import c_int, POINTER
import numpy as np

from pylibsparseir.core import _lib
from pylibsparseir.core import funcs_eval_single_float64, funcs_eval_single_complex128
from pylibsparseir.core import funcs_get_size

def funcs_get_slice(funcs_ptr, indices):
    status = c_int()
    indices = np.asarray(indices, dtype=np.int32)
    funcs = _lib.spir_funcs_get_slice(funcs_ptr, len(indices), indices.ctypes.data_as(POINTER(c_int)), status)
    if status.value != 0:
        raise RuntimeError(f"Failed to get basis function {indices}: {status.value}")
    return FunctionSet(funcs)

def funcs_ft_get_slice(funcs_ptr, indices):
    status = c_int()
    indices = np.asarray(indices, dtype=np.int32)
    funcs = _lib.spir_funcs_get_slice(funcs_ptr, len(indices), indices.ctypes.data_as(POINTER(c_int)), status)
    if status.value != 0:
        raise RuntimeError(f"Failed to get basis function {indices}: {status.value}")
    return FunctionSetFT(funcs)

class FunctionSet:
    """Wrapper for basis function evaluation."""

    def __init__(self, funcs_ptr):
        self._ptr = funcs_ptr

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if not isinstance(x, np.ndarray):
            o = funcs_eval_single_float64(self._ptr, x)
            if len(o) == 1:
                return o[0]
            else:
                return o
        else:
            o = np.stack([funcs_eval_single_float64(self._ptr, e) for e in x]).T
            if len(o) == 1:
                return o[0]
            else:
                return o

    def __getitem__(self, index):
        """Get a single basis function."""
        sz = funcs_get_size(self._ptr)
        return funcs_get_slice(self._ptr, [index % sz])

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_funcs_release(self._ptr)

class FunctionSetFT:
    """Wrapper for basis function evaluation."""

    def __init__(self, funcs_ptr):
        self._ptr = funcs_ptr

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if not isinstance(x, np.ndarray):
            o = funcs_eval_single_complex128(self._ptr, x)
            if len(o) == 1:
                return o[0]
            else:
                return o
        else:
            o = np.stack([funcs_eval_single_complex128(self._ptr, e) for e in x]).T
            if len(o) == 1:
                return o[0]
            else:
                return o

    def __getitem__(self, index):
        """Get a single basis function."""
        sz = funcs_get_size(self._ptr)
        return funcs_ft_get_slice(self._ptr, [index % sz])

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_funcs_release(self._ptr)

class PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

class PiecewiseLegendrePolyVector:
    """Piecewise Legendre polynomial."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function."""
        return PiecewiseLegendrePoly(self._funcs[index], self._xmin, self._xmax)


class PiecewiseLegendrePolyFT:
    """Piecewise Legendre polynomial Fourier transform."""

    def __init__(self, funcs: FunctionSetFT):
        print(type(funcs))

        assert isinstance(funcs, FunctionSetFT), "funcs must be a FunctionSetFT"
        self._funcs = funcs
        self._xmin = -1.0
        self._xmax = 1.0

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

class PiecewiseLegendrePolyFTVector:
    """Piecewise Legendre polynomial Fourier transform."""

    def __init__(self, funcs: FunctionSetFT):
        print(type(funcs))

        assert isinstance(funcs, FunctionSetFT), "funcs must be a FunctionSetFT"
        self._funcs = funcs
        self._xmin = -1.0
        self._xmax = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function."""
        return PiecewiseLegendrePolyFT(self._funcs[index])
