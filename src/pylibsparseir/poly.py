"""
Piecewise polynomial functionality for SparseIR.

This module provides piecewise Legendre polynomial representation and
their Fourier transforms, which serve as core mathematical infrastructure
for IR basis functions.
"""

import numpy as np

from pylibsparseir.core import _lib


class FunctionSet:
    """Wrapper for basis function evaluation."""

    def __init__(self, funcs_ptr):
        self._ptr = funcs_ptr

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return _lib.funcs_evaluate(self._ptr, x)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.funcs_release(self._ptr)


class PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given points."""
        return self._funcs(x)

class PiecewiseLegendrePolyFT:
    """Piecewise Legendre polynomial Fourier transform."""

    def __init__(self, funcs: FunctionSet):
        self._funcs = funcs
        self._xmin = -1.0
        self._xmax = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given points."""
        return self._funcs(x)
