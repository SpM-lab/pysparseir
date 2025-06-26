"""
SparseIR Python bindings

This package provides Python bindings for the SparseIR C library.
"""

from .core import *
from .constants import *
from .ctypes_wrapper import *
from .abstract import AbstractBasis
from .basis import FiniteTempBasis, finite_temp_bases
from .sampling import TauSampling, MatsubaraSampling
from .kernel import LogisticKernel, RegularizedBoseKernel
from .sve import SVEResult, compute_sve
from .basis_set import FiniteTempBasisSet

# New augmented functionality
from .augment import (
    AugmentedBasis, AugmentedTauFunction, AugmentedMatsubaraFunction,
    AbstractAugmentation, TauConst, TauLinear, MatsubaraConst
)

# DLR functionality
from .dlr import (
    DiscreteLehmannRepresentation, TauPoles, MatsubaraPoles
)

# Compatibility adapter
from .adapter import (
    Basis as AdapterBasis, load, sampling_points_x, sampling_points_y,
    sampling_points_matsubara, create_basis, get_sampling_points
)

# Polynomial functionality
from .poly import (
    PiecewiseLegendrePoly, PiecewiseLegendreFT, legendre_basis, fit_legendre
)

# Enhanced SVD functionality
from .svd_enhanced import (
    compute as compute_svd_enhanced, truncated_svd, adaptive_rank_svd,
    condition_number, effective_rank, SVDResult as SVDResultEnhanced
)

# Get version from the library
try:
    __version__ = ".".join(map(str, get_version()))
except Exception:
    # Fallback version if library is not available
    __version__ = "0.0.1"

# Backward compatibility aliases
compute = compute_sve  # Alias for backward compatibility

# Export list for better documentation
__all__ = [
    # Core functionality
    'AbstractBasis', 'FiniteTempBasis', 'finite_temp_bases',
    'TauSampling', 'MatsubaraSampling', 'FiniteTempBasisSet',
    'LogisticKernel', 'RegularizedBoseKernel',
    'SVEResult', 'compute_sve', 'compute',

    # Augmented functionality
    'AugmentedBasis', 'AugmentedTauFunction', 'AugmentedMatsubaraFunction',
    'AbstractAugmentation', 'TauConst', 'TauLinear', 'MatsubaraConst',

    # DLR functionality
    'DiscreteLehmannRepresentation', 'TauPoles', 'MatsubaraPoles',

    # Adapter functionality
    'AdapterBasis', 'load', 'sampling_points_x', 'sampling_points_y',
    'sampling_points_matsubara', 'create_basis', 'get_sampling_points',

    # Polynomial functionality
    'PiecewiseLegendrePoly', 'PiecewiseLegendreFT', 'legendre_basis', 'fit_legendre',

    # Enhanced SVD functionality
    'compute_svd_enhanced', 'truncated_svd', 'adaptive_rank_svd',
    'condition_number', 'effective_rank', 'SVDResultEnhanced',
]