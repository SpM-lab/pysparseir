"""
Core functionality for the SparseIR Python bindings.
"""

import os
import sys
import ctypes
from ctypes import *
import numpy as np

# Define c_bool for compatibility
c_bool = c_byte

from .ctypes_wrapper import *
from .constants import *

def _find_library():
    """Find the SparseIR shared library."""
    if sys.platform == "darwin":
        libname = "libsparseir.dylib"
    elif sys.platform == "win32":
        libname = "sparseir.dll"
    else:
        libname = "libsparseir.so"

    # Try to find the library in common locations
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build"),
    ]

    for path in search_paths:
        libpath = os.path.join(path, libname)
        if os.path.exists(libpath):
            return libpath

    raise RuntimeError(f"Could not find {libname} in {search_paths}")

# Load the library
try:
    _lib = CDLL(_find_library())
except Exception as e:
    raise RuntimeError(f"Failed to load SparseIR library: {e}")

# Set up function prototypes
def _setup_prototypes():
    # Kernel functions
    _lib.spir_logistic_kernel_new.argtypes = [c_double, POINTER(c_int)]
    _lib.spir_logistic_kernel_new.restype = spir_kernel

    _lib.spir_reg_bose_kernel_new.argtypes = [c_double, POINTER(c_int)]
    _lib.spir_reg_bose_kernel_new.restype = spir_kernel

    _lib.spir_kernel_domain.argtypes = [
        spir_kernel, POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_kernel_domain.restype = c_int

    # SVE result functions
    _lib.spir_sve_result_new.argtypes = [spir_kernel, c_double, POINTER(c_int)]
    _lib.spir_sve_result_new.restype = spir_sve_result

    _lib.spir_sve_result_get_size.argtypes = [spir_sve_result, POINTER(c_int)]
    _lib.spir_sve_result_get_size.restype = c_int

    _lib.spir_sve_result_get_svals.argtypes = [spir_sve_result, POINTER(c_double)]
    _lib.spir_sve_result_get_svals.restype = c_int

    # Basis functions
    _lib.spir_basis_new.argtypes = [
        c_int, c_double, c_double, spir_kernel, spir_sve_result, POINTER(c_int)
    ]
    _lib.spir_basis_new.restype = spir_basis

    _lib.spir_basis_get_size.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_size.restype = c_int

    _lib.spir_basis_get_svals.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_svals.restype = c_int

    _lib.spir_basis_get_stats.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_stats.restype = c_int

    # Basis function objects
    _lib.spir_basis_get_u.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_u.restype = spir_funcs
    
    _lib.spir_basis_get_v.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_v.restype = spir_funcs
    
    _lib.spir_basis_get_uhat.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_uhat.restype = spir_funcs

    # Function evaluation
    _lib.spir_funcs_get_size.argtypes = [spir_funcs, POINTER(c_int)]
    _lib.spir_funcs_get_size.restype = c_int
    
    _lib.spir_funcs_eval.argtypes = [spir_funcs, c_double, POINTER(c_double)]
    _lib.spir_funcs_eval.restype = c_int
    
    _lib.spir_funcs_batch_eval.argtypes = [
        spir_funcs, c_int, c_size_t, POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_funcs_batch_eval.restype = c_int
    
    _lib.spir_funcs_batch_eval_matsu.argtypes = [
        spir_funcs, c_int, c_int, POINTER(c_int64), POINTER(c_double)
    ]
    _lib.spir_funcs_batch_eval_matsu.restype = c_int

    # Default sampling points
    _lib.spir_basis_get_n_default_taus.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_n_default_taus.restype = c_int
    
    _lib.spir_basis_get_default_taus.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_default_taus.restype = c_int
    
    _lib.spir_basis_get_n_default_ws.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_n_default_ws.restype = c_int
    
    _lib.spir_basis_get_default_ws.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_default_ws.restype = c_int
    
    _lib.spir_basis_get_n_default_matsus.argtypes = [spir_basis, c_bool, POINTER(c_int)]
    _lib.spir_basis_get_n_default_matsus.restype = c_int
    
    _lib.spir_basis_get_default_matsus.argtypes = [spir_basis, c_bool, POINTER(c_int64)]
    _lib.spir_basis_get_default_matsus.restype = c_int

    # Sampling objects
    _lib.spir_tau_sampling_new.argtypes = [spir_basis, c_int, POINTER(c_double), POINTER(c_int)]
    _lib.spir_tau_sampling_new.restype = spir_sampling
    
    _lib.spir_matsu_sampling_new.argtypes = [spir_basis, c_bool, c_int, POINTER(c_int64), POINTER(c_int)]
    _lib.spir_matsu_sampling_new.restype = spir_sampling
    
    # Sampling operations
    _lib.spir_sampling_eval_dd.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int, 
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_eval_dd.restype = c_int
    
    _lib.spir_sampling_fit_dd.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_fit_dd.restype = c_int

    # Additional sampling functions
    _lib.spir_sampling_get_npoints.argtypes = [spir_sampling, POINTER(c_int)]
    _lib.spir_sampling_get_npoints.restype = c_int
    
    _lib.spir_sampling_get_taus.argtypes = [spir_sampling, POINTER(c_double)]
    _lib.spir_sampling_get_taus.restype = c_int
    
    _lib.spir_sampling_get_matsus.argtypes = [spir_sampling, POINTER(c_int64)]
    _lib.spir_sampling_get_matsus.restype = c_int
    
    _lib.spir_sampling_get_cond_num.argtypes = [spir_sampling, POINTER(c_double)]
    _lib.spir_sampling_get_cond_num.restype = c_int
    
    # Multi-dimensional sampling evaluation functions
    _lib.spir_sampling_eval_dz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_eval_dz.restype = c_int
    
    _lib.spir_sampling_eval_zz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_eval_zz.restype = c_int
    
    _lib.spir_sampling_fit_zz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_fit_zz.restype = c_int

    # DLR functions
    _lib.spir_dlr_new.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_dlr_new.restype = spir_basis
    
    _lib.spir_dlr_new_with_poles.argtypes = [spir_basis, c_int, POINTER(c_double), POINTER(c_int)]
    _lib.spir_dlr_new_with_poles.restype = spir_basis
    
    _lib.spir_dlr_get_npoles.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_dlr_get_npoles.restype = c_int
    
    _lib.spir_dlr_get_poles.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_dlr_get_poles.restype = c_int
    
    _lib.spir_dlr2ir_dd.argtypes = [
        spir_basis, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_dlr2ir_dd.restype = c_int
    
    _lib.spir_dlr2ir_zz.argtypes = [
        spir_basis, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_dlr2ir_zz.restype = c_int

    # Release functions
    _lib.spir_kernel_release.argtypes = [spir_kernel]
    _lib.spir_kernel_release.restype = None
    
    _lib.spir_sve_result_release.argtypes = [spir_sve_result]
    _lib.spir_sve_result_release.restype = None
    
    _lib.spir_basis_release.argtypes = [spir_basis]
    _lib.spir_basis_release.restype = None
    
    _lib.spir_funcs_release.argtypes = [spir_funcs]
    _lib.spir_funcs_release.restype = None
    
    _lib.spir_sampling_release.argtypes = [spir_sampling]
    _lib.spir_sampling_release.restype = None

_setup_prototypes()

# Python wrapper functions
def logistic_kernel_new(lambda_val):
    """Create a new logistic kernel."""
    status = c_int()
    kernel = _lib.spir_logistic_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create logistic kernel: {status.value}")
    return kernel

def reg_bose_kernel_new(lambda_val):
    """Create a new regularized bosonic kernel."""
    status = c_int()
    kernel = _lib.spir_reg_bose_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create regularized bosonic kernel: {status.value}")
    return kernel

def kernel_domain(kernel):
    """Get the domain boundaries of a kernel."""
    xmin = c_double()
    xmax = c_double()
    ymin = c_double()
    ymax = c_double()

    status = _lib.spir_kernel_domain(
        kernel, byref(xmin), byref(xmax), byref(ymin), byref(ymax)
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get kernel domain: {status}")

    return xmin.value, xmax.value, ymin.value, ymax.value

def sve_result_new(kernel, epsilon):
    """Create a new SVE result."""
    # Validate epsilon
    if epsilon <= 0:
        raise RuntimeError(f"Failed to create SVE result: epsilon must be positive, got {epsilon}")
    
    status = c_int()
    sve = _lib.spir_sve_result_new(kernel, epsilon, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create SVE result: {status.value}")
    return sve

def sve_result_get_size(sve):
    """Get the size of an SVE result."""
    size = c_int()
    status = _lib.spir_sve_result_get_size(sve, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get SVE result size: {status}")
    return size.value

def sve_result_get_svals(sve):
    """Get the singular values from an SVE result."""
    size = sve_result_get_size(sve)
    svals = np.zeros(size, dtype=DOUBLE_DTYPE)
    status = _lib.spir_sve_result_get_svals(sve, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals

def basis_new(statistics, beta, omega_max, kernel, sve):
    """Create a new basis."""
    status = c_int()
    basis = _lib.spir_basis_new(
        statistics, beta, omega_max, kernel, sve, byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create basis: {status.value}")
    return basis

def basis_get_size(basis):
    """Get the size of a basis."""
    size = c_int()
    status = _lib.spir_basis_get_size(basis, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis size: {status}")
    return size.value

def basis_get_svals(basis):
    """Get the singular values of a basis."""
    size = basis_get_size(basis)
    svals = np.zeros(size, dtype=DOUBLE_DTYPE)
    status = _lib.spir_basis_get_svals(basis, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals

def basis_get_stats(basis):
    """Get the statistics type of a basis."""
    stats = c_int()
    status = _lib.spir_basis_get_stats(basis, byref(stats))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis statistics: {status}")
    return stats.value

def basis_get_u(basis):
    """Get the imaginary-time basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_u(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get u basis functions: {status.value}")
    return funcs

def basis_get_v(basis):
    """Get the real-frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_v(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get v basis functions: {status.value}")
    return funcs

def basis_get_uhat(basis):
    """Get the Matsubara frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_uhat(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get uhat basis functions: {status.value}")
    return funcs

def funcs_eval_single(funcs, x):
    """Evaluate basis functions at a single point."""
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")
    
    # Prepare output array
    out = np.zeros(size.value, dtype=np.float64)
    
    # Evaluate
    status = _lib.spir_funcs_eval(
        funcs, c_double(x),
        out.ctypes.data_as(POINTER(c_double))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")
    
    return out

def funcs_evaluate(funcs, x):
    """Evaluate basis functions at given points."""
    x = np.asarray(x, dtype=np.float64)
    n_points = len(x)
    
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")
    
    # Prepare output array
    out = np.zeros((size.value, n_points), dtype=np.float64)
    
    # Evaluate
    status = _lib.spir_funcs_batch_eval(
        funcs, 0, n_points, 
        x.ctypes.data_as(POINTER(c_double)),
        out.ctypes.data_as(POINTER(c_double))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")
    
    return out

def funcs_evaluate_matsubara(funcs, n):
    """Evaluate basis functions at Matsubara frequencies."""
    n = np.asarray(n, dtype=np.int64)
    n_freqs = len(n)
    
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")
    
    # For complex data, need to handle as double array with 2x size
    # Each complex number is 2 doubles (real, imag)
    out_doubles = np.zeros((n_freqs, size.value * 2), dtype=np.float64)
    
    # Evaluate at Matsubara frequencies
    status = _lib.spir_funcs_batch_eval_matsu(
        funcs, ORDER_ROW_MAJOR, n_freqs,
        n.ctypes.data_as(POINTER(c_int64)),
        out_doubles.ctypes.data_as(POINTER(c_double))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions at Matsubara frequencies: {status}")
    
    # Convert back to complex
    out_complex = np.zeros((n_freqs, size.value), dtype=np.complex128)
    out_complex.real = out_doubles[:, 0::2]  # Even indices are real parts
    out_complex.imag = out_doubles[:, 1::2]  # Odd indices are imaginary parts
    
    return out_complex

def basis_get_default_tau_sampling_points(basis):
    """Get default tau sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_taus(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default tau points: {status}")
    
    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_taus(basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default tau points: {status}")
    
    return points

def basis_get_default_omega_sampling_points(basis):
    """Get default omega (real frequency) sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_ws(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default omega points: {status}")
    
    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_ws(basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default omega points: {status}")
    
    return points

def basis_get_default_matsubara_sampling_points(basis, positive_only=False):
    """Get default Matsubara sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_matsus(basis, c_bool(positive_only), byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default Matsubara points: {status}")
    
    # Get the points
    points = np.zeros(n_points.value, dtype=np.int64)
    status = _lib.spir_basis_get_default_matsus(basis, c_bool(positive_only), points.ctypes.data_as(POINTER(c_int64)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default Matsubara points: {status}")
    
    return points

def tau_sampling_new(basis, sampling_points=None):
    """Create a new tau sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_tau_sampling_points(basis)
    
    sampling_points = np.asarray(sampling_points, dtype=np.float64)
    n_points = len(sampling_points)
    
    status = c_int()
    sampling = _lib.spir_tau_sampling_new(
        basis, n_points, 
        sampling_points.ctypes.data_as(POINTER(c_double)), 
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create tau sampling: {status.value}")
    
    return sampling

def matsubara_sampling_new(basis, positive_only=False, sampling_points=None):
    """Create a new Matsubara sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_matsubara_sampling_points(basis, positive_only)
    
    sampling_points = np.asarray(sampling_points, dtype=np.int64)
    n_points = len(sampling_points)
    
    status = c_int()
    sampling = _lib.spir_matsu_sampling_new(
        basis, c_bool(positive_only), n_points,
        sampling_points.ctypes.data_as(POINTER(c_int64)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create Matsubara sampling: {status.value}")
    
    return sampling