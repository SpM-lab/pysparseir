"""
Test core C API functionality - kernels, SVE, and basis functions.
Fixed version based on correct C API signatures.
"""
import pytest
import numpy as np
from ctypes import *
from pylibsparseir.core import _lib
from pylibsparseir.ctypes_wrapper import *
from pylibsparseir.constants import *


class TestCAPICoreFixed:
    """Test core C API functions with correct signatures"""
    
    def test_kernel_creation_and_domain(self):
        """Test kernel operations and domain retrieval"""
        # Test logistic kernel
        status = c_int()
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert kernel is not None
        
        # Test kernel domain
        xmin = c_double()
        xmax = c_double()
        ymin = c_double()
        ymax = c_double()
        status_val = _lib.spir_kernel_domain(kernel, byref(xmin), byref(xmax), 
                                            byref(ymin), byref(ymax))
        assert status_val == COMPUTATION_SUCCESS
        assert xmin.value == pytest.approx(-1.0)
        assert xmax.value == pytest.approx(1.0)
        
        # Release kernel
        _lib.spir_kernel_release(kernel)
        
        # Test regularized boson kernel
        kernel = _lib.spir_reg_bose_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert kernel is not None
        
        _lib.spir_kernel_domain(kernel, byref(xmin), byref(xmax), 
                               byref(ymin), byref(ymax))
        assert status_val == COMPUTATION_SUCCESS
        assert xmin.value == pytest.approx(-1.0)
        assert xmax.value == pytest.approx(1.0)
        
        _lib.spir_kernel_release(kernel)
    
    def test_sve_computation(self):
        """Test SVE computation"""
        status = c_int()
        
        # Create kernel
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        
        # Compute SVE
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-6), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert sve is not None
        
        # Get SVE size
        size = c_int()
        status_val = _lib.spir_sve_result_get_size(sve, byref(size))
        assert status_val == COMPUTATION_SUCCESS
        assert size.value > 0
        
        # Get singular values
        svals = np.zeros(size.value, dtype=np.float64)
        status_val = _lib.spir_sve_result_get_svals(sve, svals.ctypes.data_as(POINTER(c_double)))
        assert status_val == COMPUTATION_SUCCESS
        assert not np.any(np.isnan(svals))
        assert np.all(svals > 0)  # Singular values should be positive
        assert np.all(np.diff(svals) <= 0)  # Should be in descending order
        
        # Cleanup
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)
    
    def test_basis_constructors(self):
        """Test basis construction for different statistics"""
        for stats, stats_val in [("fermionic", STATISTICS_FERMIONIC), 
                                ("bosonic", STATISTICS_BOSONIC)]:
            status = c_int()
            beta = 2.0
            wmax = 1.0
            
            # Create kernel
            kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
            assert status.value == COMPUTATION_SUCCESS
            
            # Compute SVE
            sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), byref(status))
            assert status.value == COMPUTATION_SUCCESS
            
            # Create basis
            basis = _lib.spir_basis_new(c_int(stats_val), c_double(beta), c_double(wmax), 
                                      kernel, sve, byref(status))
            assert status.value == COMPUTATION_SUCCESS
            assert basis is not None
            
            # Check basis properties
            size = c_int()
            status_val = _lib.spir_basis_get_size(basis, byref(size))
            assert status_val == COMPUTATION_SUCCESS
            assert size.value > 0
            
            retrieved_stats = c_int()
            status_val = _lib.spir_basis_get_stats(basis, byref(retrieved_stats))
            assert status_val == COMPUTATION_SUCCESS
            assert retrieved_stats.value == stats_val
            
            # Get singular values
            svals = np.zeros(size.value, dtype=np.float64)
            status_val = _lib.spir_basis_get_svals(basis, svals.ctypes.data_as(POINTER(c_double)))
            assert status_val == COMPUTATION_SUCCESS
            assert not np.any(np.isnan(svals))
            
            # Cleanup
            _lib.spir_basis_release(basis)
            _lib.spir_sve_result_release(sve)
            _lib.spir_kernel_release(kernel)
    
    def test_tau_sampling_creation_and_properties(self):
        """Test TauSampling creation and basic properties"""
        status = c_int()
        beta = 10.0
        wmax = 1.0
        
        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), byref(status))
        basis = _lib.spir_basis_new(c_int(STATISTICS_FERMIONIC), c_double(beta), 
                                   c_double(wmax), kernel, sve, byref(status))
        
        # Get default tau points
        n_tau = c_int()
        status_val = _lib.spir_basis_get_n_default_taus(basis, byref(n_tau))
        assert status_val == COMPUTATION_SUCCESS
        assert n_tau.value > 0
        
        default_taus = np.zeros(n_tau.value, dtype=np.float64)
        status_val = _lib.spir_basis_get_default_taus(basis, 
                                                     default_taus.ctypes.data_as(POINTER(c_double)))
        assert status_val == COMPUTATION_SUCCESS
        
        # Create tau sampling
        tau_sampling = _lib.spir_tau_sampling_new(basis, n_tau.value, 
                                                 default_taus.ctypes.data_as(POINTER(c_double)), 
                                                 byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert tau_sampling is not None
        
        # Verify tau points are in valid range [0, beta]
        assert np.all(default_taus >= -beta/2)  # Transformed coordinates
        assert np.all(default_taus <= beta/2)
        
        # Cleanup
        _lib.spir_sampling_release(tau_sampling)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)
    
    def test_matsubara_sampling_creation(self):
        """Test MatsubaraSampling creation"""
        status = c_int()
        beta = 10.0
        wmax = 1.0
        
        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), byref(status))
        basis = _lib.spir_basis_new(c_int(STATISTICS_FERMIONIC), c_double(beta), 
                                   c_double(wmax), kernel, sve, byref(status))
        
        # Get default Matsubara points
        n_matsu = c_int()
        status_val = _lib.spir_basis_get_n_default_matsus(basis, c_bool(False), byref(n_matsu))
        assert status_val == COMPUTATION_SUCCESS
        assert n_matsu.value > 0
        
        default_matsus = np.zeros(n_matsu.value, dtype=np.int64)
        status_val = _lib.spir_basis_get_default_matsus(basis, c_bool(False), 
                                                       default_matsus.ctypes.data_as(POINTER(c_int64)))
        assert status_val == COMPUTATION_SUCCESS
        
        # Create Matsubara sampling
        matsu_sampling = _lib.spir_matsu_sampling_new(basis, c_bool(False), n_matsu.value,
                                                     default_matsus.ctypes.data_as(POINTER(c_int64)), 
                                                     byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert matsu_sampling is not None
        
        # Cleanup
        _lib.spir_sampling_release(matsu_sampling)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)
    
    def test_basis_functions_u(self):
        """Test u basis functions retrieval"""
        status = c_int()
        beta = 5.0
        wmax = 1.0
        
        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), byref(status))
        basis = _lib.spir_basis_new(c_int(STATISTICS_FERMIONIC), c_double(beta), 
                                   c_double(wmax), kernel, sve, byref(status))
        
        # Get u functions
        u_funcs = _lib.spir_basis_get_u(basis, byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert u_funcs is not None
        
        # Get function size
        funcs_size = c_int()
        status_val = _lib.spir_funcs_get_size(u_funcs, byref(funcs_size))
        assert status_val == COMPUTATION_SUCCESS
        assert funcs_size.value > 0
        
        # Cleanup
        _lib.spir_funcs_release(u_funcs)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)
    
    def test_memory_management(self):
        """Test that all release functions work without segfaults"""
        status = c_int()
        
        # Create full setup
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), byref(status))
        basis = _lib.spir_basis_new(c_int(STATISTICS_FERMIONIC), c_double(10.0), 
                                   c_double(1.0), kernel, sve, byref(status))
        
        u_funcs = _lib.spir_basis_get_u(basis, byref(status))
        
        n_tau = c_int()
        _lib.spir_basis_get_n_default_taus(basis, byref(n_tau))
        tau_points = np.zeros(n_tau.value, dtype=np.float64)
        _lib.spir_basis_get_default_taus(basis, tau_points.ctypes.data_as(POINTER(c_double)))
        tau_sampling = _lib.spir_tau_sampling_new(basis, n_tau.value, 
                                                 tau_points.ctypes.data_as(POINTER(c_double)), 
                                                 byref(status))
        
        # Release everything in correct order
        _lib.spir_sampling_release(tau_sampling)
        _lib.spir_funcs_release(u_funcs)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)
        
        # Test should complete without segfault
        assert True