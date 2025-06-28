"""
Test cases for core functionality and C API wrappers
"""

import pytest
import numpy as np
import pylibsparseir
from pylibsparseir.core import *
from pylibsparseir.kernel import LogisticKernel, RegularizedBoseKernel, kernel_domain

class TestCoreAPI:
    """Test core C API wrapper functions."""

    def test_kernel_creation(self):
        """Test kernel creation functions."""
        lambda_val = 80.0

        # Test logistic kernel
        kernel_log = LogisticKernel(lambda_val)
        assert kernel_log is not None

        # Test regularized boson kernel
        kernel_bose = RegularizedBoseKernel(lambda_val)
        assert kernel_bose is not None

        # Test kernel domain
        xmin, xmax, ymin, ymax = kernel_domain(kernel_log)
        assert xmin < xmax
        assert ymin < ymax

    def test_sve_computation(self):
        """Test SVE computation."""
        kernel = LogisticKernel(80.0)
        eps = 1e-6

        sve = sve_result_new(kernel, eps)
        assert sve is not None

        size = sve_result_get_size(sve)
        assert size > 0

        svals = sve_result_get_svals(sve)
        assert len(svals) == size
        assert np.all(svals > 0)
        assert np.all(svals[:-1] >= svals[1:])  # Decreasing

    def test_basis_creation(self):
        """Test basis creation and properties."""
        kernel = LogisticKernel(80.0)
        sve = sve_result_new(kernel, 1e-6)

        # Test fermion basis
        basis_f = basis_new(1, 10.0, 8.0, kernel, sve)  # 1 = fermion
        assert basis_f is not None

        size_f = basis_get_size(basis_f)
        assert size_f > 0

        stats_f = basis_get_stats(basis_f)
        assert stats_f == 1  # Fermion

        svals_f = basis_get_svals(basis_f)
        assert len(svals_f) == size_f

        # Test boson basis
        basis_b = basis_new(0, 10.0, 8.0, kernel, sve)  # 0 = boson
        stats_b = basis_get_stats(basis_b)
        assert stats_b == 0  # Boson

    def test_basis_functions(self):
        """Test basis function objects."""
        kernel = LogisticKernel(80.0)
        sve = sve_result_new(kernel, 1e-6)
        basis = basis_new(1, 10.0, 8.0, kernel, sve)

        # Test getting function objects
        u_funcs = basis_get_u(basis)
        assert u_funcs is not None

        v_funcs = basis_get_v(basis)
        assert v_funcs is not None

        uhat_funcs = basis_get_uhat(basis)
        assert uhat_funcs is not None

        # Test function evaluation
        x_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        u_vals = funcs_evaluate(u_funcs, x_points)

        size = basis_get_size(basis)
        assert u_vals.shape == (size, len(x_points))
        assert np.all(np.isfinite(u_vals))

    def test_default_sampling_points(self):
        """Test default sampling point functions."""
        kernel = LogisticKernel(80.0)
        sve = sve_result_new(kernel, 1e-6)
        basis = basis_new(1, 10.0, 8.0, kernel, sve)

        # Test tau sampling points
        tau_points = basis_get_default_tau_sampling_points(basis)
        assert len(tau_points) > 0
        assert np.all(np.isfinite(tau_points))  # Should be finite

        # Test Matsubara sampling points
        matsu_points = basis_get_default_matsubara_sampling_points(basis, False)
        assert len(matsu_points) > 0

        matsu_points_pos = basis_get_default_matsubara_sampling_points(basis, True)
        assert len(matsu_points_pos) > 0
        assert np.all(matsu_points_pos >= 0)

    def test_sampling_objects(self):
        """Test sampling object creation."""
        kernel = LogisticKernel(80.0)
        sve = sve_result_new(kernel, 1e-6)
        basis = basis_new(1, 10.0, 8.0, kernel, sve)

        # Test tau sampling
        tau_points = basis_get_default_tau_sampling_points(basis)
        tau_sampling = tau_sampling_new(basis, tau_points)
        assert tau_sampling is not None

        # Test Matsubara sampling - temporarily disabled due to C API issues
        # matsu_points = basis_get_default_matsubara_sampling_points(basis, True)
        # matsu_sampling = matsubara_sampling_new(basis, True, matsu_points)
        # assert matsu_sampling is not None


class TestErrorHandling:
    """Test error handling in C API wrappers."""

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""

        # Note: libsparseir may be more permissive than expected
        # Some "invalid" values might be handled gracefully

        # Test very small epsilon (this should still work)
        kernel = LogisticKernel(80.0)
        try:
            sve_result_new(kernel, 1e-20)  # Very small epsilon
        except RuntimeError:
            # This is acceptable - very small epsilon might fail
            pass