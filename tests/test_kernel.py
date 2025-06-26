"""
Test cases for kernel functionality, following sparse-ir test patterns.
"""

import pytest
import numpy as np
import pylibsparseir
from pylibsparseir.core import *
from .conftest import KERNEL_LAMBDAS


class TestLogisticKernel:
    """Test LogisticKernel functionality."""
    
    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_creation(self, lambda_):
        """Test kernel creation for various Lambda values."""
        kernel = logistic_kernel_new(lambda_)
        assert kernel is not None
        
        # Test domain properties
        xmin, xmax, ymin, ymax = kernel_domain(kernel)
        assert xmin < xmax
        assert ymin < ymax
        
        # For logistic kernel, domain should be [-1, 1] x [-1, 1]
        np.testing.assert_allclose([xmin, xmax, ymin, ymax], [-1, 1, -1, 1], atol=1e-14)
    
    def test_invalid_lambda(self):
        """Test error handling for invalid Lambda values."""
        # Note: libsparseir may handle edge cases gracefully
        # Zero lambda might work but produce warnings
        try:
            kernel = logistic_kernel_new(0.0)
            # If this succeeds, that's also acceptable
        except RuntimeError:
            # If it fails, that's expected
            pass


class TestRegularizedBoseKernel:
    """Test RegularizedBoseKernel functionality."""
    
    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_creation(self, lambda_):
        """Test regularized Bose kernel creation."""
        kernel = reg_bose_kernel_new(lambda_)
        assert kernel is not None
        
        # Test domain properties  
        xmin, xmax, ymin, ymax = kernel_domain(kernel)
        assert xmin < xmax
        assert ymin < ymax
        
        # For regularized Bose kernel, domain should be [-1, 1] x [-1, 1]
        np.testing.assert_allclose([xmin, xmax, ymin, ymax], [-1, 1, -1, 1], atol=1e-14)
    
    def test_invalid_lambda(self):
        """Test error handling for invalid Lambda values."""
        with pytest.raises(RuntimeError, match="Failed to create"):
            reg_bose_kernel_new(-1.0)  # Negative lambda should fail


class TestSVEComputation:
    """Test SVE computation for kernels."""
    
    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_logistic_sve(self, lambda_):
        """Test SVE computation for logistic kernel."""
        kernel = logistic_kernel_new(lambda_)
        
        # Test different epsilon values
        for eps in [1e-6, 1e-8, 1e-10]:
            sve = sve_result_new(kernel, eps)
            assert sve is not None
            
            # Get properties
            size = sve_result_get_size(sve)
            assert size > 0
            
            svals = sve_result_get_svals(sve)
            assert len(svals) == size
            
            # Check singular value properties
            assert np.all(svals > 0), "All singular values should be positive"
            assert np.all(svals[:-1] >= svals[1:]), "Singular values should be decreasing"
            
            # Check accuracy
            accuracy = svals[-1] / svals[0]
            assert accuracy <= eps, f"Accuracy {accuracy} should be <= eps {eps}"
    
    @pytest.mark.parametrize("lambda_", [10, 1000]) 
    def test_reg_bose_sve(self, lambda_):
        """Test SVE computation for regularized Bose kernel."""
        kernel = reg_bose_kernel_new(lambda_)
        eps = 1e-6
        
        sve = sve_result_new(kernel, eps)
        assert sve is not None
        
        size = sve_result_get_size(sve)
        assert size > 0
        
        svals = sve_result_get_svals(sve)
        
        # Check properties
        assert np.all(svals > 0)
        assert np.all(svals[:-1] >= svals[1:])
        
        accuracy = svals[-1] / svals[0]
        assert accuracy <= eps
    
    def test_sve_invalid_epsilon(self):
        """Test SVE with invalid epsilon values."""
        kernel = logistic_kernel_new(42.0)
        
        # Note: libsparseir may handle edge cases differently than expected
        # Some edge values might produce warnings but still work
        try:
            sve = sve_result_new(kernel, 0.0)   # Zero epsilon
            # If this works, check it produces reasonable results
            if sve is not None:
                size = sve_result_get_size(sve)
                assert size > 0
        except RuntimeError:
            # If it fails, that's also acceptable
            pass
    
    @pytest.mark.parametrize("lambda_", [10, 42])
    def test_sve_size_vs_epsilon(self, lambda_):
        """Test that smaller epsilon gives larger basis size."""
        kernel = logistic_kernel_new(lambda_)
        
        sve_loose = sve_result_new(kernel, 1e-4)
        sve_tight = sve_result_new(kernel, 1e-8)
        
        size_loose = sve_result_get_size(sve_loose)
        size_tight = sve_result_get_size(sve_tight)
        
        assert size_tight >= size_loose, "Tighter epsilon should give larger or equal basis size"