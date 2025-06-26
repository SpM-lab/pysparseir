"""Test API compatibility with sparse-ir"""
import pytest
import numpy as np
import pylibsparseir


class TestAPICompatibility:
    """Test that pysparseir provides the same API as sparse-ir"""
    
    def test_shape_property(self):
        """Test that basis has shape property"""
        basis = pylibsparseir.FiniteTempBasis('F', beta=10.0, wmax=4.0)
        assert hasattr(basis, 'shape')
        assert isinstance(basis.shape, tuple)
        assert len(basis.shape) == 1
        assert basis.shape[0] == basis.size
    
    def test_kernel_property(self):
        """Test that basis has kernel property"""
        basis = pylibsparseir.FiniteTempBasis('F', beta=10.0, wmax=4.0)
        assert hasattr(basis, 'kernel')
        assert basis.kernel is not None
        # Should be the same kernel object used internally
        assert basis.kernel == basis._kernel
    
    def test_sve_result_property(self):
        """Test that basis has sve_result property"""
        basis = pylibsparseir.FiniteTempBasis('F', beta=10.0, wmax=4.0)
        assert hasattr(basis, 'sve_result')
        assert basis.sve_result is not None
        # Should be the same SVE result object used internally
        assert basis.sve_result == basis._sve
    
    def test_rescale_method(self):
        """Test rescale method functionality"""
        beta, wmax = 10.0, 4.0
        basis = pylibsparseir.FiniteTempBasis('F', beta=beta, wmax=wmax)
        
        # Test rescaling to larger lambda
        new_lambda = 100.0
        rescaled = basis.rescale(new_lambda)
        
        # Check type
        assert isinstance(rescaled, pylibsparseir.FiniteTempBasis)
        
        # Check lambda is correct
        assert abs(rescaled.lambda_ - new_lambda) < 1e-10
        
        # Check statistics preserved
        assert rescaled.statistics == basis.statistics
        
        # Check beta * wmax = lambda
        assert abs(rescaled.beta * rescaled.wmax - new_lambda) < 1e-10
        
        # Check that size changes appropriately (larger lambda -> more basis functions)
        assert rescaled.size >= basis.size
    
    def test_rescale_preserves_ratio(self):
        """Test that rescale preserves beta/wmax ratio"""
        beta, wmax = 10.0, 4.0
        basis = pylibsparseir.FiniteTempBasis('F', beta=beta, wmax=wmax)
        original_ratio = beta / wmax
        
        # Test multiple rescalings
        for new_lambda in [20.0, 50.0, 200.0]:
            rescaled = basis.rescale(new_lambda)
            new_ratio = rescaled.beta / rescaled.wmax
            assert abs(new_ratio - original_ratio) < 1e-10
    
    def test_all_essential_properties_exist(self):
        """Test that all essential properties from sparse-ir exist"""
        basis = pylibsparseir.FiniteTempBasis('F', beta=10.0, wmax=4.0)
        
        # Core properties
        essential_properties = [
            'beta', 'wmax', 'lambda_', 'size', 's', 'shape',
            'statistics', 'accuracy', 'significance', 
            'u', 'uhat', 'v', 'kernel', 'sve_result'
        ]
        
        for prop in essential_properties:
            assert hasattr(basis, prop), f"Missing property: {prop}"
    
    def test_all_essential_methods_exist(self):
        """Test that all essential methods from sparse-ir exist"""
        basis = pylibsparseir.FiniteTempBasis('F', beta=10.0, wmax=4.0)
        
        # Core methods
        essential_methods = [
            'default_tau_sampling_points',
            'default_matsubara_sampling_points',
            'default_omega_sampling_points',
            'rescale'
        ]
        
        for method in essential_methods:
            assert hasattr(basis, method), f"Missing method: {method}"
            assert callable(getattr(basis, method)), f"{method} is not callable"