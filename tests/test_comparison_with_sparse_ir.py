"""
Tests for pysparseir API compatibility and behavior.

This module tests the pysparseir implementation to ensure it follows
the expected sparse-ir API patterns and behaviors.
"""

import pytest
import numpy as np
import pylibsparseir


class TestBasisComparison:
    """Test FiniteTempBasis API and expected behaviors."""
    
    @pytest.mark.parametrize("statistics", ['F', 'B'])
    @pytest.mark.parametrize("beta", [1.0, 4.0])
    @pytest.mark.parametrize("wmax", [10.0, 42.0])
    def test_basis_properties(self, statistics, beta, wmax):
        """Test basic basis properties."""
        eps = 1e-6
        
        # Create basis
        basis = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
        
        # Check basic properties
        assert basis.statistics == statistics
        assert basis.beta == beta
        assert basis.wmax == wmax
        assert basis.lambda_ == beta * wmax
        
        # Check that we have a reasonable number of basis functions
        assert 5 <= basis.size <= 100  # Typical range
        
        # Check singular values are in descending order
        assert np.all(np.diff(basis.s) <= 0)
        
        # Check first singular value is reasonable (typically between 0.9 and 1.5)
        assert 0.9 < basis.s[0] < 2.0
    
    @pytest.mark.parametrize("statistics", ['F'])
    def test_basis_function_evaluation(self, statistics):
        """Test basis function evaluation."""
        beta = 10.0
        wmax = 8.0
        eps = 1e-6
        
        basis = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
        
        # Test u function evaluation
        tau_points = np.linspace(0, beta, 10)
        u_vals = basis.u(tau_points)
        
        # Check shape
        assert u_vals.shape == (basis.size, len(tau_points))
        
        # Check values are finite
        assert np.all(np.isfinite(u_vals))
        
        # Test v function evaluation
        omega_points = np.linspace(-wmax, wmax, 10)
        v_vals = basis.v(omega_points)
        
        # Check shape
        assert v_vals.shape == (basis.size, len(omega_points))
        
        # Check values are finite
        assert np.all(np.isfinite(v_vals))


class TestSamplingComparison:
    """Test sampling classes behavior."""
    
    def test_tau_sampling_basic(self):
        """Test basic TauSampling functionality."""
        basis = pylibsparseir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)
        
        # Create sampling
        sampling = pylibsparseir.TauSampling(basis)
        
        # Check that we have sampling points
        assert len(sampling.tau) > 0
        assert len(sampling.tau) >= basis.size
        
        # Check tau points are within reasonable range
        # Note: tau sampling points may extend outside [0, beta] for better conditioning
        assert np.all(np.abs(sampling.tau) <= basis.beta)
        
        # Test evaluate/fit roundtrip
        al = np.random.randn(basis.size)
        ax = sampling.evaluate(al)
        al_recovered = sampling.fit(ax)
        
        # Should recover coefficients accurately
        assert np.allclose(al, al_recovered, rtol=1e-10, atol=1e-12)
    
    def test_default_sampling_points_compatibility(self):
        """Test that default sampling points have expected properties."""
        basis = pylibsparseir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)
        
        # Test default tau points
        tau_points = basis.default_tau_sampling_points()
        assert len(tau_points) >= basis.size
        # tau points may extend outside [0, beta]
        assert np.all(np.abs(tau_points) <= basis.beta)
        
        # Test default matsubara points
        wn_points = basis.default_matsubara_sampling_points()
        assert len(wn_points) >= basis.size
        # For fermions, should be odd integers
        assert np.all(wn_points % 2 == 1)


class TestNumericalAccuracy:
    """Test numerical accuracy of operations."""
    
    def test_green_function_example(self):
        """Test with a simple Green's function example."""
        # Create basis
        beta = 10.0
        wmax = 8.0
        eps = 1e-6
        
        basis = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)
        tau_sampling = pylibsparseir.TauSampling(basis)
        
        # Create a simple Green's function: G(tau) = -0.5 for non-interacting fermion at half-filling
        # This corresponds to rho(omega) = delta(omega)
        expected_gtau = -0.5 * np.ones(len(tau_sampling.tau))
        
        # Fit to get coefficients
        gl = tau_sampling.fit(expected_gtau)
        
        # Reconstruct
        gtau_reconstructed = tau_sampling.evaluate(gl)
        
        # Check accuracy - for a constant function, reconstruction should be very good
        assert np.allclose(expected_gtau, gtau_reconstructed, rtol=1e-3, atol=1e-3)


def test_performance_comparison():
    """Basic performance check."""
    import time
    
    # Create a reasonably large basis
    beta = 100.0
    wmax = 10.0
    eps = 1e-10
    
    start = time.time()
    basis = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)
    creation_time = time.time() - start
    
    # Basis creation should be reasonably fast (under 2 seconds)
    assert creation_time < 2.0
    
    # Test sampling performance
    sampling = pylibsparseir.TauSampling(basis)
    al = np.random.randn(basis.size)
    
    start = time.time()
    for _ in range(100):
        ax = sampling.evaluate(al)
        al_recovered = sampling.fit(ax)
    elapsed = time.time() - start
    
    # 100 evaluate/fit cycles should be fast
    assert elapsed < 1.0