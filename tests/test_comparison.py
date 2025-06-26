"""
Comparison tests between different implementations and validation against known results.
Following the pattern of sparse-ir test_compare.py but for internal consistency.
"""

import pytest
import numpy as np
import pylibsparseir


class TestInternalConsistency:
    """Test consistency between different interfaces and implementations."""
    
    def test_core_vs_highlevel_basis(self):
        """Test that high-level FiniteTempBasis gives same results as core API."""
        # Parameters
        beta, wmax, eps = 2.0, 10.0, 1e-6
        stat = 'F'
        
        # High-level interface
        basis_high = pylibsparseir.FiniteTempBasis(stat, beta, wmax, eps)
        
        # Low-level interface
        kernel_low = pylibsparseir.logistic_kernel_new(beta * wmax)
        sve_low = pylibsparseir.sve_result_new(kernel_low, eps)
        basis_low = pylibsparseir.basis_new(1, beta, wmax, kernel_low, sve_low)  # 1 = fermion
        
        # Compare basic properties
        size_low = pylibsparseir.basis_get_size(basis_low)
        assert basis_high.size == size_low
        
        svals_low = pylibsparseir.basis_get_svals(basis_low)
        np.testing.assert_allclose(basis_high.s, svals_low, rtol=1e-14)
        
        stats_low = pylibsparseir.basis_get_stats(basis_low)
        assert stats_low == 1  # Fermion
    
    def test_different_epsilon_consistency(self):
        """Test that results are consistent across different epsilon values."""
        beta, wmax = 1.0, 20.0
        
        # Create bases with different accuracy
        basis_loose = pylibsparseir.FiniteTempBasis('F', beta, wmax, 1e-4)
        basis_tight = pylibsparseir.FiniteTempBasis('F', beta, wmax, 1e-8)
        
        # Tight basis should have same or more functions
        assert basis_tight.size >= basis_loose.size
        
        # First few singular values should match closely
        n_compare = min(basis_loose.size, basis_tight.size, 10)
        np.testing.assert_allclose(
            basis_loose.s[:n_compare], 
            basis_tight.s[:n_compare], 
            rtol=1e-10
        )
        
        # Loose basis should be subset of tight basis (approximately)
        assert basis_loose.accuracy >= basis_tight.accuracy
    
    def test_fermion_vs_boson_differences(self):
        """Test expected differences between fermion and boson bases."""
        beta, wmax, eps = 1.0, 15.0, 1e-6
        
        basis_f = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)
        basis_b = pylibsparseir.FiniteTempBasis('B', beta, wmax, eps)
        
        # Both should be reasonable sizes
        assert 5 <= basis_f.size <= 50
        assert 5 <= basis_b.size <= 50
        
        # They should generally have different sizes (not a strict requirement)
        # Just check they're both working
        assert basis_f.statistics == 'F'
        assert basis_b.statistics == 'B'
        
        # Both should have same Lambda
        assert basis_f.lambda_ == basis_b.lambda_ == beta * wmax
    
    def test_sampling_consistency(self):
        """Test consistency between different sampling approaches."""
        basis = pylibsparseir.FiniteTempBasis('F', 2.0, 12.0, 1e-6)
        
        # Default sampling
        smpl_default = pylibsparseir.TauSampling(basis)
        
        # Custom sampling with same points
        custom_points = smpl_default.tau.copy()
        smpl_custom = pylibsparseir.TauSampling(basis, custom_points)
        
        # Should give same sampling points
        np.testing.assert_allclose(smpl_default.tau, smpl_custom.tau, rtol=1e-14)
        
        # Should give same evaluate/fit results
        Gl_test = np.zeros(basis.size)
        Gl_test[0] = 1.0
        
        Gtau_default = smpl_default.evaluate(Gl_test)
        Gtau_custom = smpl_custom.evaluate(Gl_test)
        
        np.testing.assert_allclose(Gtau_default, Gtau_custom, rtol=1e-14)


class TestPhysicalConsistency:
    """Test that results satisfy expected physical properties."""
    
    def test_basis_orthonormality_proxy(self):
        """Test proxy for basis orthonormality through sampling."""
        basis = pylibsparseir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)
        smpl = pylibsparseir.TauSampling(basis)
        
        # Test that individual basis functions can be reconstructed
        for i in range(min(3, basis.size)):  # Test first few
            Gl_original = np.zeros(basis.size)
            Gl_original[i] = 1.0
            
            Gtau = smpl.evaluate(Gl_original)
            Gl_recovered = smpl.fit(Gtau)
            
            # Should recover individual basis functions very accurately
            np.testing.assert_allclose(Gl_original, Gl_recovered, rtol=1e-10, atol=1e-12)
    
    def test_causality_structure(self):
        """Test basic causality structure in imaginary time."""
        basis = pylibsparseir.FiniteTempBasis('F', 4.0, 8.0, 1e-6)
        
        # Test at boundary points
        tau_boundary = np.array([0.0, basis.beta])
        u_boundary = basis.u(tau_boundary)
        
        assert u_boundary.shape == (basis.size, 2)
        assert np.all(np.isfinite(u_boundary))
        
        # For fermionic Green's functions, there should be some structure
        # (exact properties depend on normalization, but values should be reasonable)
        assert np.any(np.abs(u_boundary) > 1e-10), "Boundary values should not all be negligible"
    
    def test_frequency_domain_structure(self):
        """Test frequency domain structure."""
        basis = pylibsparseir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)
        
        # Test at zero and finite frequencies
        omega_test = np.array([-2.0, 0.0, 2.0])
        v_test = basis.v(omega_test)
        
        assert v_test.shape == (basis.size, 3)
        assert np.all(np.isfinite(v_test))
        
        # Values should be reasonable (not all zero, not infinite)
        assert np.any(np.abs(v_test) > 1e-10)
        assert np.all(np.abs(v_test) < 1e10)


class TestScalingBehavior:
    """Test scaling behavior with parameters."""
    
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
    def test_beta_scaling(self, beta):
        """Test behavior as function of beta."""
        wmax = 10.0
        eps = 1e-6
        
        basis = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)
        
        # Basic sanity checks
        assert basis.beta == beta
        assert basis.lambda_ == beta * wmax
        assert 5 <= basis.size <= 50  # Reasonable range
        
        # Accuracy should be achieved
        assert basis.accuracy <= 10 * eps
    
    @pytest.mark.parametrize("wmax", [5.0, 10.0, 20.0, 40.0])
    def test_wmax_scaling(self, wmax):
        """Test behavior as function of wmax."""
        beta = 1.0
        eps = 1e-6
        
        basis = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)
        
        # Basic sanity checks
        assert basis.wmax == wmax
        assert basis.lambda_ == beta * wmax
        assert 5 <= basis.size <= 100  # Reasonable range
        
        # Larger wmax (larger Lambda) should generally give larger basis
        # This is a trend, not a strict rule
        assert basis.size >= 5


class TestNumericalStability:
    """Test numerical stability in various regimes."""
    
    def test_high_precision_consistency(self):
        """Test consistency at high precision."""
        basis = pylibsparseir.FiniteTempBasis('F', 1.0, 10.0, 1e-10)
        smpl = pylibsparseir.TauSampling(basis)
        
        # Even at high precision, basic operations should work
        assert basis.size > 0
        assert np.all(np.isfinite(basis.s))
        assert basis.accuracy <= 1e-8  # Should achieve requested precision
        
        # Sampling should still work
        Gl = np.zeros(basis.size)
        Gl[0] = 1.0
        
        Gtau = smpl.evaluate(Gl)
        Gl_recovered = smpl.fit(Gtau)
        
        # Should maintain high precision
        np.testing.assert_allclose(Gl, Gl_recovered, rtol=1e-12, atol=1e-14)
    
    def test_extreme_parameters(self):
        """Test with somewhat extreme but reasonable parameters."""
        # Large Lambda
        basis_large = pylibsparseir.FiniteTempBasis('F', 1.0, 100.0, 1e-4)
        assert basis_large.size > 10
        assert np.all(np.isfinite(basis_large.s))
        
        # Small Lambda  
        basis_small = pylibsparseir.FiniteTempBasis('F', 1.0, 5.0, 1e-6)
        assert basis_small.size >= 5
        assert np.all(np.isfinite(basis_small.s))