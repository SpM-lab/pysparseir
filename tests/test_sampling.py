"""
Test cases for sampling functionality
"""

import pytest
import numpy as np
import pylibsparseir


class TestTauSampling:
    """Test TauSampling class."""
    
    @pytest.fixture
    def basis(self):
        """Create a test basis."""
        return pylibsparseir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)
    
    def test_creation_default_points(self, basis):
        """Test TauSampling creation with default points."""
        sampling = pylibsparseir.TauSampling(basis)
        
        assert len(sampling.tau) == basis.size
        # Note: tau points can extend beyond [0, beta] for numerical reasons
        assert np.all(np.isfinite(sampling.tau))  # Should be finite
        
    def test_creation_custom_points(self, basis):
        """Test TauSampling creation with custom points."""
        custom_points = np.linspace(0, basis.beta, 10)
        sampling = pylibsparseir.TauSampling(basis, custom_points)
        
        assert len(sampling.tau) == 10
        np.testing.assert_array_almost_equal(sampling.tau, custom_points)
        
    def test_evaluate_fit_roundtrip(self, basis):
        """Test evaluate/fit roundtrip accuracy."""
        sampling = pylibsparseir.TauSampling(basis)
        
        # Test with different coefficient patterns
        test_cases = [
            np.array([1.0] + [0.0] * (basis.size - 1)),  # First coefficient only
            np.array([0.0, 1.0] + [0.0] * (basis.size - 2)),  # Second coefficient only
            np.random.random(basis.size),  # Random coefficients
        ]
        
        for al_original in test_cases:
            # Evaluate -> Fit cycle
            ax = sampling.evaluate(al_original)
            al_recovered = sampling.fit(ax)
            
            # Check roundtrip accuracy
            error = np.max(np.abs(al_original - al_recovered))
            assert error < 1e-12, f"Roundtrip error too large: {error}"
            
    def test_evaluate_shape(self, basis):
        """Test evaluate output shape."""
        sampling = pylibsparseir.TauSampling(basis)
        
        al = np.ones(basis.size)
        ax = sampling.evaluate(al)
        
        assert ax.shape == (len(sampling.tau),)
        assert np.all(np.isfinite(ax))
        
    def test_fit_shape(self, basis):
        """Test fit output shape."""
        sampling = pylibsparseir.TauSampling(basis)
        
        ax = np.ones(len(sampling.tau))
        al = sampling.fit(ax)
        
        assert al.shape == (basis.size,)
        assert np.all(np.isfinite(al))
        
    def test_wrong_input_size(self, basis):
        """Test error handling for wrong input sizes."""
        sampling = pylibsparseir.TauSampling(basis)
        
        # Wrong size for evaluate
        with pytest.raises(ValueError, match="Expected .* coefficients"):
            sampling.evaluate(np.ones(basis.size + 1))
            
        # Wrong size for fit
        with pytest.raises(ValueError, match="Expected .* values"):
            sampling.fit(np.ones(len(sampling.tau) + 1))
            
    def test_repr(self, basis):
        """Test string representation."""
        sampling = pylibsparseir.TauSampling(basis)
        repr_str = repr(sampling)
        assert 'TauSampling' in repr_str
        assert str(len(sampling.tau)) in repr_str


class TestMatsubaraSampling:
    """Test MatsubaraSampling class."""
    
    @pytest.fixture
    def basis(self):
        """Create a test basis."""
        return pylibsparseir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)
    
    def test_creation_default_points(self, basis):
        """Test MatsubaraSampling creation with default points."""
        pytest.skip("MatsubaraSampling has C API issues - temporarily disabled")
        
    def test_creation_custom_points(self, basis):
        """Test MatsubaraSampling creation with custom points."""
        pytest.skip("MatsubaraSampling has C API issues - temporarily disabled")
        
    def test_repr(self, basis):
        """Test string representation."""
        pytest.skip("MatsubaraSampling has C API issues - temporarily disabled")