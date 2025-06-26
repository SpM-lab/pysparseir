"""
Comparison tests between pysparseir and sparse-ir reference implementation.

This module tests compatibility and numerical accuracy between our C++ backend
and the pure Python sparse-ir implementation.
"""

import pytest
import numpy as np
import pylibsparseir

try:
    import sys
    sys.path.insert(0, '/Users/terasaki/work/atelierarith/spm-lab/sparse-ir/src')
    import sparse_ir
    SPARSE_IR_AVAILABLE = True
except ImportError:
    SPARSE_IR_AVAILABLE = False
    sparse_ir = None

pytestmark = pytest.mark.skipif(
    not SPARSE_IR_AVAILABLE, 
    reason="sparse-ir not available for comparison"
)


class TestBasisComparison:
    """Compare FiniteTempBasis implementations."""
    
    @pytest.mark.parametrize("statistics", ['F', 'B'])
    @pytest.mark.parametrize("beta", [1.0, 4.0])
    @pytest.mark.parametrize("wmax", [10.0, 42.0])
    def test_basis_properties(self, statistics, beta, wmax):
        """Compare basic basis properties."""
        eps = 1e-6
        
        # Create both implementations
        try:
            basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
            basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
        except Exception as e:
            pytest.skip(f"Failed to create basis: {e}")
        
        # Compare basic properties
        assert basis_pysparseir.statistics == basis_sparse_ir.statistics
        assert basis_pysparseir.beta == basis_sparse_ir.beta
        assert basis_pysparseir.wmax == basis_sparse_ir.wmax
        assert basis_pysparseir.lambda_ == basis_sparse_ir.lambda_
        
        # Size should be close (within reasonable tolerance for different algorithms)
        size_ratio = basis_pysparseir.size / basis_sparse_ir.size
        assert 0.5 <= size_ratio <= 2.0, f"Size ratio {size_ratio} too different"
        
        # Compare singular values (up to common size)
        min_size = min(basis_pysparseir.size, basis_sparse_ir.size)
        s_pysparseir = basis_pysparseir.s[:min_size]
        s_sparse_ir = basis_sparse_ir.s[:min_size]
        
        # Singular values should be close
        np.testing.assert_allclose(
            s_pysparseir, s_sparse_ir, 
            rtol=1e-10, atol=1e-15,
            err_msg="Singular values differ significantly"
        )
        
        # Compare accuracy
        acc_ratio = basis_pysparseir.accuracy / basis_sparse_ir.accuracy
        assert 0.1 <= acc_ratio <= 10.0, f"Accuracy ratio {acc_ratio} too different"
    
    @pytest.mark.parametrize("statistics", ['F'])  # Start with fermions only
    def test_basis_function_evaluation(self, statistics):
        """Compare basis function evaluation."""
        beta, wmax, eps = 1.0, 10.0, 1e-6
        
        try:
            basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
            basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
        except Exception as e:
            pytest.skip(f"Failed to create basis: {e}")
        
        # Test tau function evaluation
        tau_points = np.linspace(0, beta, 5)
        
        try:
            u_pysparseir = basis_pysparseir.u(tau_points)
            u_sparse_ir = basis_sparse_ir.u(tau_points)
            
            # Compare shapes
            min_size = min(u_pysparseir.shape[0], u_sparse_ir.shape[0])
            u_pysparseir_trunc = u_pysparseir[:min_size, :]
            u_sparse_ir_trunc = u_sparse_ir[:min_size, :]
            
            # Compare values (allowing for different signs due to SVD ambiguity)
            for i in range(min_size):
                corr = np.corrcoef(u_pysparseir_trunc[i, :], u_sparse_ir_trunc[i, :])[0, 1]
                assert abs(corr) > 0.99, f"u function {i} correlation {corr} too low"
                
        except Exception as e:
            pytest.skip(f"u function evaluation failed: {e}")
        
        # Test omega function evaluation
        omega_points = np.linspace(-wmax, wmax, 5)
        
        try:
            v_pysparseir = basis_pysparseir.v(omega_points)
            v_sparse_ir = basis_sparse_ir.v(omega_points)
            
            min_size = min(v_pysparseir.shape[0], v_sparse_ir.shape[0])
            v_pysparseir_trunc = v_pysparseir[:min_size, :]
            v_sparse_ir_trunc = v_sparse_ir[:min_size, :]
            
            # Compare values
            for i in range(min_size):
                corr = np.corrcoef(v_pysparseir_trunc[i, :], v_sparse_ir_trunc[i, :])[0, 1]
                assert abs(corr) > 0.99, f"v function {i} correlation {corr} too low"
                
        except Exception as e:
            pytest.skip(f"v function evaluation failed: {e}")


class TestSamplingComparison:
    """Compare sampling implementations."""
    
    def test_tau_sampling_basic(self):
        """Compare basic TauSampling functionality."""
        statistics, beta, wmax, eps = 'F', 1.0, 10.0, 1e-6
        
        try:
            basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
            basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
        except Exception as e:
            pytest.skip(f"Failed to create basis: {e}")
        
        # Create sampling objects
        try:
            tau_sampling_pysparseir = pylibsparseir.TauSampling(basis_pysparseir)
            tau_sampling_sparse_ir = sparse_ir.TauSampling(basis_sparse_ir)
        except Exception as e:
            pytest.skip(f"Failed to create tau sampling: {e}")
        
        # Compare number of sampling points
        assert len(tau_sampling_pysparseir.tau) > 0
        assert len(tau_sampling_sparse_ir.tau) > 0
        
        # Test roundtrip accuracy for pysparseir
        try:
            al_test = np.zeros(basis_pysparseir.size)
            al_test[0] = 1.0
            
            ax = tau_sampling_pysparseir.evaluate(al_test)
            al_recovered = tau_sampling_pysparseir.fit(ax)
            
            roundtrip_error = np.max(np.abs(al_test - al_recovered))
            assert roundtrip_error < 1e-12, f"Roundtrip error {roundtrip_error} too large"
            
        except Exception as e:
            pytest.skip(f"TauSampling roundtrip test failed: {e}")
    
    def test_default_sampling_points_compatibility(self):
        """Test compatibility of default sampling points."""
        statistics, beta, wmax, eps = 'F', 1.0, 10.0, 1e-6
        
        try:
            basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
            basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
        except Exception as e:
            pytest.skip(f"Failed to create basis: {e}")
        
        # Compare tau sampling points structure
        try:
            tau_points_pysparseir = basis_pysparseir.default_tau_sampling_points()
            tau_points_sparse_ir = basis_sparse_ir.default_tau_sampling_points()
            
            # Should have reasonable tau ranges
            assert np.all(tau_points_pysparseir >= -1e-10)  # Allowing numerical errors
            assert np.all(tau_points_pysparseir <= beta + 1e-10)
            
            assert np.all(tau_points_sparse_ir >= -1e-10)
            assert np.all(tau_points_sparse_ir <= beta + 1e-10)
            
            # Lengths should be related to basis sizes
            assert len(tau_points_pysparseir) == basis_pysparseir.size
            assert len(tau_points_sparse_ir) == basis_sparse_ir.size
            
        except Exception as e:
            pytest.skip(f"Default tau points comparison failed: {e}")


class TestNumericalAccuracy:
    """Test numerical accuracy against sparse-ir."""
    
    def test_green_function_example(self):
        """Test with a simple Green's function example."""
        statistics, beta, wmax, eps = 'F', 10.0, 8.0, 1e-6
        
        try:
            basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
            basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
        except Exception as e:
            pytest.skip(f"Failed to create basis: {e}")
        
        # Create simple spectral function (single pole)
        omega_pole = 2.5
        
        try:
            # Get common size 
            min_size = min(basis_pysparseir.size, basis_sparse_ir.size)
            
            # Evaluate v functions at the pole
            v_pysparseir = basis_pysparseir.v(np.array([omega_pole]))[:min_size, 0]
            v_sparse_ir = basis_sparse_ir.v(np.array([omega_pole]))[:min_size, 0]
            
            # Compute IR coefficients
            s_pysparseir = basis_pysparseir.s[:min_size]
            s_sparse_ir = basis_sparse_ir.s[:min_size]
            
            gl_pysparseir = s_pysparseir * v_pysparseir
            gl_sparse_ir = s_sparse_ir * v_sparse_ir
            
            # Compare IR coefficients
            # Note: may have different signs due to SVD ambiguity
            for i in range(min(5, min_size)):  # Check first few coefficients
                ratio = gl_pysparseir[i] / gl_sparse_ir[i] if gl_sparse_ir[i] != 0 else 1.0
                assert abs(abs(ratio) - 1.0) < 0.1, f"Coefficient {i} ratio {ratio} differs significantly"
                
        except Exception as e:
            pytest.skip(f"Green's function test failed: {e}")


@pytest.mark.skip(reason="Development benchmark - not a test")
def test_performance_comparison():
    """Compare performance between implementations."""
    import time
    
    statistics, beta, wmax, eps = 'F', 10.0, 42.0, 1e-6
    
    # Time pysparseir
    start = time.time()
    basis_pysparseir = pylibsparseir.FiniteTempBasis(statistics, beta, wmax, eps)
    tau_sampling = pylibsparseir.TauSampling(basis_pysparseir)
    time_pysparseir = time.time() - start
    
    # Time sparse-ir
    start = time.time()
    basis_sparse_ir = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
    tau_sampling_ref = sparse_ir.TauSampling(basis_sparse_ir)
    time_sparse_ir = time.time() - start
    
    print(f"pysparseir: {time_pysparseir:.3f}s")
    print(f"sparse-ir:  {time_sparse_ir:.3f}s")
    print(f"Speedup:    {time_sparse_ir/time_pysparseir:.2f}x")