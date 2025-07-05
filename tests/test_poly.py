import importlib
import numpy as np
import pylibsparseir
importlib.reload(pylibsparseir)

def test_poly():
    eps = 1e-6
    beta = 4.2
    wmax = 10
    basis = pylibsparseir.FiniteTempBasis('F', beta, wmax, eps)

    u1 = basis.u[1]
    assert np.allclose(u1(np.array([0.5, 0.3, 1.0, 2.0])), np.array([-0.43049722, -0.67225263, -0.18450157, -0.01225698]))
    u1(1.0) == -0.18450156753665

