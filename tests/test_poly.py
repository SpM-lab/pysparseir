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

def test_poly_v():
    beta = 2
    wmax = 21
    eps = 1e-7
    basis_b = pylibsparseir.FiniteTempBasis("B", beta, wmax, eps=eps)

    omega_p = np.array([2.2, -1.0])
    o = basis_b.v(omega_p)

    expected = np.array([
       [ 2.48319534e-01,  3.67293961e-01],
       [-2.63049782e-01,  2.32656041e-01],
       [-2.21475944e-02, -2.68551628e-01],
       [ 2.68522049e-01, -3.23947764e-01],
       [-2.17706951e-01,  7.58420278e-02],
       [-1.01617786e-01,  3.05215035e-01],
       [ 2.69562001e-01,  3.02263177e-02],
       [-5.21187755e-02, -2.62891521e-01],
       [-2.25825568e-01, -8.83431760e-02],
       [ 1.49132062e-01,  2.21787514e-01],
       [ 1.51033760e-01,  1.23756393e-01],
       [-1.98378155e-01, -1.85336198e-01],
       [-7.24090088e-02, -1.47042209e-01],
       [ 2.13928286e-01,  1.52997356e-01],
       [ 3.23670710e-04,  1.63050957e-01],
       [-2.06263326e-01, -1.23801679e-01],
       [ 6.13260017e-02, -1.74220727e-01],
       [ 1.82556391e-01,  9.69383304e-02]
       ])

    np.testing.assert_allclose(o, expected, atol=300*eps, rtol=0)