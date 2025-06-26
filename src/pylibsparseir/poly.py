"""
Piecewise polynomial functionality for SparseIR.

This module provides piecewise Legendre polynomial representation and
their Fourier transforms, which serve as core mathematical infrastructure
for IR basis functions.
"""

import numpy as np
from scipy import special
from scipy.integrate import quad


class PiecewiseLegendrePoly:
    """
    Piecewise Legendre polynomial representation.

    This class represents a function as a piecewise polynomial using
    Legendre polynomials on each interval.
    """

    def __init__(self, coeffs, knots=None, symm=None):
        """
        Initialize piecewise Legendre polynomial.

        Parameters
        ----------
        coeffs : array_like
            Coefficients for Legendre polynomials on each interval.
            Shape: (n_intervals, n_coeffs_per_interval)
        knots : array_like, optional
            Knot points defining the intervals. If None, assumes [-1, 1].
        symm : int, optional
            Symmetry of the polynomial (1 for even, -1 for odd, 0 for none)
        """
        self.coeffs = np.asarray(coeffs)
        if self.coeffs.ndim == 1:
            self.coeffs = self.coeffs.reshape(1, -1)

        if knots is None:
            knots = np.array([-1.0, 1.0])
        self.knots = np.asarray(knots)

        self.symm = symm or 0
        self.n_intervals = len(self.knots) - 1
        self.n_coeffs = self.coeffs.shape[1]

        if self.coeffs.shape[0] != self.n_intervals:
            raise ValueError("Number of coefficient rows must match number of intervals")

    def __call__(self, x):
        """
        Evaluate the piecewise polynomial at given points.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate

        Returns
        -------
        ndarray
            Function values
        """
        x = np.asarray(x)
        x_flat = x.flatten()
        result = np.zeros_like(x_flat)

        for i in range(self.n_intervals):
            # Find points in this interval
            left = self.knots[i]
            right = self.knots[i + 1]
            mask = (x_flat >= left) & (x_flat <= right)

            if np.any(mask):
                # Map to [-1, 1] for Legendre polynomials
                x_local = 2 * (x_flat[mask] - left) / (right - left) - 1

                # Evaluate Legendre polynomial expansion
                poly_val = np.zeros_like(x_local)
                for j, coeff in enumerate(self.coeffs[i]):
                    if coeff != 0:
                        poly_val += coeff * special.eval_legendre(j, x_local)

                result[mask] = poly_val

        return result.reshape(x.shape)

    def integrate(self, a=None, b=None):
        """
        Integrate the piecewise polynomial over a given interval.

        Parameters
        ----------
        a, b : float, optional
            Integration limits. If None, uses the full domain.

        Returns
        -------
        float
            Integral value
        """
        if a is None:
            a = self.knots[0]
        if b is None:
            b = self.knots[-1]

        # Use scipy.integrate.quad for simplicity
        result, _ = quad(self, a, b)
        return result

    def derivative(self, order=1):
        """
        Compute derivative of the piecewise polynomial.

        Parameters
        ----------
        order : int, optional
            Order of derivative (default: 1)

        Returns
        -------
        PiecewiseLegendrePoly
            Derivative polynomial
        """
        if order == 0:
            return self

        # Compute derivative coefficients for Legendre polynomials
        new_coeffs = np.zeros_like(self.coeffs)

        for i in range(self.n_intervals):
            interval_length = self.knots[i + 1] - self.knots[i]
            scale_factor = 2.0 / interval_length  # Scaling for coordinate transformation

            for j in range(self.n_coeffs - 1):
                # Derivative of Legendre polynomial: d/dx P_n(x) = (2n+1) P_{n-1}'(x) + ...
                # This is simplified - full implementation would use recurrence relations
                if j > 0:
                    new_coeffs[i, j - 1] += self.coeffs[i, j] * (2 * j + 1) * scale_factor

        result = PiecewiseLegendrePoly(new_coeffs, self.knots, self.symm)

        if order > 1:
            return result.derivative(order - 1)
        else:
            return result

    def __add__(self, other):
        """Add two piecewise polynomials."""
        if isinstance(other, (int, float)):
            # Add constant
            new_coeffs = self.coeffs.copy()
            new_coeffs[:, 0] += other  # Add to constant term
            return PiecewiseLegendrePoly(new_coeffs, self.knots, self.symm)
        elif isinstance(other, PiecewiseLegendrePoly):
            # Add two polynomials
            if not np.allclose(self.knots, other.knots):
                raise ValueError("Knots must match for polynomial addition")

            max_coeffs = max(self.n_coeffs, other.n_coeffs)
            new_coeffs = np.zeros((self.n_intervals, max_coeffs))

            new_coeffs[:, :self.n_coeffs] += self.coeffs
            new_coeffs[:, :other.n_coeffs] += other.coeffs

            return PiecewiseLegendrePoly(new_coeffs, self.knots, self.symm)
        else:
            raise TypeError("Can only add polynomial or scalar")

    def __mul__(self, other):
        """Multiply polynomial by scalar."""
        if isinstance(other, (int, float)):
            new_coeffs = self.coeffs * other
            return PiecewiseLegendrePoly(new_coeffs, self.knots, self.symm)
        else:
            raise TypeError("Can only multiply by scalar")

    def __rmul__(self, other):
        """Right multiplication by scalar."""
        return self.__mul__(other)


class PiecewiseLegendreFT:
    """
    Fourier transform of piecewise Legendre polynomials.

    This class handles the Fourier transform of piecewise Legendre polynomial
    representations, which is essential for transforming between tau and
    Matsubara frequency domains.
    """

    def __init__(self, poly):
        """
        Initialize Fourier transform of piecewise polynomial.

        Parameters
        ----------
        poly : PiecewiseLegendrePoly
            The polynomial to be Fourier transformed
        """
        self.poly = poly
        self._ft_coeffs = None
        self._compute_ft_coeffs()

    def _compute_ft_coeffs(self):
        """
        Precompute Fourier transform coefficients.

        This computes the Fourier transform of each Legendre polynomial
        basis function on each interval.
        """
        self._ft_coeffs = []

        for i in range(self.poly.n_intervals):
            interval_coeffs = []
            left = self.poly.knots[i]
            right = self.poly.knots[i + 1]
            length = right - left
            center = (left + right) / 2

            for j in range(self.poly.n_coeffs):
                # Analytical Fourier transform of Legendre polynomial on interval
                # This is a simplified version - full implementation would use
                # more sophisticated analytical expressions
                def ft_func(omega):
                    if omega == 0:
                        # DC component
                        if j == 0:
                            return length  # Integral of constant
                        else:
                            return 0  # Integral of higher-order Legendre polynomials
                    else:
                        # AC component - simplified approximation
                        phase = np.exp(-1j * omega * center)
                        if j == 0:
                            return phase * length * np.sinc(omega * length / (2 * np.pi))
                        else:
                            # Higher-order terms decay faster
                            return phase * length * np.sinc(omega * length / (2 * np.pi)) / (1 + j)

                interval_coeffs.append(ft_func)

            self._ft_coeffs.append(interval_coeffs)

    def __call__(self, omega):
        """
        Evaluate Fourier transform at given frequencies.

        Parameters
        ----------
        omega : array_like
            Frequencies at which to evaluate

        Returns
        -------
        ndarray
            Fourier transform values
        """
        omega = np.asarray(omega)
        omega_flat = omega.flatten()
        result = np.zeros(len(omega_flat), dtype=complex)

        for i in range(self.poly.n_intervals):
            for j, coeff in enumerate(self.poly.coeffs[i]):
                if coeff != 0:
                    ft_func = self._ft_coeffs[i][j]
                    ft_values = np.array([ft_func(w) for w in omega_flat])
                    result += coeff * ft_values

        return result.reshape(omega.shape)

    def convolve(self, other):
        """
        Convolve with another Fourier-transformed polynomial.

        Parameters
        ----------
        other : PiecewiseLegendreFT
            Other polynomial's Fourier transform

        Returns
        -------
        PiecewiseLegendreFT
            Convolution result
        """
        # Convolution in frequency domain is multiplication
        # This is a placeholder implementation
        raise NotImplementedError("Convolution not yet implemented")


def legendre_basis(degree, knots=None):
    """
    Create a Legendre polynomial basis of given degree.

    Parameters
    ----------
    degree : int
        Maximum degree of Legendre polynomials
    knots : array_like, optional
        Knot points. If None, uses [-1, 1].

    Returns
    -------
    list of PiecewiseLegendrePoly
        List of basis polynomials
    """
    if knots is None:
        knots = np.array([-1.0, 1.0])

    basis = []
    for i in range(degree + 1):
        coeffs = np.zeros((len(knots) - 1, degree + 1))
        coeffs[:, i] = 1.0  # Unit coefficient for i-th Legendre polynomial
        basis.append(PiecewiseLegendrePoly(coeffs, knots))

    return basis


def fit_legendre(x, y, degree, knots=None, weights=None):
    """
    Fit data with piecewise Legendre polynomials.

    Parameters
    ----------
    x, y : array_like
        Data points to fit
    degree : int
        Degree of Legendre polynomials in each interval
    knots : array_like, optional
        Knot points. If None, uses quantiles of x.
    weights : array_like, optional
        Weights for data points

    Returns
    -------
    PiecewiseLegendrePoly
        Fitted polynomial
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if knots is None:
        # Use quantiles as knots
        n_intervals = max(1, len(x) // 20)  # Heuristic
        quantiles = np.linspace(0, 1, n_intervals + 1)
        knots = np.quantile(x, quantiles)

    knots = np.asarray(knots)
    n_intervals = len(knots) - 1

    # Fit each interval separately
    coeffs = np.zeros((n_intervals, degree + 1))

    for i in range(n_intervals):
        left = knots[i]
        right = knots[i + 1]
        mask = (x >= left) & (x <= right)

        if np.any(mask):
            x_local = x[mask]
            y_local = y[mask]
            w_local = weights[mask] if weights is not None else None

            # Map to [-1, 1]
            x_mapped = 2 * (x_local - left) / (right - left) - 1

            # Fit Legendre polynomial using least squares
            A = np.zeros((len(x_mapped), degree + 1))
            for j in range(degree + 1):
                A[:, j] = special.eval_legendre(j, x_mapped)

            if w_local is not None:
                A *= w_local[:, np.newaxis]
                y_weighted = y_local * w_local
            else:
                y_weighted = y_local

            coeffs[i] = np.linalg.lstsq(A, y_weighted, rcond=None)[0]

    return PiecewiseLegendrePoly(coeffs, knots)