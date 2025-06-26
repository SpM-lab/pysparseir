"""
Augmented basis functionality for SparseIR.

This module implements augmented bases on imaginary-time/frequency axis
for handling vertex-like quantities and self-energies in many-body physics.
"""

import numpy as np
from abc import ABC, abstractmethod
from .abstract import AbstractBasis
from .basis import FiniteTempBasis


class AbstractAugmentation(ABC):
    """Base class for augmentations."""

    @abstractmethod
    def eval_tau(self, tau):
        """Evaluate augmentation function at tau points."""
        raise NotImplementedError()

    @abstractmethod
    def eval_matsubara(self, n):
        """Evaluate augmentation function at Matsubara frequencies."""
        raise NotImplementedError()


class TauConst(AbstractAugmentation):
    """
    Constant function in imaginary time.

    This augmentation represents a constant function c(τ) = c for all τ ∈ [0, β].
    """

    def __init__(self, value=1.0):
        """
        Initialize constant augmentation.

        Parameters
        ----------
        value : float, optional
            Constant value (default: 1.0)
        """
        self.value = float(value)

    def eval_tau(self, tau):
        """Evaluate at tau points."""
        tau = np.asarray(tau)
        return np.full(tau.shape, self.value)

    def eval_matsubara(self, n):
        """Evaluate at Matsubara frequencies."""
        n = np.asarray(n)
        # Constant in tau -> delta function at n=0
        result = np.zeros(n.shape, dtype=complex)
        result[n == 0] = self.value
        return result


class TauLinear(AbstractAugmentation):
    """
    Linear function in imaginary time.

    This augmentation represents a linear function c(τ) = slope * τ + intercept.
    """

    def __init__(self, slope=1.0, intercept=0.0):
        """
        Initialize linear augmentation.

        Parameters
        ----------
        slope : float, optional
            Slope of the linear function (default: 1.0)
        intercept : float, optional
            Intercept of the linear function (default: 0.0)
        """
        self.slope = float(slope)
        self.intercept = float(intercept)

    def eval_tau(self, tau):
        """Evaluate at tau points."""
        tau = np.asarray(tau)
        return self.slope * tau + self.intercept

    def eval_matsubara(self, n):
        """Evaluate at Matsubara frequencies."""
        n = np.asarray(n)
        result = np.zeros(n.shape, dtype=complex)

        # For linear function: slope * τ + intercept
        # Fourier transform gives contributions at n=0 (intercept) and n=±1 (slope)
        if np.any(n == 0):
            result[n == 0] = self.intercept

        # Linear term contributes to n=±1 terms
        # This is a simplified implementation - full implementation would need
        # proper Fourier transform of the linear function
        nonzero_mask = n != 0
        if np.any(nonzero_mask):
            result[nonzero_mask] = self.slope * 1j / n[nonzero_mask]

        return result


class MatsubaraConst(AbstractAugmentation):
    """
    Constant function in Matsubara frequency.

    This augmentation represents a constant function in Matsubara space.
    """

    def __init__(self, value=1.0):
        """
        Initialize constant Matsubara augmentation.

        Parameters
        ----------
        value : float, optional
            Constant value (default: 1.0)
        """
        self.value = float(value)

    def eval_tau(self, tau):
        """Evaluate at tau points."""
        # Constant in Matsubara -> delta functions in tau (simplified)
        tau = np.asarray(tau)
        return np.full(tau.shape, self.value)

    def eval_matsubara(self, n):
        """Evaluate at Matsubara frequencies."""
        n = np.asarray(n)
        return np.full(n.shape, self.value, dtype=complex)


class AugmentedTauFunction:
    """Augmented function in imaginary time."""

    def __init__(self, basis, coeffs, augmentations=None, augment_coeffs=None):
        """
        Initialize augmented tau function.

        Parameters
        ----------
        basis : FiniteTempBasis
            Underlying IR basis
        coeffs : array_like
            IR expansion coefficients
        augmentations : list of AbstractAugmentation, optional
            List of augmentation functions
        augment_coeffs : array_like, optional
            Coefficients for augmentation functions
        """
        self.basis = basis
        self.coeffs = np.asarray(coeffs)
        self.augmentations = augmentations or []
        self.augment_coeffs = np.asarray(augment_coeffs) if augment_coeffs is not None else np.array([])

        if len(self.augmentations) != len(self.augment_coeffs):
            raise ValueError("Number of augmentations must match number of augmentation coefficients")

    def __call__(self, tau):
        """Evaluate at tau points."""
        tau = np.asarray(tau)

        # Base IR contribution
        u_vals = self.basis.u(tau)  # Shape: (..., size)
        result = np.sum(u_vals * self.coeffs, axis=-1)

        # Add augmentation contributions
        for aug, coeff in zip(self.augmentations, self.augment_coeffs):
            result += coeff * aug.eval_tau(tau)

        return result


class AugmentedMatsubaraFunction:
    """Augmented function in Matsubara frequency."""

    def __init__(self, basis, coeffs, augmentations=None, augment_coeffs=None):
        """
        Initialize augmented Matsubara function.

        Parameters
        ----------
        basis : FiniteTempBasis
            Underlying IR basis
        coeffs : array_like
            IR expansion coefficients
        augmentations : list of AbstractAugmentation, optional
            List of augmentation functions
        augment_coeffs : array_like, optional
            Coefficients for augmentation functions
        """
        self.basis = basis
        self.coeffs = np.asarray(coeffs)
        self.augmentations = augmentations or []
        self.augment_coeffs = np.asarray(augment_coeffs) if augment_coeffs is not None else np.array([])

        if len(self.augmentations) != len(self.augment_coeffs):
            raise ValueError("Number of augmentations must match number of augmentation coefficients")

    def __call__(self, n):
        """Evaluate at Matsubara frequencies."""
        n = np.asarray(n)

        # Base IR contribution
        uhat_vals = self.basis.uhat(n)  # Shape: (..., size)
        result = np.sum(uhat_vals * self.coeffs, axis=-1)

        # Add augmentation contributions
        for aug, coeff in zip(self.augmentations, self.augment_coeffs):
            result += coeff * aug.eval_matsubara(n)

        return result


class AugmentedBasis(AbstractBasis):
    """
    Augmented basis on imaginary-time/frequency axis.

    This class extends a regular IR basis with additional augmentation functions
    to better handle vertex-like quantities and self-energies that may not be
    well-represented by the standard IR basis alone.
    """

    def __init__(self, basis, augmentations):
        """
        Initialize augmented basis.

        Parameters
        ----------
        basis : FiniteTempBasis
            Underlying IR basis
        augmentations : list of AbstractAugmentation
            List of augmentation functions to add
        """
        if not isinstance(basis, FiniteTempBasis):
            raise TypeError("basis must be a FiniteTempBasis")

        self.basis = basis
        self.augmentations = list(augmentations)
        self._size = basis.size + len(self.augmentations)

    @property
    def u(self):
        """Imaginary-time basis functions (IR + augmentations)."""
        return AugmentedTauBasisFunctions(self.basis, self.augmentations)

    @property
    def uhat(self):
        """Matsubara frequency basis functions (IR + augmentations)."""
        return AugmentedMatsubaraBasisFunctions(self.basis, self.augmentations)

    @property
    def statistics(self):
        """Quantum statistic."""
        return self.basis.statistics

    @property
    def size(self):
        """Total number of basis functions (IR + augmentations)."""
        return self._size

    @property
    def significance(self):
        """Significances of basis functions."""
        # IR significances + augmentation significances (set to small values)
        ir_sig = self.basis.significance
        aug_sig = np.full(len(self.augmentations), 1e-15)  # Very small significance
        return np.concatenate([ir_sig, aug_sig])

    @property
    def lambda_(self):
        """Basis cutoff parameter."""
        return self.basis.lambda_

    @property
    def beta(self):
        """Inverse temperature."""
        return self.basis.beta

    def default_tau_sampling_points(self, *, npoints=None):
        """Default sampling points on imaginary time axis."""
        return self.basis.default_tau_sampling_points(npoints=npoints)

    def default_matsubara_sampling_points(self, *, npoints=None, positive_only=False):
        """Default sampling points on Matsubara frequency axis."""
        return self.basis.default_matsubara_sampling_points(npoints=npoints, positive_only=positive_only)


class AugmentedTauBasisFunctions:
    """Augmented basis functions for imaginary time evaluation."""

    def __init__(self, basis, augmentations):
        self.basis = basis
        self.augmentations = augmentations

    def __call__(self, tau):
        """Evaluate all basis functions at tau points."""
        tau = np.asarray(tau)

        # IR basis functions
        u_vals = self.basis.u(tau)  # Shape: (..., ir_size)

        # Augmentation functions
        aug_vals = []
        for aug in self.augmentations:
            aug_vals.append(aug.eval_tau(tau))

        if aug_vals:
            aug_vals = np.array(aug_vals).T  # Shape: (..., n_aug)
            return np.concatenate([u_vals, aug_vals], axis=-1)
        else:
            return u_vals


class AugmentedMatsubaraBasisFunctions:
    """Augmented basis functions for Matsubara frequency evaluation."""

    def __init__(self, basis, augmentations):
        self.basis = basis
        self.augmentations = augmentations

    def __call__(self, n):
        """Evaluate all basis functions at Matsubara frequencies."""
        n = np.asarray(n)

        # IR basis functions
        uhat_vals = self.basis.uhat(n)  # Shape: (..., ir_size)

        # Augmentation functions
        aug_vals = []
        for aug in self.augmentations:
            aug_vals.append(aug.eval_matsubara(n))

        if aug_vals:
            aug_vals = np.array(aug_vals).T  # Shape: (..., n_aug)
            return np.concatenate([uhat_vals, aug_vals], axis=-1)
        else:
            return uhat_vals