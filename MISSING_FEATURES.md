# Missing Features in pysparseir

This document provides a comprehensive list of features available in the pure Python `sparse-ir` library that are not yet implemented in `pysparseir`.

## Summary

The following modules from `sparse-ir` are completely missing in `pysparseir`:
- `augment` - Augmented basis functionality
- `dlr` - Discrete Lehmann Representation
- `adapter` - Drop-in replacement for irbasis module
- `poly` - Piecewise polynomial functionality
- `abstract` - Abstract base classes
- `svd` - SVD computation utilities

## Detailed Analysis

### 1. Augmented Basis (`augment.py`)

**Missing Classes:**
- `AugmentedBasis` - Augmented basis on imaginary-time/frequency axis
- `AugmentedTauFunction` - Augmented functions in tau space
- `AugmentedMatsubaraFunction` - Augmented functions in Matsubara space
- `AbstractAugmentation` - Base class for augmentations
- `TauConst` - Constant in imaginary time
- `TauLinear` - Linear function in imaginary time
- `MatsubaraConst` - Constant in Matsubara frequency

**Use Case:** Essential for vertex-like quantities and self-energies in many-body physics.

### 2. Discrete Lehmann Representation (`dlr.py`)

**Missing Classes:**
- `DiscreteLehmannRepresentation` - DLR basis with poles at IR extrema
- `MatsubaraPoles` - Matsubara frequency representation using poles
- `TauPoles` - Imaginary time representation using poles

**Use Case:** Alternative representation that can be more efficient for certain calculations.

### 3. Adapter Module (`adapter.py`)

**Missing Classes:**
- `Basis` - Compatibility wrapper for irbasis

**Missing Functions:**
- `load(statistics, Lambda, h5file=None)` - Load basis with irbasis interface
- `sampling_points_x(b, whichl)` - Optimal sampling points in x space
- `sampling_points_y(b, whichl)` - Optimal sampling points in y space  
- `sampling_points_matsubara(b, whichl)` - Sampling points in Matsubara domain

**Use Case:** Provides backward compatibility with existing irbasis code.

### 4. Polynomial Functions (`poly.py`)

**Missing Classes:**
- `PiecewiseLegendrePoly` - Piecewise Legendre polynomial representation
- `PiecewiseLegendreFT` - Fourier transform of piecewise Legendre polynomials

**Use Case:** Core mathematical infrastructure used internally by basis functions.

### 5. Abstract Base Classes (`abstract.py`)

**Missing Classes:**
- `AbstractBasis` - Abstract base class defining the basis interface

**Use Case:** Defines the common interface for all basis types.

### 6. SVD Computation (`svd.py`)

**Missing Functions:**
- `compute(a_matrix, n_sv_hint=None, strategy='fast')` - Flexible SVD computation with multiple strategies

**Use Case:** Advanced SVD computation with different accuracy/speed tradeoffs.

### 7. Partially Implemented Features

#### SVEResult Class
Currently in `pysparseir`, the `SVEResult` class is missing:
- Access to `u` functions (left singular functions)
- Access to `v` functions (right singular functions)
- The `part()` method that returns the (u, s, v) tuple

#### FiniteTempBasis Class
Missing methods/properties:
- `significance` property - Significance values for basis functions
- `default_omega_sampling_points()` - Sampling points on real frequency axis
- Direct array indexing with `__getitem__` for basis truncation
- `v` property - Access to v basis functions (omega/frequency space)

#### Missing Utility Functions
From the main `__init__.py`, the following are not exposed:
- Direct access to `compute` as `compute_sve` (currently available as `compute_sve()`)

### 8. Internal Utilities

While not part of the public API, `sparse-ir` has several internal modules that provide functionality that may need to be replicated:
- `_util` - Utility functions for range checking, Matsubara frequency validation
- `_roots` - Root finding algorithms
- `_gauss` - Gauss quadrature rules

## Priority Recommendations

Based on typical use cases, the implementation priority should be:

1. **High Priority:**
   - Complete `SVEResult` implementation (u, v access)
   - Complete `FiniteTempBasis` methods (significance, v functions, omega sampling)
   - `AbstractBasis` base class

2. **Medium Priority:**
   - `AugmentedBasis` and related classes (important for vertex calculations)
   - `DiscreteLehmannRepresentation` (alternative representation)

3. **Lower Priority:**
   - `adapter` module (mainly for backward compatibility)
   - `poly` module (unless needed internally)
   - Advanced SVD strategies

## Implementation Notes

Most of these features will require either:
1. Extensions to the C++ libsparseir library
2. Pure Python implementations on top of the existing C API
3. Hybrid approaches where performance-critical parts use C++ and high-level logic stays in Python

The specific approach will depend on performance requirements and the complexity of interfacing with the existing C++ code.