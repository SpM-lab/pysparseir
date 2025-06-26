"""
Enhanced SVD computation for SparseIR.

This module provides flexible SVD computation with multiple strategies
for different accuracy/speed tradeoffs.
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import svds
import warnings


def compute(a_matrix, n_sv_hint=None, strategy='fast'):
    """
    Flexible SVD computation with multiple strategies.

    This function provides various SVD computation strategies to balance
    accuracy and computational speed based on the specific requirements.

    Parameters
    ----------
    a_matrix : array_like
        Matrix to decompose via SVD
    n_sv_hint : int, optional
        Hint for the expected number of significant singular values.
        If provided, can be used to optimize the computation.
    strategy : str, optional
        SVD computation strategy:
        - 'fast': Fast computation with potentially lower accuracy
        - 'accurate': High accuracy computation (slower)
        - 'economic': Memory-efficient computation
        - 'adaptive': Automatically choose strategy based on matrix properties

    Returns
    -------
    u : ndarray
        Left singular vectors
    s : ndarray
        Singular values in descending order
    vt : ndarray
        Right singular vectors (transposed)

    Notes
    -----
    The 'fast' strategy may use approximation methods for large matrices.
    The 'accurate' strategy uses high-precision algorithms.
    The 'economic' strategy minimizes memory usage.
    The 'adaptive' strategy chooses the best approach based on matrix size and condition.
    """
    a_matrix = np.asarray(a_matrix)

    if a_matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    m, n = a_matrix.shape

    # Choose strategy
    if strategy == 'adaptive':
        strategy = _choose_adaptive_strategy(a_matrix, n_sv_hint)

    if strategy == 'fast':
        return _fast_svd(a_matrix, n_sv_hint)
    elif strategy == 'accurate':
        return _accurate_svd(a_matrix, n_sv_hint)
    elif strategy == 'economic':
        return _economic_svd(a_matrix, n_sv_hint)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _choose_adaptive_strategy(a_matrix, n_sv_hint):
    """
    Automatically choose the best SVD strategy based on matrix properties.

    Parameters
    ----------
    a_matrix : ndarray
        Input matrix
    n_sv_hint : int or None
        Hint for number of singular values

    Returns
    -------
    str
        Chosen strategy
    """
    m, n = a_matrix.shape
    matrix_size = m * n

    # For small matrices, use accurate method
    if matrix_size < 10000:
        return 'accurate'

    # For very large matrices, use economic method
    if matrix_size > 1000000:
        return 'economic'

    # For medium matrices with hint about low rank, use fast method
    if n_sv_hint is not None and n_sv_hint < min(m, n) // 4:
        return 'fast'

    # Default to accurate for medium-sized matrices
    return 'accurate'


def _fast_svd(a_matrix, n_sv_hint):
    """
    Fast SVD computation using approximation methods when beneficial.

    Parameters
    ----------
    a_matrix : ndarray
        Input matrix
    n_sv_hint : int or None
        Hint for number of singular values

    Returns
    -------
    tuple
        (u, s, vt) SVD decomposition
    """
    m, n = a_matrix.shape

    # If we have a hint and it suggests low rank, use sparse SVD
    if n_sv_hint is not None and n_sv_hint < min(m, n) // 2:
        try:
            # Use sparse SVD for potentially faster computation
            k = min(n_sv_hint * 2, min(m, n) - 1)  # Get a few extra for safety
            u, s, vt = svds(a_matrix, k=k, which='LM')

            # Sort in descending order (svds returns ascending)
            idx = np.argsort(s)[::-1]
            u = u[:, idx]
            s = s[idx]
            vt = vt[idx, :]

            return u, s, vt

        except Exception:
            # Fall back to full SVD if sparse SVD fails
            warnings.warn("Sparse SVD failed, falling back to full SVD")
            pass

    # Use standard scipy SVD
    return scipy_svd(a_matrix, full_matrices=False)


def _accurate_svd(a_matrix, n_sv_hint):
    """
    High-accuracy SVD computation.

    Parameters
    ----------
    a_matrix : ndarray
        Input matrix
    n_sv_hint : int or None
        Hint for number of singular values (unused for accurate computation)

    Returns
    -------
    tuple
        (u, s, vt) SVD decomposition
    """
    # Use scipy's SVD with full precision
    try:
        # Try with lapack_driver='gesdd' for potentially better accuracy
        u, s, vt = scipy_svd(a_matrix, full_matrices=False, lapack_driver='gesdd')
    except Exception:
        # Fall back to default driver
        u, s, vt = scipy_svd(a_matrix, full_matrices=False)

    return u, s, vt


def _economic_svd(a_matrix, n_sv_hint):
    """
    Memory-efficient SVD computation.

    Parameters
    ----------
    a_matrix : ndarray
        Input matrix
    n_sv_hint : int or None
        Hint for number of singular values

    Returns
    -------
    tuple
        (u, s, vt) SVD decomposition
    """
    m, n = a_matrix.shape

    # For very wide matrices, compute SVD of A^T * A
    if n > 2 * m:
        # Compute eigendecomposition of A^T * A
        ata = a_matrix.T @ a_matrix
        eigenvals, eigenvecs = np.linalg.eigh(ata)

        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Compute singular values and vectors
        s = np.sqrt(np.maximum(eigenvals, 0))
        vt = eigenvecs.T

        # Compute u
        nonzero_mask = s > 1e-15
        if np.any(nonzero_mask):
            u_reduced = a_matrix @ eigenvecs[:, nonzero_mask] / s[nonzero_mask]
            u = np.zeros((m, len(s)))
            u[:, nonzero_mask] = u_reduced
        else:
            u = np.zeros((m, len(s)))

        return u, s, vt

    # For very tall matrices, compute SVD of A * A^T
    elif m > 2 * n:
        # Compute eigendecomposition of A * A^T
        aat = a_matrix @ a_matrix.T
        eigenvals, eigenvecs = np.linalg.eigh(aat)

        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Compute singular values and vectors
        s = np.sqrt(np.maximum(eigenvals, 0))
        u = eigenvecs

        # Compute vt
        nonzero_mask = s > 1e-15
        if np.any(nonzero_mask):
            vt_reduced = (eigenvecs[:, nonzero_mask].T @ a_matrix) / s[nonzero_mask][:, np.newaxis]
            vt = np.zeros((len(s), n))
            vt[nonzero_mask] = vt_reduced
        else:
            vt = np.zeros((len(s), n))

        return u, s, vt

    # For reasonably shaped matrices, use standard SVD
    return scipy_svd(a_matrix, full_matrices=False)


def truncated_svd(a_matrix, rank, strategy='fast'):
    """
    Compute truncated SVD with specified rank.

    Parameters
    ----------
    a_matrix : array_like
        Matrix to decompose
    rank : int
        Desired rank of the approximation
    strategy : str, optional
        SVD strategy to use

    Returns
    -------
    tuple
        (u, s, vt) Truncated SVD with exactly 'rank' components
    """
    u, s, vt = compute(a_matrix, n_sv_hint=rank, strategy=strategy)

    # Truncate to desired rank
    rank = min(rank, len(s))
    return u[:, :rank], s[:rank], vt[:rank, :]


def adaptive_rank_svd(a_matrix, tolerance=1e-12, strategy='fast'):
    """
    Compute SVD with adaptive rank based on singular value tolerance.

    Parameters
    ----------
    a_matrix : array_like
        Matrix to decompose
    tolerance : float, optional
        Relative tolerance for truncating small singular values
    strategy : str, optional
        SVD strategy to use

    Returns
    -------
    tuple
        (u, s, vt) SVD truncated to significant singular values
    """
    u, s, vt = compute(a_matrix, strategy=strategy)

    # Find cutoff based on relative tolerance
    if len(s) > 0:
        cutoff = s[0] * tolerance
        significant_mask = s >= cutoff
        rank = np.sum(significant_mask)

        if rank > 0:
            return u[:, :rank], s[:rank], vt[:rank, :]

    # Return at least one singular value if available
    if len(s) > 0:
        return u[:, :1], s[:1], vt[:1, :]
    else:
        return u, s, vt


def condition_number(a_matrix, strategy='fast'):
    """
    Compute condition number of matrix using SVD.

    Parameters
    ----------
    a_matrix : array_like
        Input matrix
    strategy : str, optional
        SVD strategy to use

    Returns
    -------
    float
        Condition number (ratio of largest to smallest singular value)
    """
    _, s, _ = compute(a_matrix, strategy=strategy)

    if len(s) == 0:
        return np.inf

    s_max = s[0]
    s_min = s[-1]

    if s_min == 0:
        return np.inf
    else:
        return s_max / s_min


def effective_rank(a_matrix, tolerance=1e-12, strategy='fast'):
    """
    Compute effective rank of matrix based on singular value threshold.

    Parameters
    ----------
    a_matrix : array_like
        Input matrix
    tolerance : float, optional
        Relative tolerance for significant singular values
    strategy : str, optional
        SVD strategy to use

    Returns
    -------
    int
        Effective rank
    """
    _, s, _ = compute(a_matrix, strategy=strategy)

    if len(s) == 0:
        return 0

    cutoff = s[0] * tolerance
    return np.sum(s >= cutoff)


class SVDResult:
    """
    Container for SVD results with additional utilities.

    This class provides a convenient interface for working with SVD results
    and includes methods for reconstruction, rank analysis, etc.
    """

    def __init__(self, u, s, vt, strategy_used=None):
        """
        Initialize SVD result.

        Parameters
        ----------
        u : ndarray
            Left singular vectors
        s : ndarray
            Singular values
        vt : ndarray
            Right singular vectors (transposed)
        strategy_used : str, optional
            Strategy that was used for computation
        """
        self.u = u
        self.s = s
        self.vt = vt
        self.strategy_used = strategy_used

    @property
    def rank(self):
        """Rank of the decomposition."""
        return len(self.s)

    def reconstruct(self, rank=None):
        """
        Reconstruct matrix from SVD components.

        Parameters
        ----------
        rank : int, optional
            Rank for reconstruction. If None, uses all components.

        Returns
        -------
        ndarray
            Reconstructed matrix
        """
        if rank is None:
            rank = self.rank

        rank = min(rank, self.rank)
        return self.u[:, :rank] @ np.diag(self.s[:rank]) @ self.vt[:rank, :]

    def truncate(self, rank):
        """
        Create truncated version of SVD result.

        Parameters
        ----------
        rank : int
            Desired rank

        Returns
        -------
        SVDResult
            Truncated SVD result
        """
        rank = min(rank, self.rank)
        return SVDResult(
            self.u[:, :rank],
            self.s[:rank],
            self.vt[:rank, :],
            self.strategy_used
        )

    def condition_number(self):
        """Compute condition number."""
        if self.rank == 0:
            return np.inf
        return self.s[0] / self.s[-1] if self.s[-1] != 0 else np.inf

    def effective_rank(self, tolerance=1e-12):
        """Compute effective rank."""
        if self.rank == 0:
            return 0
        cutoff = self.s[0] * tolerance
        return np.sum(self.s >= cutoff)