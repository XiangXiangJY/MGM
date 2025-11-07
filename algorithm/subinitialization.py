# initialization2.py
# -*- coding: utf-8 -*-

"""
update 08-28-2025
author Xiang Xiang
Lightweight NNDSVD-based initializer for TNMF.
Input:  X  (genes_in_cluster \times cells)  i.e., X = X_sub.T
Output: W (genes_in_cluster \times r), H (r \times cells)
"""

import numpy as np
from sklearn.utils.extmath import squared_norm
from math import sqrt

def sub_norm_vec(x: np.ndarray) -> float:
    """Euclidean norm using squared_norm (stable/fast)."""
    return sqrt(squared_norm(x))

def sub_nndsvd(X: np.ndarray, n_components: int, eps: float = 1e-8):
    """
    NNDSVD (full SVD) initializer (nonnegative). Returns W, H.
    Shapes:
      X: (m, n)  -> W: (m, r), H: (r, n), r = n_components
    """
    # full SVD (more accurate than randomized init for small blocks)
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    r = n_components

    # guard if requested rank > available
    r_eff = min(r, u.shape[1], vh.shape[0])
    U = u[:, :r_eff]
    V = vh[:r_eff, :]
    S = s[:r_eff]

    W = np.zeros((X.shape[0], r), dtype=X.dtype)
    H = np.zeros((r, X.shape[1]), dtype=X.dtype)

    # leading component (nonnegative)
    if r_eff >= 1:
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    # remaining components
    for j in range(1, r_eff):
        x = U[:, j]
        y = V[j, :]

        # positive/negative parts
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # norms
        x_p_nrm, y_p_nrm = sub_norm_vec(x_p), sub_norm_vec(y_p)
        x_n_nrm, y_n_nrm = sub_norm_vec(x_n), sub_norm_vec(y_n)

        m_p = x_p_nrm * y_p_nrm
        m_n = x_n_nrm * y_n_nrm

        if m_p > m_n:
            u_hat = x_p / (x_p_nrm + 1e-20)
            v_hat = y_p / (y_p_nrm + 1e-20)
            sigma = m_p
        else:
            u_hat = x_n / (x_n_nrm + 1e-20)
            v_hat = y_n / (y_n_nrm + 1e-20)
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u_hat
        H[j, :] = lbd * v_hat

    # fill any tiny entries with the average of X (classic NNDSVDa)
    avg = float(np.mean(X[X > 0])) if np.any(X > 0) else eps
    W[W < eps] = avg
    H[H < eps] = avg

    # if requested rank > effective rank, pad with small positive noise
    if r_eff < r:
        rng = np.random.default_rng(0)
        pad_w = rng.random((X.shape[0], r - r_eff)) * avg
        pad_h = rng.random((r - r_eff, X.shape[1])) * avg
        W[:, r_eff:] = pad_w
        H[r_eff:, :] = pad_h

    return W, H

def sub_initialize(X: np.ndarray, n_components: int = 3, eps: float = 1e-8):
    """
    Public API:
      X must be X_sub.T (genes_in_cluster  cells).
      Returns W (genes_in_cluster  n_components), H (n_components  cells).
    """
    # safety: replace NaNs/inf with zeros
    X = np.asarray(X, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)

    # ensure nonnegative input for TNMF initialization (clip tiny negatives)
    X = np.clip(X, 0.0, None)

    # run NNDSVD initializer
    W, H = sub_nndsvd(X, n_components=n_components, eps=eps)

    # final safety: strictly nonnegative
    W = np.clip(W, 0.0, None)
    H = np.clip(H, 0.0, None)
    return W, H