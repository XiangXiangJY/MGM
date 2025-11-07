# -*- coding: utf-8 -*-
"""
Quaternion k-means + clustering metrics (ARI/NMI/Purity/ACC)

- Works directly on dtype=quaternion arrays (from `numpy-quaternion`)
- Uses k-means++ initialization adapted to quaternion distances
- No real-structure expansion; distances are the Euclidean norms of
  quaternion differences (which are real scalars)

Author: xiangxiang
"""

import numpy as np
import quaternion  # pip install numpy-quaternion
import warnings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------
# Distance helpers (return float64 arrays)
# ------------------------------------------------------------

def quat_dist2_rows(Q, C):
    """
    Squared distances between N quaternion-vectors and k centers.

    Q : (N, d) quaternion array
    C : (k, d) quaternion array
    Returns
    -------
    D2 : (N, k) float64
         D2[i, j] = sum_{t=1..d} | Q[i,t] - C[j,t] |^2
    """
    diff = Q[:, None, :] - C[None, :, :]   # (N, k, d) quaternion
    dist2 = np.abs(diff)**2                # (N, k, d) float
    return dist2.sum(axis=2).astype(np.float64)  # (N, k) float64


# ------------------------------------------------------------
# k-means++ init for quaternion data
# ------------------------------------------------------------

def kmeanspp_init_quat(Q, n_clusters, rng=None):
    """
    k-means++ initialization indices for quaternion data.

    Q : (N, d) quaternion array
    n_clusters : int
    rng : numpy Generator (optional)

    Returns
    -------
    idx : (n_clusters,) int array of chosen initial centers
    """
    N = Q.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    centers_idx = [int(rng.integers(0, N))]

    # distances to first center (float64!)
    D2 = quat_dist2_rows(Q, Q[centers_idx]).min(axis=1)  # (N,)
    D2 = np.maximum(D2, 0.0).astype(np.float64)

    while len(centers_idx) < n_clusters:
        tot = D2.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            # fallback: uniform among non-chosen
            pool = np.setdiff1d(np.arange(N), np.array(centers_idx))
            centers_idx.append(int(rng.choice(pool)))
        else:
            p = (D2 / tot).astype(np.float64)
            centers_idx.append(int(rng.choice(N, p=p)))

        # update distances to nearest chosen center
        D2 = np.minimum(D2, quat_dist2_rows(Q, Q[centers_idx][-1:]).ravel())
        D2 = np.maximum(D2, 0.0).astype(np.float64)

    return np.array(centers_idx, dtype=int)


# ------------------------------------------------------------
# Core quaternion k-means
# ------------------------------------------------------------

def _quat_kmeans_single(Q, n_clusters, max_iter=500, tol=1e-4,
                        init="k-means++", rng=None):
    """
    One run of quaternion k-means.

    Q : (N, d) quaternion array
    n_clusters : int
    max_iter : int
    tol : float (relative center movement tolerance)
    init : "k-means++" or ndarray[(k,d)] of initial centers
    rng : numpy Generator

    Returns
    -------
    labels : (N,) int
    centers : (k, d) quaternion
    inertia : float (sum of squared distances)
    """
    N, d = Q.shape
    if rng is None:
        rng = np.random.default_rng()

    # --- init centers ---
    if isinstance(init, str) and init == "k-means++":
        idx0 = kmeanspp_init_quat(Q, n_clusters, rng=rng)
        centers = Q[idx0].copy()
    elif isinstance(init, np.ndarray):
        assert init.shape == (n_clusters, d)
        centers = init.astype(quaternion.quaternion)
    else:
        raise ValueError("init must be 'k-means++' or an array of centers")

    prev_move = np.inf
    labels = np.zeros(N, dtype=int)

    for it in range(max_iter):
        # assignment
        D2 = quat_dist2_rows(Q, centers)       # (N, k) float
        labels = np.argmin(D2, axis=1)

        # update centers: arithmetic mean in H^d (component-wise mean)
        new_centers = centers.copy()
        for k in range(n_clusters):
            mask = (labels == k)
            if not np.any(mask):
                # empty cluster: re-seed with farthest point
                far = np.argmax(D2.min(axis=1))
                new_centers[k] = Q[far]
            else:
                new_centers[k] = np.mean(Q[mask], axis=0)

        # convergence check (center movement)
        move = np.sum(np.abs(new_centers - centers))  # float
        if prev_move != 0 and move / max(prev_move, 1e-12) < tol:
            centers = new_centers
            break

        centers = new_centers
        prev_move = move

    inertia = float(quat_dist2_rows(Q, centers).min(axis=1).sum())
    return labels, centers, inertia


def quaternion_kmeans(Q, n_clusters, n_init=5, max_iter=500, tol=1e-4,
                      init="k-means++", random_state=None):
    """
    Multi-start quaternion k-means: keep best (lowest inertia).

    Returns best_labels, best_centers
    """
    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        labels, centers, inertia = _quat_kmeans_single(
            Q, n_clusters, max_iter=max_iter, tol=tol, init=init, rng=rng
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers


# ------------------------------------------------------------
# Metrics (same as your real-valued version)
# ------------------------------------------------------------

def computeARI(LABELS, y):
    ARI = np.zeros(LABELS.shape[0])
    for s in range(LABELS.shape[0]):
        ARI[s] = adjusted_rand_score(y, LABELS[s, :])
    return float(np.mean(ARI))

def computeNMI(LABELS, y):
    NMI = np.zeros(LABELS.shape[0])
    for s in range(LABELS.shape[0]):
        NMI[s] = normalized_mutual_info_score(y, LABELS[s, :])
    return float(np.mean(NMI))

def computePurity(LABELS, y):
    PUR = np.zeros(LABELS.shape[0])
    for s in range(LABELS.shape[0]):
        cm = contingency_matrix(y, LABELS[s, :])
        PUR[s] = np.sum(np.max(cm, axis=0)) / y.shape[0]
    return float(np.mean(PUR))

def computeACC(LABELS, y):
    ACC = np.zeros(LABELS.shape[0])
    for s in range(LABELS.shape[0]):
        cm = contingency_matrix(y, LABELS[s, :])
        r, c = linear_sum_assignment(cm, maximize=True)
        ACC[s] = np.sum(cm[r, c]) / y.shape[0]
    return float(np.mean(ACC))


# ------------------------------------------------------------
# Public API: run quaternion k-means and scores (mirrors your pattern)
# ------------------------------------------------------------

def computeKMeans_quaternion(Q, y, max_state=10, n_clusters=None,
                             max_iter=500, tol=1e-4, n_init=5,
                             init="k-means++"):
    """
    Quaternion k-means. Returns LABELS with one row per random_state.

    Q : (N, d) quaternion array
    y : (N,) int labels (ground truth)
    max_state : number of different random seeds to try
    n_clusters : defaults to number of unique labels
    """
    if n_clusters is None:
        n_clusters = np.unique(y).shape[0]

    N = Q.shape[0]
    LABELS = np.zeros((max_state, N), dtype=int)

    for state in range(max_state):
        labels, _ = quaternion_kmeans(
            Q, n_clusters,
            n_init=n_init, max_iter=max_iter, tol=tol,
            init=init, random_state=state
        )
        LABELS[state, :] = labels
    return LABELS


def computeClusteringScore_quaternion(Q, y, max_state=10,
                                      n_clusters=None,
                                      max_iter=500, tol=1e-4, n_init=5,
                                      init="k-means++"):
    """
    Q : (N, d) quaternion data
    y : (N,) ground-truth labels
    Returns
    -------
    ari, nmi, purity, acc, LABELS
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LABELS = computeKMeans_quaternion(
            Q, y,
            max_state=max_state,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            init=init
        )
    ari = computeARI(LABELS, y)
    nmi = computeNMI(LABELS, y)
    purity = computePurity(LABELS, y)
    acc = computeACC(LABELS, y)
    return ari, nmi, purity, acc, LABELS