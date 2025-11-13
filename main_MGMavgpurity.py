#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_MGM.py — PCA → multi-scale UMAP → Grassmann distance → Clustering
Baselines:
  (1) PCA→UMAP (per-scale) → clustering + average
  (2) PCA (at pca_dim) → clustering
  (3) PCA → PCA(out_dim) → clustering
  (4) RAW X → PCA(out_dim) → clustering   [NEW]

Batch runner WITHOUT TSV:
  - Fill DATASET_CONFIG below (a text block). One line per run with columns:
      name  out_dim  distance  subspace_dim  pca_dim  cluster_method  n_scales  mode  min_cap  max_cap
    * distance: geodesic | chordal | reflection
    * subspace_dim: integer or "auto"
    * pca_dim: integer (pre-UMAP PCA dimension)
    * cluster_method: spectral | kmeans  (applies to Grassmann + all baselines)
    * n_scales: integer, how many scales to generate
    * mode: lin | log | hybrid
    * min_cap / max_cap: integers or "auto"
       - These define the range of UMAP n_neighbors to sample from.
       - If "auto", values are derived from dataset size (see generate_scales()).

  - Run:    python main_grassmann_reflection.py
  - Output: results*/SUMMARY_all_<timestamp>.csv + .xlsx
"""

import os, json, time, glob, re
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

# ==================== USER CONFIG BLOCK ====================
DATASET_CONFIG = """
# name           out_dim  distance  subspace_dim  pca_dim  cluster_method  n_scales  mode     min_cap  max_cap
# -------------------------------------------------------------------------------------------------------------
#  
 # GSE84133human4      50      chordal   19            100      spectral        20       power1.6   5        50
#    GSE84133mouse1       20      chordal   10            50      spectral        11       power1.6   5        20
#   GSE84133mouse2       20      chordal   10            50      spectral        11       power1.6   5        20
#     GSE75748time       20      chordal   10            50      kmeans        11       power1.6   5        20
#    GSE94820       20      chordal   10            50     kmeans        11       power1.6   5        20
#    GSE67835       20      chordal    10            50    kmeans        11       power1.6   5        20
#    GSE75748cell       20     chordal   10            50     kmeans         11       power1.6   5        20
#    GSE109979_329cell       20      chordal    10            50      kmeans        11       power1.6   5        20
#  GSE84133human4      50      geodesic   19            100      spectral        19       power1.6   5        50

#  GSE75748time       100      chordal  23           200      spectral       25      power2   5       100
#  GSE94820       100     chordal  23           200     spectral       25      power2  5        100
#  GSE67835       100     chordal   23           200      spectral        25       power2  5        100
#  GSE75748cell       100      chordal   23          200      spectral        25       power2   5        100
#    GSE109979_329cell       100     chordal  23            200     spectral        25       power2   5        100
#  GSE84133human4    100     chordal   23           200      spectral        25       power2   5        100
#  GSE84133human4     100    geodesic   23           200      spectral       25      power2   5        100


#  GSE84133human1      100      chordal   23            200      kmeans        25       power2   5        100
  #GSE84133human2      100      chordal   23            200      spectral        25       power2   5        100

   #GSE84133human3     200      chordal   47            300     spectral        50       power2   5        200
        # GSE82187 50      chordal   19           100     spectral       20      power1.6   5        50
# GSE84133human3    100     chordal   23           200      spectral        25       power2   5        100
   GSE57249     15     chordal   8            20     kmeans        10       power1.6   5        15

# GSE84133human1     50      chordal   19           100     kmeans       20      power1.6   5        50
# GSE84133human2     50      chordal   19           100    kmeans       20      power1.6   5        50
# GSE84133human4     50      chordal   19           100    kmeans       20      power1.6   5        50


#  GSE84133human2      100       geodesic    30            200      spectral        30       power1.6   5        100

""".strip()
# ===========================================================

# -------------------- Paths / Global switches --------------------
DATA_PROCESS_PATH = "./SingleCellDataProcess/"
DATA_PATH = "./data/"
SAVE_DIR = "./projections"
RESULTS_DIR = "./results"
RANDOM_STATE = 1                 # default fallback; per-seed 会覆盖
SEEDS = [1, 3, 5, 7, 9]          # <<< 这里定义多 seed
STANDARDIZE_UMAP = True
SAVE_STACK = True
SKIP_PROJ = False
MAX_SAMPLES = None               # e.g., 2000 or None
USE_FLOAT64 = False              # False→float32
GAMMA = 1.0                      # kernel width factor for spectral graph (RBF on distances)
PCA_DIM_DEFAULT = 300            # used if a line omits pca_dim

# -------------------- Dependencies --------------------
from algorithm.auxilary import load_X, load_y, preprocess_data
from algorithm.reduction import reduction_wrapper

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from numpy.linalg import eigvals

# -------------------- Utility functions --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

# 保留原来的这两个函数，但后面我们不再调用它们
def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.max(cm, axis=0)) / len(y_true))

def acc_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(cm, maximize=True)
    return float(np.sum(cm[r, c]) / len(y_true))

# ===== 新增：只算 avg-purity =====
def avg_purity_score(y_true, y_pred):
    """
    avg-purity(P_t) = (1 / |L|) * sum_{t in labels}  sum_j (C[t,j] / n_t)^2
    这里的 C 是 contingency matrix (真实标签 × 预测标签)
    """
    cm = contingency_matrix(y_true, y_pred)
    per_label = []
    for i in range(cm.shape[0]):
        row = cm[i]
        n_i = row.sum()
        if n_i == 0:
            continue
        per_label.append(np.sum((row / n_i) ** 2))
    if not per_label:
        return 0.0
    return float(np.mean(per_label))

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def parse_int_auto(s):
    s = str(s).strip().lower()
    if s == "auto": return None
    return int(s)

# -------------------- Scale generation --------------------
def _auto_caps(n_samples, min_cap, max_cap):
    """
    Derive min/max caps if any is None (i.e., 'auto').
    Heuristics:
      - min_cap: default 5, lower-bounded by 3, upper-bounded by max_cap-1.
      - max_cap: min( max(15, int(0.15*n)), 120 ), also >= min_cap+1.
    """
    mcap = max_cap
    if mcap is None:
        mcap = int(min(max(15, round(0.15 * n_samples)), 120))
    ncap = min_cap
    if ncap is None:
        ncap = 5
    ncap = int(max(3, min(ncap, mcap - 1)))
    mcap = int(max(ncap + 1, mcap))
    return ncap, mcap

def _unique_sorted_ints(arr):
    arr = sorted(set(int(v) for v in arr if v is not None))
    return [v for v in arr if v > 0]

def generate_scales(n_samples, n_scales, mode="lin", min_cap=None, max_cap=None):
    """
    Generate a monotonically increasing list of 'n_neighbors' values.
      mode ∈ {"lin","log","hybrid","sqrt","power{p}','inv"}:
        - lin:    evenly spaced in [min_cap, max_cap]
        - log:    logarithmically spaced (denser at small neighbors)
        - hybrid: mix of small local (log-like), medium linear, and a few large caps
        - sqrt:   k(t) = min_cap + (max_cap-min_cap) * t^(1/2)
        - power{p}: k(t) = min_cap + (max_cap-min_cap) * t^p, e.g. "power0.4"
        - inv:    inverse-shaped schedule via normalized 1/(t+eps)
    Caps "auto" are resolved based on n_samples via _auto_caps().
    """
    n_scales = int(n_scales)
    mode = str(mode).lower()
    min_cap, max_cap = _auto_caps(n_samples, min_cap, max_cap)

    if n_scales <= 0:
        raise ValueError("n_scales must be >= 1")

    def _quantize_and_unique(vals):
        return _unique_sorted_ints(np.round(vals))

    if mode == "lin":
        vals = np.linspace(min_cap, max_cap, num=n_scales)
        scales = _quantize_and_unique(vals)

    elif mode == "log":
        a, b = np.log(max(min_cap, 1)), np.log(max_cap)
        vals = np.exp(np.linspace(a, b, num=n_scales))
        scales = _quantize_and_unique(vals)

    elif mode == "sqrt":
        t = np.linspace(0.0, 1.0, num=n_scales)
        vals = min_cap + (max_cap - min_cap) * np.sqrt(t)
        scales = _quantize_and_unique(vals)

    elif mode.startswith("power"):
        import re
        m = re.search(r"(power|pow)\s*([0-9]*\.?[0-9]+)", mode)
        p = float(m.group(2)) if m else 0.5
        p = max(1e-3, min(5.0, p))
        t = np.linspace(0.0, 1.0, num=n_scales)
        vals = min_cap + (max_cap - min_cap) * (t ** p)
        scales = _quantize_and_unique(vals)

    elif mode == "inv":
        t = np.linspace(0.0, 1.0, num=n_scales)
        eps = 1e-6
        f0 = 1.0 / (0.0 + eps)
        f1 = 1.0 / (1.0 + eps)
        f = 1.0 / (t + eps)
        g = (f0 - f) / (f0 - f1)
        vals = min_cap + (max_cap - min_cap) * g
        scales = _quantize_and_unique(vals)

    elif mode == "hybrid":
        k_small = max(2, int(round(0.4 * n_scales)))
        k_mid   = max(2, int(round(0.4 * n_scales)))
        k_large = max(1, n_scales - k_small - k_mid)

        a, b = np.log(max(min_cap, 1)), np.log(max_cap)
        small_vals = np.exp(np.linspace(a, np.log(min_cap + (max_cap - min_cap) * 0.35), num=k_small))
        mid_start = min_cap + (max_cap - min_cap) * 0.25
        mid_end   = min_cap + (max_cap - min_cap) * 0.75
        mid_vals  = np.linspace(mid_start, mid_end, num=k_mid)
        large_start = min_cap + (max_cap - min_cap) * 0.65
        large_vals  = np.linspace(large_start, max_cap, num=k_large)

        vals = np.concatenate([small_vals, mid_vals, large_vals])
        scales = _quantize_and_unique(vals)

        while len(scales) < n_scales:
            cand = int(np.clip(scales[-1] + 1, min_cap, max_cap))
            if cand not in scales:
                scales.append(cand)
            else:
                break
        scales = scales[:n_scales]

    else:
        raise ValueError("mode must be one of {'lin','log','hybrid','sqrt','power{p}','inv'}")

    upper_bound = max(2, min(max_cap, n_samples - 1))
    scales = [int(np.clip(s, 2, upper_bound)) for s in scales]
    scales = _unique_sorted_ints(scales)
    if len(scales) == 0:
        scales = [min(upper_bound, max(2, min_cap))]
    while len(scales) < n_scales and scales[-1] < upper_bound:
        nxt = scales[-1] + 1
        if nxt not in scales:
            scales.append(nxt)
        else:
            break
    return scales

# -------------------- Config parsing (now WITHOUT manual scales) --------------------
def parse_dataset_config_block(text):
    """
    Returns a list of dicts (one per run) with keys:
      name, out_dim, distance, subspace_dim, pca_dim, cluster_method,
      n_scales, mode, min_cap, max_cap
    """
    runs = []
    if not text:
        return runs
    for ln in text.splitlines():
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        cols = re.split(r'[ \t]+', s)
        if len(cols) < 10:
            raise ValueError(
                "Config line has too few columns (<10):\n"
                f"  {ln}\n"
                "Expected: name out_dim distance subspace_dim pca_dim cluster_method n_scales mode min_cap max_cap"
            )
        name           = cols[0]
        out_dim        = int(cols[1])
        distance       = cols[2].lower()
        subspace_dim   = cols[3]
        pca_dim        = int(cols[4])
        cluster_method = cols[5].lower()
        n_scales       = int(cols[6])
        mode           = cols[7].lower()
        min_cap        = parse_int_auto(cols[8])
        max_cap        = parse_int_auto(cols[9])

        if cluster_method not in ("spectral", "kmeans"):
            raise ValueError(f"cluster_method must be 'spectral' or 'kmeans', got: {cluster_method}")
        if distance not in ("geodesic", "chordal", "reflection"):
            raise ValueError(f"distance must be 'geodesic'|'chordal'|'reflection', got: {distance}")
        if mode not in ("lin", "log", "hybrid", "sqrt", "inv") and not mode.startswith("power"):
            raise ValueError(
            f"mode must be one of "
            "'lin' | 'log' | 'hybrid' | 'sqrt' | 'inv' | 'powerX' (e.g., power0.4), got: {mode}"
            )
        runs.append(dict(
            name=name, out_dim=out_dim, distance=distance, subspace_dim=subspace_dim,
            pca_dim=pca_dim, cluster_method=cluster_method,
            n_scales=n_scales, mode=mode, min_cap=min_cap, max_cap=max_cap
        ))
    return runs

# -------------------- PCA preprocessing --------------------
def pca_reduce(X, m=PCA_DIM_DEFAULT, standardize=True, random_state=RANDOM_STATE):
    """
    Reduce features X to m dimensions (optional z-score first, then PCA).
    Returns X_pca with shape (n_samples × m)
    """
    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    m = int(m)
    m = max(2, min(m, X.shape[1]))
    pca = PCA(n_components=m, random_state=random_state)
    Xp = pca.fit_transform(X)
    return Xp

# -------------------- UMAP embedding (run on PCA space) --------------------
def umap_embed(X, nn, out_dim, random_state, normalize=False):
    """
    Use your reduction_wrapper (preferred). Fallback to umap-learn if needed.
    """
    emb_try = reduction_wrapper(
        X, method='UMAP',
        secondary_param=int(nn),
        n_components=int(out_dim),
        normalize=normalize,
        random_state=random_state
    )
    emb = emb_try[0] if isinstance(emb_try, (tuple, list)) else emb_try
    if emb.shape[1] == out_dim:
        return emb
    import umap
    um = umap.UMAP(
        n_neighbors=int(nn),
        n_components=int(out_dim),
        random_state=int(random_state),
        metric="euclidean",
        min_dist=0.1,
    )
    return um.fit_transform(X)

# -------------------- Build and save projection stack (NO reduce_rank) --------------------
def build_and_save_projections(
    X_pca, scales, out_dim, random_state,
    save_dir, basename,
    standardize=True,
    save_float32=True,
    also_save_stack_npz=True,
    max_samples=None,
    pca_dim=PCA_DIM_DEFAULT
):
    """
    Note: X_pca is the PCA-reduced data. We directly use all scales (no column reduction).
    """
    ensure_dir(save_dir)
    n = X_pca.shape[0]
    if max_samples is not None:
        n = min(n, int(max_samples)); X_pca = X_pca[:n]

    embeds = []
    for nn in scales:
        E = umap_embed(X_pca, nn, out_dim, random_state, normalize=False)
        if standardize:
            E = StandardScaler().fit_transform(E)
        embeds.append(E)

    dtype_save = np.float32 if save_float32 else np.float64
    stack_memmap = None
    if also_save_stack_npz:
        stack_memmap = np.memmap(
            os.path.join(save_dir, f"{basename}_stack_tmp.dat"),
            mode="w+",
            dtype=dtype_save,
            shape=(n, out_dim, out_dim)
        )

    t0 = time.time()
    for i in range(n):
        V = np.column_stack([E[i, :] for E in embeds])
        Q, _ = np.linalg.qr(V, mode="reduced")
        P = Q @ Q.T

        if save_float32: P = P.astype(np.float32, copy=False)
        if also_save_stack_npz: stack_memmap[i, :, :] = P
        if (i+1) % 50 == 0 or (i+1) == n:
            print(f"[Projections] saved {i+1}/{n}  elapsed={time.time()-t0:.1f}s")

    if also_save_stack_npz:
        del stack_memmap
        stack_memmap = np.memmap(
            os.path.join(save_dir, f"{basename}_stack_tmp.dat"),
            mode="r",
            dtype=dtype_save,
            shape=(n, out_dim, out_dim)
        )
        np.savez_compressed(os.path.join(save_dir, f"{basename}_stack.npz"), P=stack_memmap)
        del stack_memmap
        try: os.remove(os.path.join(save_dir, f"{basename}_stack_tmp.dat"))
        except Exception: pass

    meta = dict(
        save_dir=save_dir, basename=basename,
        neighbors_list=list(scales), out_dim=out_dim,
        n_samples=n, dtype=("float32" if save_float32 else "float64"),
        standardize=standardize, stack_npz=also_save_stack_npz,
        pca_dim=int(pca_dim)
    )
    np.savez(os.path.join(save_dir, f"{basename}_meta.npz"), **meta)
    return meta

# -------------------- Load projection stack --------------------
def load_projections(save_dir, basename=None, stack_npz=None, dtype=np.float32):
    """
    Load P_stack (n × d × d) from saved files. Prefer *_stack.npz; fall back to per-sample .npy.
    """
    if stack_npz is not None and os.path.exists(stack_npz):
        data = np.load(stack_npz)
        return data["P"].astype(dtype, copy=False)
    pattern = os.path.join(save_dir, f"{basename}_sample*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No projection files found; looked for: {stack_npz or pattern}")
    P0 = np.load(files[0]).astype(dtype, copy=False)
    n, d = len(files), P0.shape[0]
    P = np.empty((n, d, d), dtype=dtype); P[0] = P0
    for i, f in enumerate(files[1:], start=1):
        P[i] = np.load(f).astype(dtype, copy=False)
    return P

# -------------------- Grassmann distance computation --------------------
def reflection_log_distance(P, Q, symmetrize=True):
    d = P.shape[0]
    if symmetrize:
        P = 0.5 * (P + P.T); Q = 0.5 * (Q + Q.T)
    I = np.eye(d, dtype=P.dtype)
    M = (I - 2.0*P) @ (I - 2.0*Q)
    L = eigvals(M)
    logs = np.log(L)
    return float(np.linalg.norm(logs))

def pairwise_reflection_log(P_stack, verbose=True):
    n = P_stack.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        Pi = P_stack[i]
        for j in range(i+1, n):
            D[i, j] = D[j, i] = reflection_log_distance(Pi, P_stack[j])
        if verbose and ((i+1) % 10 == 0 or (i+1) == n):
            print(f"[Distances] row {i+1}/{n}  elapsed={time.time()-t0:.1f}s")
    return D

def projection_to_basis(P, r=None, tol=1e-8):
    P = 0.5 * (P + P.T)
    vals, vecs = np.linalg.eigh(P)
    if r is None:
        r = int(np.sum(vals > tol))
        if r == 0: r = 1
    idx = np.argsort(vals)[-r:]
    Q = vecs[:, idx]
    Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q, r

def principal_angles_from_bases(Q1, Q2):
    r = min(Q1.shape[1], Q2.shape[1])
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s[:r], 0.0, 1.0)
    theta = np.arccos(s)
    return np.sort(theta)

def grassmann_distance_from_thetas(theta, mode="geodesic"):
    if mode == "geodesic":
        return float(np.linalg.norm(theta))
    elif mode == "chordal":
        return float(np.linalg.norm(np.sin(theta)))
    else:
        raise ValueError("mode must be 'geodesic' or 'chordal'")

def pairwise_principal_angle(P_stack, r=None, mode="geodesic", verbose=True):
    n, d, _ = P_stack.shape
    Q_list, r_list = [], []
    for i in range(n):
        Qi, ri = projection_to_basis(P_stack[i], r=r, tol=1e-8)
        Q_list.append(Qi); r_list.append(ri)
    r_use = int(max(1, min(r_list))) if r is None else int(r)
    for i in range(n):
        if Q_list[i].shape[1] > r_use:
            Q_list[i] = Q_list[i][:, :r_use]
    D = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        Qi = Q_list[i]
        for j in range(i+1, n):
            theta = principal_angles_from_bases(Qi, Q_list[j])
            if theta.shape[0] > r_use: theta = theta[:r_use]
            D[i, j] = D[j, i] = grassmann_distance_from_thetas(theta, mode=mode)
        if verbose and ((i+1) % 10 == 0 or (i+1) == n):
            print(f"[Distances-PA] row {i+1}/{n}  elapsed={time.time()-t0:.1f}s")
    return D

# -------------------- Clustering helpers --------------------
def spectral_cluster_from_dist(D, n_clusters, gamma=1.0, random_state=1):
    """Spectral clustering on a precomputed distance matrix via RBF affinity."""
    D2 = (D**2).astype(np.float64, copy=False)
    pos = D2[D2 > 0]
    sigma2 = float(np.median(pos)) if pos.size > 0 else 1.0
    if not np.isfinite(sigma2) or sigma2 <= 0: sigma2 = 1.0
    A = np.exp(-D2 / (gamma * sigma2))
    sp = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        assign_labels='kmeans', random_state=random_state, n_init=10
    )
    labels = sp.fit_predict(A)
    return labels, sigma2

def classical_mds_from_dist(D, target_dim=20):
    """
    Classical multidimensional scaling (Torgerson). Inputs: distance matrix D.
    Returns an embedding Z (n × target_dim).
    """
    n = D.shape[0]
    target_dim = int(max(1, min(target_dim, n-1)))
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    pos = w > 1e-12
    if not np.any(pos):
        pos = idx[:target_dim]
    w_pos = w[pos][:target_dim]
    V_pos = V[:, pos][:, :target_dim]
    Z = V_pos * np.sqrt(np.maximum(w_pos, 0.0))
    return Z

def kmeans_from_dist(D, n_clusters, random_state=1):
    """K-Means on classical MDS embedding of D."""
    Z = classical_mds_from_dist(D, target_dim=min(20, D.shape[0]-1))
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(Z)
    return labels

# -------------------- Baselines --------------------
def baseline_umap_compare(X_pca, y, scales, out_dim, random_state, results_dir, tag_prefix,
                          cluster_method="spectral", gamma=1.0):
    """
    cluster_method: 'spectral' (distance-based) or 'kmeans' (on UMAP embedding)
    现在只计算 avg-purity
    """
    n_clusters = np.unique(y).size
    rows = []
    print(f"\n[Baseline] PCA→UMAP -> {out_dim}D -> {cluster_method} (per n_neighbors):")
    for nn in scales:
        emb = umap_embed(X_pca, nn=nn, out_dim=out_dim, random_state=random_state, normalize=False)
        emb = StandardScaler(with_mean=True, with_std=True).fit_transform(emb)

        if cluster_method == "kmeans":
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
            y_pred = km.fit_predict(emb)
        else:
            D = pairwise_distances(emb, metric="euclidean")
            y_pred, _ = spectral_cluster_from_dist(D, n_clusters=n_clusters,
                                                   gamma=gamma, random_state=random_state)

        avgp = avg_purity_score(y, y_pred)
        print(f"  nn={nn:>3d}  Avg-Purity={avgp:.4f}")
        rows.append((nn, avgp))

    arr = np.array([a for (_, a) in rows], dtype=float)
    avgp_m = float(arr.mean())
    print(f"\n[Baseline Avg] over nn={list(scales)}  Avg-Purity={avgp_m:.4f}\n")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag_prefix}_baseline_pca_umap{out_dim}d_{cluster_method}.csv")
    with open(csv_path, "w") as f:
        f.write("nn,Avg-Purity\n")
        for nn, avgp in rows:
            f.write(f"{nn},{avgp:.6f}\n")
        f.write(f"avg,{avgp_m:.6f}\n")

    json_path = os.path.join(results_dir, f"{tag_prefix}_baseline_pca_umap{out_dim}d_{cluster_method}.json")
    payload = {
        "scales": list(scales), "out_dim": int(out_dim), "random_state": int(random_state),
        "cluster_method": cluster_method, "gamma": gamma,
        "per_scale": [{"nn": int(nn), "Avg-Purity": a} for (nn, a) in rows],
        "average": {"Avg-Purity": avgp_m},
    }
    save_json(payload, json_path)
    return {"Avg-Purity": avgp_m}

def baseline_pca_compare(X_pca, y, results_dir, tag_prefix, cluster_method="spectral", gamma=1.0, random_state=RANDOM_STATE):
    """
    cluster_method: 'spectral' (on distances in the PCA space) or 'kmeans' (on PCA features)
    现在只计算 avg-purity
    """
    n_clusters = np.unique(y).size
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(X_pca)

    if cluster_method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        y_pred = km.fit_predict(Z)
    else:
        D = pairwise_distances(Z, metric="euclidean")
        y_pred, _ = spectral_cluster_from_dist(D, n_clusters=n_clusters,
                                               gamma=gamma, random_state=random_state)

    avgp = avg_purity_score(y, y_pred)
    print(f"[Baseline-PCA@{Z.shape[1]}-{cluster_method}] Avg-Purity={avgp:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{tag_prefix}_baseline_pca_only_{cluster_method}.csv"), "w") as f:
        f.write("dim,Avg-Purity\n")
        f.write(f"{Z.shape[1]},{avgp:.6f}\n")
    save_json({"dim": int(Z.shape[1]), "Avg-Purity": avgp, "cluster_method": cluster_method},
              os.path.join(results_dir, f"{tag_prefix}_baseline_pca_only_{cluster_method}.json"))
    return {"Avg-Purity": avgp}

def baseline_pca_to_outdim_compare(X_pca, y, out_dim, results_dir, tag_prefix,
                                   cluster_method="spectral", gamma=1.0, random_state=RANDOM_STATE):
    """
    cluster_method: 'spectral' (on distances in the reduced PCA space) or 'kmeans'
    现在只计算 avg-purity
    """
    n_clusters = np.unique(y).size
    Z0 = StandardScaler(with_mean=True, with_std=True).fit_transform(X_pca)
    out_dim = int(out_dim)
    out_dim = max(2, min(out_dim, Z0.shape[1]))
    Z = PCA(n_components=out_dim, random_state=random_state).fit_transform(Z0)
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(Z)

    if cluster_method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        y_pred = km.fit_predict(Z)
    else:
        D = pairwise_distances(Z, metric="euclidean")
        y_pred, _ = spectral_cluster_from_dist(D, n_clusters=n_clusters,
                                               gamma=gamma, random_state=random_state)

    avgp = avg_purity_score(y, y_pred)
    print(f"[Baseline-PCA→PCA{out_dim}-{cluster_method}] Avg-Purity={avgp:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{tag_prefix}_baseline_pca_to_{out_dim}_{cluster_method}.csv"), "w") as f:
        f.write("dim,Avg-Purity\n")
        f.write(f"{out_dim},{avgp:.6f}\n")
    save_json({"dim": int(out_dim), "Avg-Purity": avgp, "cluster_method": cluster_method},
              os.path.join(results_dir, f"{tag_prefix}_baseline_pca_to_{out_dim}_{cluster_method}.json"))
    return {"Avg-Purity": avgp}

# -------------------- NEW: RAW X → PCA(out_dim) baseline --------------------
def baseline_pca_direct_to_outdim_compare(X, y, out_dim, results_dir, tag_prefix,
                                          cluster_method="spectral", gamma=1.0, random_state=RANDOM_STATE):
    """
    Directly standardize raw X, PCA to out_dim, then clustering (bypass pca_dim stage).
    现在只计算 avg-purity
    """
    n_clusters = np.unique(y).size

    Z0 = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    out_dim = int(out_dim)
    out_dim = max(2, min(out_dim, Z0.shape[1]))
    Z = PCA(n_components=out_dim, random_state=random_state).fit_transform(Z0)
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(Z)

    if cluster_method == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        y_pred = km.fit_predict(Z)
    else:
        D = pairwise_distances(Z, metric="euclidean")
        y_pred, _ = spectral_cluster_from_dist(D, n_clusters=n_clusters,
                                               gamma=gamma, random_state=random_state)

    avgp = avg_purity_score(y, y_pred)
    print(f"[Baseline-RAW-PCA→{out_dim}-{cluster_method}] Avg-Purity={avgp:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{tag_prefix}_baseline_raw_pca_to_{out_dim}_{cluster_method}.csv"), "w") as f:
        f.write("dim,Avg-Purity\n")
        f.write(f"{out_dim},{avgp:.6f}\n")

    save_json(
        {"dim": int(out_dim), "Avg-Purity": avgp, "cluster_method": cluster_method},
        os.path.join(results_dir, f"{tag_prefix}_baseline_raw_pca_to_{out_dim}_{cluster_method}.json")
    )
    return {"Avg-Purity": avgp}

# -------------------- Single dataset pipeline (single seed) --------------------
def run_one(data_name, out_dim, distance, subspace_dim,
            pca_dim=PCA_DIM_DEFAULT, cluster_method="spectral",
            n_scales=12, mode="lin", min_cap=None, max_cap=None,
            seed=RANDOM_STATE):
    """
    Executes a single run (single seed):
      - Generate scales
      - PCA → multi-scale UMAP → projection stack → Grassmann distances
      - Clustering (spectral/kmeans)
      - Baselines (incl. RAW→PCA(out_dim))
      - Returns a summary dict (one row) for THIS seed
    """
    np.random.seed(seed)
    print(f"\n========== [Load] data={data_name} (seed={seed}) ==========")
    X = load_X(data_name, DATA_PATH, DATA_PROCESS_PATH)
    y = load_y(data_name, DATA_PATH, DATA_PROCESS_PATH)
    if data_name != 'GSE57249':
        X, y = preprocess_data(X, y)
    else:
        X = np.log10(1 + X).T

    if (MAX_SAMPLES is not None) and (X.shape[0] > MAX_SAMPLES):
        X = X[:MAX_SAMPLES]; y = y[:MAX_SAMPLES]
        print(f"[Subsample] using first {X.shape[0]} samples")

    auto_scales = generate_scales(
        n_samples=X.shape[0],
        n_scales=n_scales,
        mode=mode,
        min_cap=min_cap,
        max_cap=max_cap
    )
    print(f"[Scales] mode={mode} n_scales={n_scales} range=({min_cap},{max_cap}) -> {auto_scales}")

    print(f"[PCA] reducing to m={pca_dim} ...")
    X_pca = pca_reduce(X, m=pca_dim, standardize=True, random_state=seed)

    basename = (
        f"{data_name}_PCA{int(pca_dim)}_GrassProj_UMAP{int(out_dim)}"
        f"_AUTO_S{len(auto_scales)}_{mode}_"
        f"R{auto_scales[0]}-{auto_scales[-1]}_{cluster_method}_S{seed}"
    )

    if not SKIP_PROJ:
        print(f"[Build Projections] scales={auto_scales} UMAP_out_dim={out_dim} -> {SAVE_DIR}")
        meta = build_and_save_projections(
            X_pca, scales=auto_scales, out_dim=out_dim, random_state=seed,
            save_dir=SAVE_DIR, basename=basename,
            standardize=STANDARDIZE_UMAP,
            save_float32=(not USE_FLOAT64),
            also_save_stack_npz=SAVE_STACK,
            max_samples=MAX_SAMPLES,
            pca_dim=pca_dim
        )
        meta["generated_scales"] = list(map(int, auto_scales))
        meta["scale_mode"] = mode
        meta["n_scales"] = int(n_scales)
        meta["min_cap"] = (None if min_cap is None else int(min_cap))
        meta["max_cap"] = (None if max_cap is None else int(max_cap))
        save_json(meta, os.path.join(RESULTS_DIR, f"{basename}_meta.json"))
    else:
        print("[Skip] projection building; using existing files")

    P_stack = load_projections(
        save_dir=SAVE_DIR, basename=basename,
        stack_npz=os.path.join(SAVE_DIR, f"{basename}_stack.npz") if SAVE_STACK else None,
        dtype=(np.float32 if (not USE_FLOAT64) else np.float64)
    )
    n = P_stack.shape[0]
    print(f"[Projections] loaded P with shape={P_stack.shape}")

    sub_dim = parse_int_auto(subspace_dim) if isinstance(subspace_dim, str) else subspace_dim
    if distance == "reflection":
        print("[Distances] reflection-log ...")
        D = pairwise_reflection_log(P_stack, verbose=True)
    elif distance == "geodesic":
        print("[Distances] principal angles (geodesic) ...")
        D = pairwise_principal_angle(P_stack, r=sub_dim, mode="geodesic", verbose=True)
    elif distance == "chordal":
        print("[Distances] principal angles (chordal) ...")
        D = pairwise_principal_angle(P_stack, r=sub_dim, mode="chordal", verbose=True)
    else:
        raise ValueError("distance must be one of {'reflection','geodesic','chordal'}")

    np.save(os.path.join(RESULTS_DIR, f"{basename}_D.npy"), D)

    n_clusters = np.unique(y).size
    if cluster_method == "kmeans":
        labels = kmeans_from_dist(D, n_clusters=n_clusters, random_state=seed)
    else:
        labels, _ = spectral_cluster_from_dist(D, n_clusters=n_clusters,
                                               gamma=GAMMA, random_state=seed)
    np.save(os.path.join(RESULTS_DIR, f"{basename}_labels.npy"), labels)

    # ====== 只算 avg-purity ======
    g_avgp = avg_purity_score(y, labels)
    print(f"[Grassmann-{cluster_method}] Avg-Purity={g_avgp:.4f}")

    # --- Baselines (同一 seed) ---
    b_umap = baseline_umap_compare(
        X_pca, y, scales=auto_scales, out_dim=out_dim,
        random_state=seed,
        results_dir=RESULTS_DIR,
        tag_prefix=basename,
        cluster_method=cluster_method,
        gamma=GAMMA
    )
    b_pca = baseline_pca_compare(
        X_pca, y, results_dir=RESULTS_DIR, tag_prefix=basename,
        cluster_method=cluster_method, gamma=GAMMA, random_state=seed
    )
    b_pca_out = baseline_pca_to_outdim_compare(
        X_pca, y, out_dim=out_dim, results_dir=RESULTS_DIR, tag_prefix=basename,
        cluster_method=cluster_method, gamma=GAMMA, random_state=seed
    )
    b_pca_raw_out = baseline_pca_direct_to_outdim_compare(
        X, y, out_dim=out_dim, results_dir=RESULTS_DIR, tag_prefix=basename,
        cluster_method=cluster_method, gamma=GAMMA, random_state=seed
    )

   
    row = OrderedDict()
    row["data"] = data_name
    row["n"] = int(n)
    row["pca_dim"] = int(pca_dim)
    row["scales"] = ",".join(map(str, auto_scales))
    row["out_dim"] = int(out_dim)
    row["distance"] = distance
    row["subspace_dim"] = ("auto" if sub_dim is None else int(sub_dim))
    row["gamma"] = float(GAMMA)
    row["cluster_method"] = cluster_method
    row["scale_mode"] = mode
    row["n_scales"] = int(n_scales)
    row["min_cap"] = "" if min_cap is None else int(min_cap)
    row["max_cap"] = "" if max_cap is None else int(max_cap)
    row["seed"] = int(seed)

    # Grassmann
    row["G_AvgPurity"] = g_avgp
    # UMAP (avg across scales)
    row["B_AvgPurity"] = float(b_umap["Avg-Purity"])
    # PCA@pca_dim
    row["PCA_AvgPurity"] = float(b_pca["Avg-Purity"])
    # PCA→PCA(out_dim)
    row["PCAout_AvgPurity"] = float(b_pca_out["Avg-Purity"])
    # RAW→PCA(out_dim) [NEW]
    row["PCAraw_AvgPurity"] = float(b_pca_raw_out["Avg-Purity"])

    return row

# -------------------- Pretty-print the summary table --------------------
def print_summary_table(rows, ordered_keys):
    colw = {k: max(len(k), max(len(str(r.get(k,""))) for r in rows)) for k in ordered_keys}
    def fmt_row(r):
        return " | ".join(str(r.get(k,"")).ljust(colw[k]) for k in ordered_keys)
    sep = "-+-".join("-"*colw[k] for k in ordered_keys)
    print("\n=== SUMMARY ===")
    print(" | ".join(k.ljust(colw[k]) for k in ordered_keys))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()

# -------------------- Main --------------------
def main():
    ensure_dir(SAVE_DIR); ensure_dir(RESULTS_DIR)

    run_cfgs = parse_dataset_config_block(DATASET_CONFIG)
    if not run_cfgs:
        raise SystemExit(
            "DATASET_CONFIG is empty. Please add lines:\n"
            "name out_dim distance subspace_dim pca_dim cluster_method n_scales mode min_cap max_cap"
        )

    aggregated_rows = [] 

    for cfg in run_cfgs:
        per_seed_rows = []
        for seed in SEEDS:
            try:
                row = run_one(
                    data_name=cfg["name"],
                    out_dim=int(cfg["out_dim"]),
                    distance=str(cfg["distance"]).lower(),
                    subspace_dim=cfg["subspace_dim"],
                    pca_dim=int(cfg.get("pca_dim", PCA_DIM_DEFAULT)),
                    cluster_method=str(cfg.get("cluster_method", "spectral")).lower(),
                    n_scales=int(cfg.get("n_scales", 12)),
                    mode=str(cfg.get("mode", "lin")).lower(),
                    min_cap=cfg.get("min_cap", None),
                    max_cap=cfg.get("max_cap", None),
                    seed=seed
                )
                per_seed_rows.append(row)
            except Exception as e:
                print(f"[Error] dataset={cfg['name']} seed={seed}: {repr(e)}")
                per_seed_rows.append(OrderedDict(data=cfg["name"], seed=seed, error=str(e)))

   
        rows_ok = [r for r in per_seed_rows if "error" not in r]
        if not rows_ok:
            aggregated_rows.append(OrderedDict(data=cfg["name"], error="all seeds failed"))
            continue

        base = rows_ok[0].copy()
        base.pop("seed", None)
        base["seeds"] = ",".join(str(r["seed"]) for r in rows_ok)

       
        metric_keys = [
            "G_AvgPurity",
            "B_AvgPurity",
            "PCA_AvgPurity",
            "PCAout_AvgPurity",
            "PCAraw_AvgPurity",
        ]
        for k in metric_keys:
            vals = [r.get(k, np.nan) for r in rows_ok]
            base[k] = float(np.nanmean(vals))

        aggregated_rows.append(base)

    # Save one combined CSV + Excel with ONLY means
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"SUMMARY_all_means_{ts}.csv")
    xlsx_path = os.path.join(RESULTS_DIR, f"SUMMARY_all_means_{ts}.xlsx")

    preferred = [
        "data","n","pca_dim","scales","out_dim","distance","subspace_dim",
        "gamma","cluster_method","scale_mode","n_scales","min_cap","max_cap",
        "seeds",
        "G_AvgPurity",
        "B_AvgPurity",
        "PCA_AvgPurity",
        "PCAout_AvgPurity",
        "PCAraw_AvgPurity",
        "error"
    ]
    all_keys = OrderedDict()
    for r in aggregated_rows:
        for k in r.keys(): all_keys[k] = True
    ordered_keys = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]

    # Write CSV (only means)
    with open(csv_path, "w") as f:
        f.write(",".join(ordered_keys) + "\n")
        for r in aggregated_rows:
            vals = []
            for k in ordered_keys:
                v = r.get(k, "")
                if isinstance(v, str) and ("," in v or " " in v):
                    v = f"\"{v}\""
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

    # Write Excel (only means)
    df = pd.DataFrame([{k: row.get(k, "") for k in ordered_keys} for row in aggregated_rows])
    wrote_xlsx = False
    for engine in ("xlsxwriter", "openpyxl"):
        try:
            with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name="summary_means")
            wrote_xlsx = True
            break
        except Exception as e:
            print(f"[Warn] Excel engine '{engine}' not available or failed: {e}")
            continue

    # 打印 summary（only means）
    print_summary_table(aggregated_rows, ordered_keys)
    print(f"[Summary Means] wrote:\n  {csv_path}")
    if wrote_xlsx:
        print(f"  {xlsx_path}")
    else:
        print("  (Excel not written; install 'xlsxwriter' or 'openpyxl' to enable .xlsx export)")

if __name__ == "__main__":
    main()
