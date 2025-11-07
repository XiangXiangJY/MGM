#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_grumap_vs_umap.py

Rows = datasets; columns = methods:
[ Gr-UMAP | PCA→UMAP | Raw→UMAP ]

Bigger fonts everywhere, clear column headers and row labels.
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# ---------------- Paths ----------------
DATA_PROCESS_PATH = "./SingleCellDataProcess/"
DATA_PATH         = "./data/"
RESULTS_DIR       = "./results"
FIG_DIR           = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# --------- Project loaders/wrapper ------
from algorithm.auxilary import load_X, load_y, preprocess_data
from algorithm.reduction import reduction_wrapper

# --------- Global styling (bigger fonts) ---------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 22,          # base
    "axes.titlesize": 22,     # column headers
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

MARKER_SIZE = 12   # was 6
ALPHA       = 1

# ===================== helpers =====================
def pca_reduce(X, m=50, seed=1):
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    m = int(max(2, min(m, X.shape[1])))
    return PCA(n_components=m, random_state=seed).fit_transform(X)

def generate_scales(n_samples, n_scales=11, mode="power1.6", min_cap=5, max_cap=20):
    p = 1.6
    m = re.search(r"(power|pow)\s*([0-9]*\.?[0-9]+)", str(mode).lower())
    if m: p = float(m.group(2))
    min_cap = int(min_cap); max_cap = int(max_cap)
    t = np.linspace(0.0, 1.0, n_scales)
    vals = min_cap + (max_cap - min_cap) * (t ** p)
    scales = sorted(set(int(round(v)) for v in vals))
    scales = [s for s in scales if 2 <= s < min(max_cap, n_samples-1)]
    while len(scales) < n_scales and scales[-1] < max_cap:
        nxt = scales[-1] + 1
        if nxt not in scales: scales.append(nxt)
        else: break
    return scales

def umap_embed(X, nn, out_dim, seed):
    # try your wrapper first; fallback to umap-learn
    E_try = reduction_wrapper(
        X, method='UMAP',
        secondary_param=int(nn),
        n_components=int(out_dim),
        normalize=False,
        random_state=seed
    )
    E = E_try[0] if isinstance(E_try, (tuple, list)) else E_try
    if E.shape[1] == out_dim: return E
    return umap.UMAP(n_neighbors=int(nn), n_components=int(out_dim),
                     random_state=seed, metric="euclidean").fit_transform(X)

def build_projection_stack(X_pca, scales, out_dim, seed):
    n = X_pca.shape[0]
    embeds = []
    for nn in scales:
        E = umap_embed(X_pca, nn, out_dim, seed)
        E = StandardScaler().fit_transform(E)
        embeds.append(E)
    P = np.zeros((n, out_dim, out_dim), dtype=np.float32)
    for i in range(n):
        V = np.column_stack([E[i, :] for E in embeds])  # (out_dim, n_scales)
        Q, _ = np.linalg.qr(V, mode="reduced")
        P[i] = Q @ Q.T
    return P

def projection_to_basis(P, r=None, tol=1e-8):
    P = 0.5*(P + P.T)
    vals, vecs = np.linalg.eigh(P)
    if r is None:
        r = int(np.sum(vals > tol)) or 1
    idx = np.argsort(vals)[-r:]
    Q = vecs[:, idx]
    Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q, r

def principal_angles(Q1, Q2):
    r = min(Q1.shape[1], Q2.shape[1])
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)[:r]
    s = np.clip(s, 0.0, 1.0)
    return np.sort(np.arccos(s))

def chordal_from_thetas(theta):
    return float(np.linalg.norm(np.sin(theta)))

def pairwise_chordal(P_stack, r=None):
    n = P_stack.shape[0]
    Qs, rs = [], []
    for i in range(n):
        Qi, ri = projection_to_basis(P_stack[i], r=r)
        Qs.append(Qi); rs.append(ri)
    r_use = r if r is not None else int(max(1, min(rs)))
    for i in range(n):
        if Qs[i].shape[1] > r_use:
            Qs[i] = Qs[i][:, :r_use]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            theta = principal_angles(Qs[i], Qs[j])
            if theta.shape[0] > r_use: theta = theta[:r_use]
            d = chordal_from_thetas(theta)
            D[i, j] = D[j, i] = d
    return D

# ===================== compute views =====================
def compute_views(dataset, pca_dim=50, out_dim=20, n_scales=11,
                  mode="power1.6", min_cap=5, max_cap=20,
                  subspace_dim="auto", seed=1):
    """
    Returns Zg (Gr-UMAP), Ze (PCA→UMAP), Zu (Raw→UMAP), y
    """
    X = load_X(dataset, DATA_PATH, DATA_PROCESS_PATH)
    y = load_y(dataset, DATA_PATH, DATA_PROCESS_PATH)
    if dataset != 'GSE57249':
        X, y = preprocess_data(X, y)
    else:
        X = np.log10(1 + X).T

    # PCA space
    Xp = pca_reduce(X, m=pca_dim, seed=seed)

    # Scales + Grassmann stack
    scales  = generate_scales(Xp.shape[0], n_scales=n_scales, mode=mode,
                              min_cap=min_cap, max_cap=max_cap)
    P_stack = build_projection_stack(Xp, scales, out_dim, seed=seed)

    # Chordal distances
    r = None if str(subspace_dim).lower() == "auto" else int(subspace_dim)
    D = pairwise_chordal(P_stack, r=r)

    # Gr-UMAP (metric='precomputed')
    Zg = umap.UMAP(n_neighbors=15, n_components=2,
                   metric="precomputed", random_state=seed).fit_transform(D)

    # PCA → UMAP
    Ze = umap.UMAP(n_neighbors=max(10, scales[len(scales)//2])),
    Ze = umap.UMAP(n_neighbors=max(10, scales[len(scales)//2]),
                   n_components=2, metric="euclidean",
                   random_state=seed).fit_transform(
         StandardScaler().fit_transform(Xp)
    )

    # Raw → UMAP
    X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    nn_raw = max(10, scales[len(scales)//2])
    Zu = umap.UMAP(n_neighbors=nn_raw, n_components=2, metric="euclidean",
                   random_state=seed).fit_transform(X_std)

    return Zg, Ze, Zu, y

# ===================== panel figure =====================
def panel_figure(results, dataset_names, save_path_png, save_path_pdf):
    """
    results: list of (Zg, Ze, Zu, y) per dataset
    dataset_names: list of names (for row labels)
    Layout: rows = datasets, columns = [Gr-UMAP, PCA→UMAP, Raw→UMAP]
    """
    n_rows = len(results)
    n_cols = 3
    # bigger canvas to accommodate larger labels
    fig_w = 15
    fig_h = 4.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Column headers (methods)
    col_titles = ["Gr-UMAP", "PCA→UMAP", "UMAP"]
    for c in range(n_cols):
        axes[0, c].set_title(col_titles[c], pad=8)  # pad adds space below title

    for r, (Zg, Ze, Zu, y) in enumerate(results):
        # leftmost y-axis as dataset row label
        axes[r, 0].set_ylabel(dataset_names[r], rotation=90, labelpad=18, va="center")

        # plot each method
        for c, Z in enumerate([Zg, Ze, Zu]):
            ax = axes[r, c]
            ax.scatter(Z[:, 0], Z[:, 1],
                       c=y, s=MARKER_SIZE, alpha=ALPHA, linewidths=0, cmap="tab10")
            ax.set_xticks([]); ax.set_yticks([])
            # thin frame for neat grid
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path_png, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")
    plt.close(fig)

# ===================== main =====================
def run_one(dataset, pca_dim=50, out_dim=20, n_scales=11,
            mode="power1.6", min_cap=5, max_cap=20,
            subspace_dim="auto", seed=1):
    print(f"\n=== [{dataset}] seed={seed} ===")
    Zg, Ze, Zu, y = compute_views(
        dataset, pca_dim, out_dim, n_scales, mode, min_cap, max_cap, subspace_dim, seed
    )
    # also save per-method single images if you like (optional)
    return Zg, Ze, Zu, y

if __name__ == "__main__":
    runs = [
        ("GSE67835",           50, 20, 11, "power1.6", 5, 20, "10", 1),
        ("GSE75748time",       50, 20, 11, "power1.6", 5, 20, "10", 1),
        ("GSE109979",  50, 20, 11, "power1.6", 5, 20, "10", 1),
        ("GSE75748cell",       50, 20, 11, "power1.6", 5, 20, "10", 1),
        ("GSE94820",           50, 20, 11, "power1.6", 5, 20, "10", 1),
        # ("GSE57249",         20, 15, 10, "power1.6", 5, 15, "auto", 1),
    ]

    results, names = [], []
    for (name, pca_dim, out_dim, n_scales, mode, kmin, kmax, sub_d, seed) in runs:
        Zg, Ze, Zu, y = run_one(name, pca_dim, out_dim, n_scales, mode, kmin, kmax, sub_d, seed)
        results.append((Zg, Ze, Zu, y))
        names.append(name)

    panel_png = os.path.join(FIG_DIR, "PANEL_GrUMAP_vs_PCAUMAP_vs_RAWUMAP1.png")
    panel_pdf = os.path.join(FIG_DIR, "PANEL_GrUMAP_vs_PCAUMAP_vs_RAWUMAP1.pdf")
    panel_figure(results, names, panel_png, panel_pdf)
    print(f"[panel saved] {panel_png} / {panel_pdf}")