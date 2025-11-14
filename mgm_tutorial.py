#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mgm_tutorial_single_dataset.py

Tutorial: single-cell RNA-seq → PCA → multi-scale UMAP → Grassmann projections
→ Grassmann distance matrix.

This script shows how to:
  1. Load one dataset (here: GSE75748time)
  2. Run PCA
  3. Choose a fixed list of UMAP neighbor sizes (scales)
  4. Build multiscale Grassmann projections
  5. Compute Grassmann distances between cells

It reuses the core functions from main_MGM.py with minimal changes.
"""

import os
import numpy as np

from algorithm.auxilary import load_X, load_y, preprocess_data
from main_MGM import (
    pca_reduce,
    umap_embed,
    build_and_save_projections,
    load_projections,
    pairwise_principal_angle,
)

# -------------------- Local paths (no import from main_MGM) --------------------

# You can change these paths if your data is somewhere else
DATA_PROCESS_PATH = "./SingleCellDataProcess/"
DATA_PATH = "./data/"

# -------------------- Tutorial configuration --------------------

# 1. Choose ONE dataset
DATA_NAME = "GSE75748time"   # you can change to another dataset name

# 2. PCA dimension before UMAP
PCA_DIM = 200

# 3. UMAP output dimension (the dimension of each UMAP embedding)
UMAP_OUT_DIM = 100

# 4. Grassmann distance type and subspace dimension
#    "geodesic" | "chordal" correspond to two distances from principal angles
DISTANCE_MODE = "chordal"   # or "geodesic"
SUBSPACE_DIM = 23           # consistent with your main_MGM settings

# 5. A fixed list of UMAP neighbor sizes (this is the "multiscale" part)
NEIGHBOR_LIST = [5, 10, 20, 40, 80]

# 6. Random seed
RANDOM_STATE = 1

# 7. Where to save projections and distances
SAVE_DIR = "./tutorial_projections"
RESULTS_DIR = "./tutorial_results"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def tutorial_run():
    # ------------------------------------------------------------
    # Step 1: Load and preprocess single-cell data
    # ------------------------------------------------------------
    print(f"\n========== Tutorial: data = {DATA_NAME} ==========")
    X = load_X(DATA_NAME, DATA_PATH, DATA_PROCESS_PATH)
    y = load_y(DATA_NAME, DATA_PATH, DATA_PROCESS_PATH)

    if DATA_NAME != "GSE57249":
        X, y = preprocess_data(X, y)
    else:
        # same special case as in main_MGM.py
        X = np.log10(1 + X).T

    n_cells, n_genes = X.shape
    print(f"[Data] X shape = {X.shape}, n_cells = {n_cells}, n_genes = {n_genes}")
    print(f"[Data] y shape = {y.shape}, unique labels = {np.unique(y).size}")

    # ------------------------------------------------------------
    # Step 2: PCA
    # ------------------------------------------------------------
    print(f"\n[Step 2] PCA to {PCA_DIM} dimensions ...")
    X_pca = pca_reduce(X, m=PCA_DIM, standardize=True, random_state=RANDOM_STATE)
    print(f"[PCA] X_pca shape = {X_pca.shape}")

    # ------------------------------------------------------------
    # Step 3: Build multiscale UMAP embeddings and Grassmann projections
    # ------------------------------------------------------------
    print(f"\n[Step 3] Multiscale UMAP + projection stack")
    print(f"        UMAP_out_dim = {UMAP_OUT_DIM}")
    print(f"        Neighbor list (scales) = {NEIGHBOR_LIST}")

    basename = (
        f"{DATA_NAME}_TUTORIAL_PCA{int(PCA_DIM)}_UMAP{int(UMAP_OUT_DIM)}"
        f"_S{len(NEIGHBOR_LIST)}_R{NEIGHBOR_LIST[0]}-{NEIGHBOR_LIST[-1]}"
    )

    meta = build_and_save_projections(
        X_pca,
        scales=NEIGHBOR_LIST,
        out_dim=UMAP_OUT_DIM,
        random_state=RANDOM_STATE,
        save_dir=SAVE_DIR,
        basename=basename,
        standardize=True,
        save_float32=True,
        also_save_stack_npz=True,
        max_samples=None,
        pca_dim=PCA_DIM,
    )

    print("\n[Meta info]")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------
    # Step 4: Load Grassmann projection stack P (n × d × d)
    # ------------------------------------------------------------
    print(f"\n[Step 4] Load projection stack from {SAVE_DIR}")
    stack_path = os.path.join(SAVE_DIR, f"{basename}_stack.npz")
    P_stack = load_projections(
        save_dir=SAVE_DIR,
        basename=basename,
        stack_npz=stack_path,
        dtype=np.float32,
    )
    print(f"[Projections] P_stack shape = {P_stack.shape}  (n_cells × d × d)")

    # ------------------------------------------------------------
    # Step 5: Compute Grassmann distance matrix
    # ------------------------------------------------------------
    print(f"\n[Step 5] Compute Grassmann distances: mode = {DISTANCE_MODE}")
    if DISTANCE_MODE not in ("geodesic", "chordal"):
        raise ValueError("DISTANCE_MODE must be 'geodesic' or 'chordal'")

    D = pairwise_principal_angle(
        P_stack,
        r=SUBSPACE_DIM,
        mode=DISTANCE_MODE,
        verbose=True,
    )
    print(f"[Distances] D shape = {D.shape} (n_cells × n_cells)")

    # Save distance matrix for downstream OT
    dist_path = os.path.join(RESULTS_DIR, f"{basename}_D_{DISTANCE_MODE}.npy")
    np.save(dist_path, D)
    print(f"[Save] Grassmann distance matrix saved to: {dist_path}")

    # ------------------------------------------------------------
    # Step 6 (optional): show a few entries for sanity check
    # ------------------------------------------------------------
    print("\n[Check] First 5 distances among first 5 cells:")
    print(D[:5, :5])

    print("\nTutorial finished.")
    print("You now have:")
    print(f"  * Multiscale Grassmann projections: {stack_path}")
    print(f"  * Grassmann distance matrix D:      {dist_path}")
    print("You can feed D into OT / trajectory analysis.")


if __name__ == "__main__":
    tutorial_run()