# Multiscale Grassmanns Manifold (MGM)

This repository contains the implementation of the **Multiscale Grassmann Manifolds (MGM)** framework for Single-Cell Data Analysis.

## Overview
The MGM framework integrates multiscale low-dimensional embeddings (e.g., UMAP) on the Grassmann manifold to produce stable and discriminative subspace representations.  
Clustering evaluation metrics include **Accuracy (ACC)**, **Normalized Mutual Information (NMI)**, **Adjusted Rand Index (ARI)**, **Purity** and **average Purity**.

---

## Repository Structure

| File / Folder | Description |
|----------------|-------------|
| `algorithm/` | Contains basic functions and utilities required for dimensionality reduction and Grassmann manifold operations. The code in this folder is adapted from [hozumiyu/CCP-scRNAseq-UMAP-TSNE](https://github.com/hozumiyu/CCP-scRNAseq-UMAP-TSNE). |
| `plotting_code/` | Includes scripts for generating performance plots and visualizations. |
| `main_MGM.py` | Main script for running MGM with clustering evaluation using four metrics: **ACC**, **NMI**, **ARI**, and **Purity**. |
| `main_MGMavgpurity.py` | Main script including the **average Purity** metric computation. |
| `hpcc_projenv.yml` | Environment configuration file used for running experiments on the Michigan State University (MSU) HPCC system. |
| `requirement.txt` | Lists the Python package versions used in this project. |

---

## Environment Setup

You can reproduce the environment using either `conda` or `pip`:

### Using Conda
```bash
conda env create -f hpcc_projenv.yml
conda activate MGM
