import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX text rendering and Computer Modern font to match your paper
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9,           # match your LaTeX body text
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# -----------------------------
# 1) Data (Setup I)
# -----------------------------
data = {
    "ACC": {
        "Gr-UMAP":      [0.7219, 0.7314, 0.6819, 0.8316, 0.7732, 0.6920, 0.7053, 0.7677, 0.9102],
        "Average-UMAP": [0.6599, 0.7391, 0.4970, 0.7568, 0.7009, 0.6599, 0.5972, 0.7323, 0.8507],
        "PCA":          [0.4222, 0.6496, 0.5581, 0.3984, 0.3463, 0.5714, 0.4401, 0.5307, 0.5388],
    },
    "NMI": {
        "Gr-UMAP":      [0.6508, 0.6721, 0.6044, 0.8164, 0.5590, 0.7382, 0.7399, 0.7571, 0.7249],
        "Average-UMAP": [0.6005, 0.6472, 0.4520, 0.7750, 0.4939, 0.7223, 0.6364, 0.7043, 0.6589],
        "PCA":          [0.2831, 0.4916, 0.4218, 0.4220, 0.1319, 0.4677, 0.3673, 0.4010, 0.2589],
    },
    "ARI": {
        "Gr-UMAP":      [0.5238, 0.5876, 0.4904, 0.6999, 0.5189, 0.5515, 0.5871, 0.6749, 0.7293],
        "Average-UMAP": [0.4795, 0.5546, 0.2889, 0.6451, 0.4518, 0.5084, 0.4545, 0.4570, 0.6449],
        "PCA":          [0.1620, 0.3934, 0.2657, 0.2028, 0.0495, 0.2982, 0.2235, 0.2651, 0.0839],
    },
    "Purity": {
        "Gr-UMAP":      [0.7219, 0.7693, 0.7629, 0.8316, 0.7732, 0.8434, 0.9235, 0.8933, 0.9102],
        "Average-UMAP": [0.6946, 0.7556, 0.6275, 0.7908, 0.7049, 0.8524, 0.8285, 0.8441, 0.8592],
        "PCA":          [0.4844, 0.6672, 0.6314, 0.5477, 0.3963, 0.7404, 0.6741, 0.6504, 0.5918],
    },
}

metrics = ["ACC", "NMI", "ARI", "Purity"]
methods = ["Gr-UMAP", "Average-UMAP", "PCA"]

avg = {m: {meth: np.mean(data[m][meth]) for meth in methods} for m in data}

# -----------------------------
# 2) Plot (clean, paper-style)
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(7, 3.5), constrained_layout=True)

colors = {
    "Gr-UMAP": "#1f77b4",
    "Average-UMAP": "#ff7f0e",
    "PCA": "#2ca02c",
}

def add_value_labels(ax, bars, fmt="{:.3f}", dy=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom")

for i, metric in enumerate(metrics):
    ax = axes[i]
    vals = [avg[metric][meth] for meth in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods], width=0.6)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    if i == 0:
        ax.set_ylabel("Average score")
    ax.tick_params(axis='x', rotation=60)
    add_value_labels(ax, bars)

plt.savefig("setup1_method_averages_paper.png", dpi=600, bbox_inches="tight")
plt.savefig("setup1_method_averages_paper.pdf", bbox_inches="tight")
