# plot_setup2_method_averages.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Enter the numbers (Setup II)
# -----------------------------

data = {
    "ACC": {
        "Gr-UMAP":      [0.8734, 0.9016, 0.8786, 0.9849, 0.9774, 0.6946, 0.6470, 0.8720, 0.9796],
        "Average-UMAP": [0.8475, 0.8237, 0.7214, 0.9685, 0.9710, 0.6687, 0.6192, 0.8098, 0.9741],
        "NMF":          [0.6875, 0.7189, 0.8364, 0.8006, 0.7073, 0.7370, 0.8994, 0.8847, 0.9796],
        "PCA":          [0.7504, 0.3195, 0.3833, 0.8114, 0.9451, 0.4876, 0.4035, 0.5873, 0.6408],
    },
    "NMI": {
        "Gr-UMAP":      [0.8030, 0.7725, 0.8549, 0.9586, 0.9201, 0.7637, 0.7400, 0.8019, 0.9293],
        "Average-UMAP": [0.7878, 0.7180, 0.7683, 0.9456, 0.8998, 0.7463, 0.7127, 0.7752, 0.9243],
        "NMF":          [0.7244, 0.6693, 0.8017, 0.8854, 0.6321, 0.7949, 0.8829, 0.8694, 0.9293],
        "PCA":          [0.8296, 0.0102, 0.2549, 0.9130, 0.8271, 0.0965, 0.0449, 0.4207, 0.4351],
    },
    "ARI": {
        "Gr-UMAP":      [0.7289, 0.7862, 0.8147, 0.9670, 0.9413, 0.5327, 0.5565, 0.8049, 0.9483],
        "Average-UMAP": [0.7035, 0.6883, 0.6264, 0.9379, 0.9249, 0.5017, 0.5163, 0.7224, 0.9399],
        "NMF":          [0.5996, 0.5556, 0.7314, 0.7479, 0.5727, 0.6120, 0.8929, 0.8311, 0.9483],
        "PCA":          [0.6947, -0.0037, 0.0872, 0.8354, 0.8618, 0.0408, 0.0107, 0.2651, 0.2557],
    },
    "Purity": {
        "Gr-UMAP":      [0.8734, 0.9016, 0.9052, 0.9849, 0.9774, 0.8823, 0.9109, 0.8913, 0.9796],
        "Average-UMAP": [0.8527, 0.8326, 0.8553, 0.9689, 0.9710, 0.8784, 0.8839, 0.8772, 0.9759],
        "NMF":          [0.7455, 0.7531, 0.8719, 0.8301, 0.7073, 0.9099, 0.9600, 0.9412, 0.9796],
        "PCA":          [0.7976, 0.3265, 0.3933, 0.8939, 0.9451, 0.4907, 0.4068, 0.6329, 0.6408],
    },
}

# -----------------------------
# 2) Compute averages per method
# -----------------------------
metrics = ["ACC", "NMI", "ARI", "Purity"]
methods = ["Gr-UMAP", "Average-UMAP", "NMF", "PCA"]

avg = {m: {meth: np.mean(data[m][meth]) for meth in methods} for m in data}

# -----------------------------
# 3) Plot bar charts (1×4 row)
# -----------------------------
fig, axes = plt.subplots(1, 4, figsize=(8, 4), constrained_layout=True)

colors = {
    "Gr-UMAP":      "#1f77b4",  # blue
    "Average-UMAP": "#ff7f0e",  # orange
    "NMF":          "#9467bd",  # purple
    "PCA":          "#2ca02c",  # green
}

def add_value_labels(ax, bars, fmt="{:.3f}", dy=0.01, fs=27):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


for i, metric in enumerate(metrics):
    ax = axes[i]
    vals = [avg[metric][meth] for meth in methods]
    bars = ax.bar(methods, vals, color=[colors[m] for m in methods], width=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(metric, fontsize=27)
    if i == 0:
        ax.set_ylabel("Average score", fontsize=27)
    ax.tick_params(axis='x', rotation=60)
    ax.tick_params(axis='both', labelsize=27)
    add_value_labels(ax, bars, fs=27)

# fig.suptitle("Setup II • Average metrics across datasets", fontsize=14, y=1.03)
plt.savefig("setup2_method_averages.png", dpi=300, bbox_inches="tight")
plt.savefig("setup2_method_averages.pdf", bbox_inches="tight")
