import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
     "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
})

# --------------------------
# GSE84133Human2 (your log)
# --------------------------
scales = [5, 6, 8, 9, 11, 13, 16, 18, 21, 25, 29, 33, 37, 42, 47, 53, 58, 65, 71, 78, 85, 92, 100]

# PCA→UMAP (per s)
acc = [0.6792, 0.6657, 0.6293, 0.6575, 0.6304, 0.6575, 0.5975, 0.6463, 0.5617,
       0.6240, 0.6011, 0.5864, 0.5476, 0.5323, 0.4877, 0.5635, 0.5505, 0.5076,
       0.5546, 0.5740, 0.5170, 0.6340, 0.6052]

ari = [0.6194, 0.5738, 0.5714, 0.5289, 0.5853, 0.5842, 0.5325, 0.5235, 0.4665,
       0.5554, 0.5168, 0.4432, 0.4072, 0.4501, 0.3824, 0.3660, 0.3711, 0.3519,
       0.3556, 0.4006, 0.3879, 0.3686, 0.3366]

 
nmi = [0.7873, 0.7610, 0.7534, 0.7233, 0.7672, 0.7362, 0.7332, 0.7104, 0.6904,
       0.7402, 0.6996, 0.6657, 0.6479, 0.6383, 0.5905, 0.5608, 0.5627, 0.5590,
       0.5574, 0.5486, 0.5400, 0.5384, 0.5249]

pur = [0.9448, 0.9189, 0.9313, 0.8942, 0.9418, 0.9042, 0.9095, 0.8819, 0.8766,
       0.9107, 0.8684, 0.8502, 0.8502, 0.8414, 0.8114, 0.7297, 0.7867, 0.8038,
       0.7609, 0.7485, 0.7685, 0.7186, 0.7215]

# Gr-UMAP (one number per metric)
gr_acc = 0.7145
gr_ari = 0.6015
gr_nmi = 0.7468
gr_pur = 0.9271

# baseline average (from your log)
avg_acc = 0.5918
avg_ari = 0.4643
avg_nmi = 0.6538
avg_pur = 0.8423

fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.1), constrained_layout=True)

def one_panel(ax, x, y, gr_val, avg_val, title, ylim=None):
    # 单scale曲线
    ax.plot(x, y, 'o-', color='#1f77b4', markersize=3, linewidth=1.0,
            label='PCA→UMAP (per $s$)')
    # baseline average
    ax.axhline(avg_val, color='gray', linestyle='--', linewidth=1.0,
               label='Average PCA→UMAP')
    # Gr-UMAP
    ax.axhline(gr_val, color='red', linestyle='--', linewidth=1.0,
               label='MGM (multi-scale)')
    ax.set_title(title)
    ax.set_xlabel(r'neighborhood $s$')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

# ACC
one_panel(axes[0], scales, acc, gr_acc, avg_acc, 'ACC', ylim=(0.3, 1.02))
axes[0].set_ylabel('score')

# NMI
one_panel(axes[1], scales, nmi, gr_nmi, avg_nmi, 'NMI', ylim=(0.45, 1.02))

# ARI
one_panel(axes[2], scales, ari, gr_ari, avg_ari, 'ARI', ylim=(0.25, 1.02))

# Purity
one_panel(axes[3], scales, pur, gr_pur, avg_pur, 'Purity', ylim=(0.6, 1.02))

 
# axes[3].legend(frameon=False, fontsize=6, loc='lower right')
# axes[0].legend(frameon=False, fontsize=7, loc='upper right')

axes[-1].legend(frameon=False, fontsize=7,
                loc='center', bbox_to_anchor=(1.6, 0.5),
                ncol=1, columnspacing=0.5)

plt.savefig("GSEHuman2_scales_4metrics.pdf", dpi=600, bbox_inches="tight")
plt.savefig("GSEHuman2_scales_4metrics.png", dpi=600, bbox_inches="tight")
plt.close(fig)