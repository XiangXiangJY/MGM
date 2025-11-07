import matplotlib.pyplot as plt

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

# ----------------------------
# Data (updated to your latest numbers)
# ----------------------------
scales = [5, 6, 7, 8, 9, 10, 12, 13, 15]

acc = [0.9796, 0.9796, 0.9796, 0.9796, 0.9796, 0.9796, 0.9592, 0.9592, 0.9592]
ari = [0.9483, 0.9483, 0.9483, 0.9483, 0.9483, 0.9483, 0.8996, 0.8996, 0.8996]
nmi = [0.9293, 0.9293, 0.9293, 0.9293, 0.9293, 0.9293, 0.8609, 0.8609, 0.8609]
pur = [0.9796, 0.9796, 0.9796, 0.9796, 0.9796, 0.9796, 0.9592, 0.9592, 0.9592]

# averages & MGM values
avg = {"ACC": 0.9728, "ARI": 0.9320, "NMI": 0.9065, "Purity": 0.9728}
mgm = {"ACC": 0.9796, "ARI": 0.9483, "NMI": 0.9293, "Purity": 0.9796}

# ----------------------------
# Plot (1×4 panels)
# ----------------------------
metrics = ["ACC", "ARI", "NMI", "Purity"]
ydata = [acc, ari, nmi, pur]
yavg = [avg[m] for m in metrics]
ymgm = [mgm[m] for m in metrics]

fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.1), constrained_layout=True)

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.plot(scales, ydata[i], 'o-', color='#1f77b4', markersize=3.5, linewidth=1.0,
            label='PCA$\\to$UMAP (per $s$)')
    ax.axhline(yavg[i], color='gray', linestyle='--', linewidth=1.0,
               label='Average PCA$\\to$UMAP')
    ax.axhline(ymgm[i], color='red', linestyle='--', linewidth=1.0,
               label='MGM (multi-scale)')
    ax.set_title(metric)
    ax.set_xlabel(r'neighborhood size $s$')
    ax.set_ylim(0.7, 1.02)
    if i == 0:
        ax.set_ylabel('score')
    ax.grid(alpha=0.2)

# 
axes[-1].legend(frameon=False, fontsize=7,
                loc='center', bbox_to_anchor=(1.7, 0.5),
                ncol=1, columnspacing=0.5)

plt.savefig("GSE57249_scales_4metrics.pdf", dpi=600, bbox_inches="tight")
plt.savefig("GSE57249_scales_4metrics.png", dpi=600, bbox_inches="tight")