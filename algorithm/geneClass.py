import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def divide_features_auto_k_simple(
    X,
    k_list=(4,6,8,10,12,16,20),
    random_state=1,
    n_init=10,
    zscore=False
):
    """
    自动选择 k（用 silhouette 最大化），返回 index_features（仅簇索引列表）。
    X: (n_cells, n_genes)  行=细胞，列=基因（与你原函数一致）
    """
    # genes × cells
    G = X.T.astype(float, copy=False)

    # 可选：按细胞维度做 z-score，防止量纲影响
    if zscore:
        m = G.mean(axis=1, keepdims=True)
        s = G.std(axis=1, keepdims=True) + 1e-8
        G = (G - m) / s

    best_k, best_sil = None, -1.0
    for k in k_list:
        if k <= 1 or k >= G.shape[0]:
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(G)
        # 需要至少2个簇且每簇>1个点
        if len(np.unique(labels)) < 2:
            continue
        try:
            sil = silhouette_score(G, labels, metric='euclidean')
        except Exception:
            sil = -1.0
        if sil > best_sil:
            best_sil, best_k = sil, k

    # 兜底：如果所有 k 都失败，就用 k=4
    if best_k is None:
        best_k = 4

    # 用选出的 k* 在所有基因上再拟合一次
    km = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init)
    final_labels = km.fit_predict(G)

    # 组装成 index_features（每个簇的基因下标）
    index_features = [np.where(final_labels == l)[0] for l in range(best_k)]
    return index_features, best_k