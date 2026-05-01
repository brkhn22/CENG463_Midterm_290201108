"""
Question 4: Cluster Stability Analysis and Cluster Ensembling.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering


def kmeans_stability_analysis(X, true_labels, kmeans_model, n_iterations=10,
                              sample_frac=0.8, random_state=42):
    """
    Perform bootstrap stability analysis for KMeans.

    Returns:
        mean_ari: float
        std_ari: float
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_sub = int(n_samples * sample_frac)

    ari_scores = []

    for _ in range(n_iterations):
        idx = rng.choice(n_samples, size=n_sub, replace=True)
        X_sub = X[idx]
        y_sub = true_labels[idx]

        model = kmeans_model
        model.fit(X_sub)
        pred = model.labels_

        ari = adjusted_rand_score(y_sub, pred)
        ari_scores.append(ari)

    mean_ari = float(np.mean(ari_scores))
    std_ari = float(np.std(ari_scores))

    return mean_ari, std_ari


def cluster_ensemble(labels_kmeans, labels_gmm, labels_agg, true_labels, n_clusters=10):
    """
    Build co-association matrix from 3 labelings and derive ensemble labels.

    Returns:
        ensemble_labels: array
        ari: float
        nmi: float
    """
    labels_list = [labels_kmeans, labels_gmm, labels_agg]
    n = len(labels_kmeans)

    co_assoc = np.zeros((n, n), dtype=np.float32)

    for labels in labels_list:
        same = labels[:, None] == labels[None, :]
        co_assoc += same.astype(np.float32)

    co_assoc /= len(labels_list)

    dist_matrix = 1.0 - co_assoc

    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
        metric="precomputed"
    )
    ensemble_labels = agg.fit_predict(dist_matrix)

    ari = adjusted_rand_score(true_labels, ensemble_labels)
    nmi = normalized_mutual_info_score(true_labels, ensemble_labels)

    return ensemble_labels, float(ari), float(nmi)


if __name__ == "__main__":
    print("This module provides stability analysis and ensemble utilities.")
