"""
Question 4: Train and evaluate clustering models on the digits dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)


def evaluate_clustering(X, true_labels, predicted_labels):
    """
    Compute internal and external clustering metrics.

    Returns:
        metrics: dict
    """
    metrics = {
        "Silhouette": silhouette_score(X, predicted_labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, predicted_labels),
        "Davies-Bouldin": davies_bouldin_score(X, predicted_labels),
        "ARI": adjusted_rand_score(true_labels, predicted_labels),
        "NMI": normalized_mutual_info_score(true_labels, predicted_labels),
        "Fowlkes-Mallows": fowlkes_mallows_score(true_labels, predicted_labels),
    }
    return metrics


def train_and_evaluate_models(X, true_labels):
    """
    Train KMeans, GMM, DBSCAN, and AgglomerativeClustering and evaluate.

    Returns:
        results_df: DataFrame with metrics for each model
    """
    results = []

    # 1. KMeans
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    results.append({"Model": "KMeans", **evaluate_clustering(X, true_labels, kmeans_labels)})

    # 2. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=10, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    results.append({"Model": "GMM", **evaluate_clustering(X, true_labels, gmm_labels)})

    # 3. DBSCAN
    dbscan = DBSCAN(eps=4.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    results.append({"Model": "DBSCAN", **evaluate_clustering(X, true_labels, dbscan_labels)})

    # 4. Agglomerative Clustering (Ward)
    agg = AgglomerativeClustering(n_clusters=10, linkage="ward")
    agg_labels = agg.fit_predict(X)
    results.append({"Model": "Agglomerative", **evaluate_clustering(X, true_labels, agg_labels)})

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    y = digits.target

    df = train_and_evaluate_models(X, y)
    print(df.to_string(index=False))
