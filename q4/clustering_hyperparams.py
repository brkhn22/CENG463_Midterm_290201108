"""
Question 4: Hyperparameter search helpers for clustering algorithms.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Outputs are organized in outputs/q4/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q4')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_kmeans_elbow_and_silhouette(X, k_range=range(2, 16), random_state=42):
    """
    Plot K-Means inertia (Elbow) and Silhouette scores for k in [2, 15].
    Saves plot to outputs/q4/kmeans_elbow_silhouette.png.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow Method (Inertia)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, marker='o', color='tab:orange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'kmeans_elbow_silhouette.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return inertias, silhouettes


def plot_gmm_bic_aic(X, n_components_range=range(2, 16), random_state=42):
    """
    Fit GMMs and plot BIC/AIC for n_components in [2, 15].
    Saves plot to outputs/q4/gmm_bic_aic.png.
    """
    bics = []
    aics = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    plt.figure(figsize=(8, 5))
    plt.plot(list(n_components_range), bics, marker='o', label='BIC')
    plt.plot(list(n_components_range), aics, marker='o', label='AIC')
    plt.title('GMM Model Selection (BIC/AIC)')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'gmm_bic_aic.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return bics, aics


def plot_k_distance_graph(X, k=5):
    """
    Plot k-distance graph to help choose DBSCAN eps.
    Saves plot to outputs/q4/dbscan_k_distance.png.
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # Sort distances of the k-th neighbor
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.title(f'k-Distance Graph (k={k})')
    plt.xlabel('Points (sorted)')
    plt.ylabel(f'{k}-NN Distance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'dbscan_k_distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return k_distances


def plot_agglomerative_dendrogram(X, method='ward'):
    """
    Plot dendrogram for Agglomerative Clustering using Ward linkage.
    Saves plot to outputs/q4/agglomerative_dendrogram.png.
    """
    Z = linkage(X, method=method)

    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Agglomerative Clustering Dendrogram (Ward)')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'agglomerative_dendrogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return Z


if __name__ == "__main__":
    print("This module provides clustering hyperparameter helper plots.")
