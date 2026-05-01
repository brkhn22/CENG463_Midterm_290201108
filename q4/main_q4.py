"""
Question 4: Orchestrator for clustering pipeline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from clustering_tuning import (
    plot_kmeans_elbow_and_silhouette,
    plot_gmm_bic_aic,
    plot_k_distance_graph,
    plot_agglomerative_dendrogram,
)
from clustering_eval import train_and_evaluate_models
from advanced_clustering import kmeans_stability_analysis, cluster_ensemble

# Outputs are organized in outputs/q4/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q4')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 80)
    print("QUESTION 4: CLUSTERING PIPELINE")
    print("=" * 80)

    # 1. Load and scale data
    print("\n[1] Loading digits dataset and scaling...")
    digits = load_digits()
    X = digits.data
    true_labels = digits.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data shape: {X_scaled.shape}")

    # 2. Tuning plots
    print("\n[2] Running clustering tuning plots...")
    plot_kmeans_elbow_and_silhouette(X_scaled)
    plot_gmm_bic_aic(X_scaled)
    plot_k_distance_graph(X_scaled, k=5)
    plot_agglomerative_dendrogram(X_scaled)
    print("Tuning plots saved to outputs/q4/")

    # 3. Evaluation metrics
    print("\n[3] Evaluating clustering models...")
    metrics_df = train_and_evaluate_models(X_scaled, true_labels)
    print(metrics_df.to_string(index=False))

    # 4. Stability analysis and ensemble
    print("\n[4] Stability analysis and ensemble...")
    kmeans_model = KMeans(n_clusters=10, random_state=42, n_init=10)
    mean_ari, std_ari = kmeans_stability_analysis(X_scaled, true_labels, kmeans_model)
    print(f"KMeans stability ARI: mean={mean_ari:.4f}, std={std_ari:.4f}")

    # Fit models for labels
    kmeans_labels = kmeans_model.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=10, random_state=42)
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)

    dbscan = DBSCAN(eps=4.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    agg = AgglomerativeClustering(n_clusters=10, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)

    ensemble_labels, ens_ari, ens_nmi = cluster_ensemble(
        kmeans_labels, gmm_labels, agg_labels, true_labels, n_clusters=10
    )
    print(f"Ensemble ARI: {ens_ari:.4f}, NMI: {ens_nmi:.4f}")

    # 5. UMAP visualization
    print("\n[5] Generating UMAP cluster comparison plot...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    plot_specs = [
        ("K-Means", kmeans_labels),
        ("GMM", gmm_labels),
        ("DBSCAN", dbscan_labels),
        ("Ensemble", ensemble_labels),
    ]

    for ax, (title, labels) in zip(axes, plot_specs):
        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=labels,
            palette='tab10',
            s=12,
            linewidth=0,
            alpha=0.85,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'cluster_comparisons.jpg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster comparison plot to {output_path}")


if __name__ == "__main__":
    main()
