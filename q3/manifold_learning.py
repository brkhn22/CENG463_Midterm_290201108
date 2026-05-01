"""
Question 3: Non-linear manifold learning and evaluation.
"""

import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
import umap


def run_tsne_grid_search(X, perplexities=(5, 30, 50), random_state=42):
    """
    Run t-SNE over a small perplexity grid and return best embeddings by KL.

    Returns:
        best_embedding: (n_samples, 2)
        best_perplexity: int
        best_kl: float
    """
    best_embedding = None
    best_perplexity = None
    best_kl = np.inf

    for p in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            random_state=random_state,
            init="pca",
            learning_rate="auto"
        )
        embedding = tsne.fit_transform(X)
        kl = float(tsne.kl_divergence_)

        if kl < best_kl:
            best_kl = kl
            best_perplexity = p
            best_embedding = embedding

    return best_embedding, best_perplexity, best_kl


def run_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """
    Run UMAP and return 2D embeddings.

    Returns:
        embedding: (n_samples, 2)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(X)
    return embedding


def evaluate_embeddings(X, embedding, subset_size=1000, random_state=42):
    """
    Evaluate 2D embeddings using trustworthiness and Kruskal's Stress.

    Returns:
        trust: float
        stress: float
    """
    trust = trustworthiness(X, embedding, n_neighbors=5)

    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if subset_size > n:
        subset_size = n

    idx = rng.choice(n, size=subset_size, replace=False)
    X_sub = X[idx]
    emb_sub = embedding[idx]

    dist_orig = pairwise_distances(X_sub, metric="euclidean")
    dist_emb = pairwise_distances(emb_sub, metric="euclidean")

    num = np.sum((dist_orig - dist_emb) ** 2)
    den = np.sum(dist_orig ** 2)
    stress = np.sqrt(num / den) if den > 0 else 0.0

    return trust, stress


if __name__ == "__main__":
    from pca_fashion_mnist import load_fashion_mnist_subsample

    X, y = load_fashion_mnist_subsample()

    tsne_emb, best_p, best_kl = run_tsne_grid_search(X)
    tsne_trust, tsne_stress = evaluate_embeddings(X, tsne_emb)

    umap_emb = run_umap(X)
    umap_trust, umap_stress = evaluate_embeddings(X, umap_emb)

    print(f"t-SNE best perplexity: {best_p}, KL: {best_kl:.6f}")
    print(f"t-SNE trustworthiness: {tsne_trust:.6f}, stress: {tsne_stress:.6f}")
    print(f"UMAP trustworthiness: {umap_trust:.6f}, stress: {umap_stress:.6f}")
