"""
Question 3: Orchestrator for dimensionality reduction, manifold learning, and evaluation.
"""

from pca_methods import load_fashion_mnist_subsample, apply_pca, apply_kernel_pca
from manifold_methods import run_tsne_grid_search, run_umap, evaluate_embeddings
from autoencoder_method import train_autoencoder
from eval_viz import (
    knn_cv_accuracy,
    plot_all_embeddings,
    build_decoder_from_autoencoder,
    plot_ae_manifold,
)


def main():
    print("\n" + "=" * 80)
    print("QUESTION 3: DIMENSIONALITY REDUCTION AND MANIFOLD LEARNING")
    print("=" * 80)

    # 1. Load subsampled data
    print("\n[1] Loading Fashion-MNIST (subsample=10,000)...")
    X, y = load_fashion_mnist_subsample(n_samples=10000, seed=42)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # 2. Run PCA and Kernel PCA
    print("\n[2] Running PCA...")
    pca_emb, pca_model, pca_mse = apply_pca(X)
    print(f"PCA reconstruction MSE: {pca_mse:.6f}")

    print("\n[3] Running Kernel PCA...")
    kpca_emb, kpca_model, kpca_mse = apply_kernel_pca(X)
    print(f"Kernel PCA reconstruction MSE: {kpca_mse:.6f}")

    # 3. Run t-SNE and UMAP
    print("\n[4] Running t-SNE grid search...")
    tsne_emb, best_p, best_kl = run_tsne_grid_search(X)
    print(f"Best t-SNE perplexity: {best_p}, KL: {best_kl:.6f}")

    print("\n[5] Running UMAP...")
    umap_emb = run_umap(X)

    # 4. Run Autoencoder
    print("\n[6] Training Autoencoder...")
    encoder, autoencoder, ae_emb, ae_mse = train_autoencoder(X)
    print(f"Autoencoder reconstruction MSE: {ae_mse:.6f}")

    # 5. Evaluation
    print("\n" + "-" * 80)
    print("SUMMARY REPORT")
    print("-" * 80)

    # Reconstruction errors
    print("Reconstruction Errors (MSE):")
    print(f"  PCA:   {pca_mse:.6f}")
    print(f"  KPCA:  {kpca_mse:.6f}")
    print(f"  AE:    {ae_mse:.6f}")

    # Trustworthiness and stress
    print("\nTrustworthiness and Kruskal's Stress:")
    tsne_trust, tsne_stress = evaluate_embeddings(X, tsne_emb)
    umap_trust, umap_stress = evaluate_embeddings(X, umap_emb)
    print(f"  t-SNE: Trust={tsne_trust:.6f}, Stress={tsne_stress:.6f}")
    print(f"  UMAP:  Trust={umap_trust:.6f}, Stress={umap_stress:.6f}")

    # KNN accuracy
    print("\n5-Fold KNN Accuracy (k=5):")
    knn_pca = knn_cv_accuracy(pca_emb, y)
    knn_kpca = knn_cv_accuracy(kpca_emb, y)
    knn_tsne = knn_cv_accuracy(tsne_emb, y)
    knn_umap = knn_cv_accuracy(umap_emb, y)
    knn_ae = knn_cv_accuracy(ae_emb, y)

    print(f"  PCA:         {knn_pca:.4f}")
    print(f"  Kernel PCA:  {knn_kpca:.4f}")
    print(f"  t-SNE:       {knn_tsne:.4f}")
    print(f"  UMAP:        {knn_umap:.4f}")
    print(f"  Autoencoder: {knn_ae:.4f}")

    # 6. Visualizations
    print("\n[7] Generating visualizations...")
    embeddings_dict = {
        "PCA": pca_emb,
        "Kernel PCA": kpca_emb,
        "t-SNE": tsne_emb,
        "UMAP": umap_emb,
        "Autoencoder": ae_emb,
    }
    plot_all_embeddings(embeddings_dict, y)

    decoder = build_decoder_from_autoencoder(autoencoder)
    plot_ae_manifold(decoder)

    print("\nDone. Check outputs/q3 for generated images.")


if __name__ == "__main__":
    main()
