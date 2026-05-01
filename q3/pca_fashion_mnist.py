"""
Question 3: Fashion-MNIST loading and PCA-based dimensionality reduction.
"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
from keras.datasets import fashion_mnist


def load_fashion_mnist_subsample(n_samples=10000, seed=42):
    """
    Load Fashion-MNIST, flatten to 784, scale to [0, 1], and subsample.

    Returns:
        X_sub: (n_samples, 784) float32
        y_sub: (n_samples,) int
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    x = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    y = y.astype(int)

    if n_samples > x.shape[0]:
        raise ValueError(f"n_samples={n_samples} exceeds dataset size {x.shape[0]}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(x.shape[0], size=n_samples, replace=False)

    X_sub = x[indices]
    y_sub = y[indices]

    return X_sub, y_sub


def apply_pca(X, n_components=2, random_state=42):
    """
    Apply standard PCA and compute reconstruction error (MSE).

    Returns:
        X_embedded: (n_samples, n_components)
        pca_model: fitted PCA instance
        recon_error: float
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_embedded = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_embedded)

    recon_error = mean_squared_error(X, X_recon)
    return X_embedded, pca, recon_error


def apply_kernel_pca(X, n_components=2, gamma=None):
    """
    Apply Kernel PCA (RBF) and compute reconstruction error (MSE).

    Returns:
        X_embedded: (n_samples, n_components)
        kpca_model: fitted KernelPCA instance
        recon_error: float
    """
    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True
    )
    X_embedded = kpca.fit_transform(X)
    X_recon = kpca.inverse_transform(X_embedded)

    recon_error = mean_squared_error(X, X_recon)
    return X_embedded, kpca, recon_error


if __name__ == "__main__":
    X_sub, y_sub = load_fashion_mnist_subsample()
    X_pca, pca_model, pca_mse = apply_pca(X_sub)
    X_kpca, kpca_model, kpca_mse = apply_kernel_pca(X_sub)

    print(f"Subsampled shape: X={X_sub.shape}, y={y_sub.shape}")
    print(f"PCA reconstruction MSE: {pca_mse:.6f}")
    print(f"Kernel PCA reconstruction MSE: {kpca_mse:.6f}")
