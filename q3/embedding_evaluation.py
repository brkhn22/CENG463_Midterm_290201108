"""
Question 3: Downstream evaluation and plotting for 2D embeddings.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Outputs are organized in outputs/q3/ subfolder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q3')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def knn_cv_accuracy(embeddings, y, n_neighbors=5, cv=5):
    """
    Train KNN on 2D embeddings and return 5-fold CV accuracy.

    Returns:
        mean_acc: float
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn, embeddings, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores))


def plot_all_embeddings(embeddings_dict, y, save_path=None):
    """
    Plot a 2x3 grid of scatter plots for multiple 2D embeddings.

    embeddings_dict keys should include:
        'PCA', 'Kernel PCA', 't-SNE', 'UMAP', 'Autoencoder'
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'all_embeddings.jpg')

    methods = ['PCA', 'Kernel PCA', 't-SNE', 'UMAP', 'Autoencoder']
    palette = sns.color_palette('tab10', n_colors=len(np.unique(y)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, method in enumerate(methods):
        ax = axes[i]
        emb = embeddings_dict.get(method)
        if emb is None:
            ax.set_title(f"{method} (missing)")
            ax.axis('off')
            continue

        sns.scatterplot(
            x=emb[:, 0], y=emb[:, 1],
            hue=y,
            palette=palette,
            s=10,
            linewidth=0,
            alpha=0.8,
            legend=(i == 0),
            ax=ax
        )
        ax.set_title(method)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    # Hide the 6th subplot if unused
    if len(methods) < len(axes):
        axes[-1].axis('off')

    if axes[0].get_legend() is not None:
        axes[0].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def build_decoder_from_autoencoder(autoencoder, bottleneck_name='bottleneck'):
    """
    Build a decoder model from a full autoencoder using the bottleneck layer.

    Returns:
        decoder: Keras Model
    """
    import tensorflow as tf

    bottleneck_layer = autoencoder.get_layer(bottleneck_name)
    bottleneck_index = autoencoder.layers.index(bottleneck_layer)

    bottleneck_dim = int(bottleneck_layer.output.shape[-1])
    decoder_input = tf.keras.Input(shape=(bottleneck_dim,))
    x = decoder_input

    for layer in autoencoder.layers[bottleneck_index + 1:]:
        x = layer(x)

    decoder = tf.keras.Model(decoder_input, x, name='decoder')
    return decoder


def plot_ae_manifold(decoder, grid_min=-4.0, grid_max=4.0, grid_size=15, save_path=None):
    """
    Visualize autoencoder manifold by decoding a 2D grid.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'ae_manifold.jpg')

    grid_x = np.linspace(grid_min, grid_max, grid_size)
    grid_y = np.linspace(grid_min, grid_max, grid_size)

    figure = np.zeros((28 * grid_size, 28 * grid_size))

    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            z = np.array([[x, y]], dtype=np.float32)
            decoded = decoder.predict(z, verbose=0)
            img = decoded.reshape(28, 28)
            figure[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title('Autoencoder Manifold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Example usage (requires embeddings to be computed elsewhere)
    print("This module provides evaluation and plotting utilities.")
