"""
Question 3: Undercomplete Autoencoder for Fashion-MNIST.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error


def build_autoencoder(input_dim=784):
    """
    Build an undercomplete autoencoder with a 2D bottleneck.

    Returns:
        encoder: Keras Model
        autoencoder: Keras Model
    """
    inputs = keras.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    bottleneck = layers.Dense(2, activation=None, name="bottleneck")(x)

    # Decoder
    x = layers.Dense(64, activation="relu")(bottleneck)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    encoder = keras.Model(inputs, bottleneck, name="encoder")
    autoencoder = keras.Model(inputs, outputs, name="autoencoder")

    autoencoder.compile(optimizer="adam", loss="mse")

    return encoder, autoencoder


def train_autoencoder(X, epochs=20, batch_size=256, validation_split=0.1, verbose=1):
    """
    Train the autoencoder on X and return models, embeddings, and MSE.

    Returns:
        encoder: trained encoder model
        autoencoder: trained autoencoder model
        embeddings: (n_samples, 2)
        recon_mse: float
    """
    encoder, autoencoder = build_autoencoder(input_dim=X.shape[1])

    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=verbose
    )

    embeddings = encoder.predict(X, batch_size=batch_size, verbose=0)
    reconstructed = autoencoder.predict(X, batch_size=batch_size, verbose=0)

    recon_mse = mean_squared_error(X, reconstructed)

    return encoder, autoencoder, embeddings, recon_mse


if __name__ == "__main__":
    from pca_fashion_mnist import load_fashion_mnist_subsample

    X, y = load_fashion_mnist_subsample()
    encoder, autoencoder, embeddings, mse = train_autoencoder(X)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Reconstruction MSE: {mse:.6f}")
