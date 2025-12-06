"""
Transformer regression model for lead price prediction.

This module implements a Transformer-based neural network for time series regression.
The model uses multi-head self-attention mechanisms to capture long-term dependencies
and temporal patterns in sequential data.

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression model performance using MAE and RMSE.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'mae' and 'rmse' metrics.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes or if arrays are empty.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {"mae": float(mae), "rmse": float(rmse)}


def create_sequences(
    X_data: np.ndarray, y_data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data for Transformer input.

    Parameters
    ----------
    X_data : np.ndarray
        Input features of shape (n_samples, n_features).
    y_data : np.ndarray
        Target values of shape (n_samples,).
    seq_length : int
        Length of input sequences (number of time steps).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: Sequences of shape (n_sequences, seq_length, n_features)
        y: Target values of shape (n_sequences,)
    """
    X, y = [], []
    for i in range(len(X_data) - seq_length):
        X.append(X_data[i : i + seq_length])
        # Target is the return_7d at the end of the sequence (aligned with Linear Regression)
        # Using features from i to i+seq_length-1 to predict return_7d[i+seq_length-1]
        y.append(y_data[i + seq_length - 1])
    return np.array(X), np.array(y)


def train_transformer(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seq_length: int = 10,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dff: int = 128,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    verbose: int = 0,
    random_state: int = 42,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a Transformer regression model for time series.

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    train_y : np.ndarray
        Training target vector of shape (n_samples,).
    test_X : np.ndarray
        Test feature matrix of shape (n_samples_test, n_features).
    test_y : np.ndarray
        Test target vector of shape (n_samples_test,).
    seq_length : int, default=10
        Length of input sequences (number of time steps to look back).
    d_model : int, default=64
        Dimension of the model (embedding dimension).
    num_heads : int, default=4
        Number of attention heads.
    num_layers : int, default=2
        Number of Transformer encoder layers.
    dff : int, default=128
        Dimension of feed-forward network.
    dropout_rate : float, default=0.1
        Dropout rate for regularization.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    batch_size : int, default=32
        Batch size for training.
    epochs : int, default=50
        Number of training epochs.
    validation_split : float, default=0.2
        Fraction of training data to use for validation.
    verbose : int, default=0
        Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'transformer'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    ImportError
        If tensorflow/keras is not installed.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "tensorflow is required for Transformer model. Install it using: pip install tensorflow"
        )

    # Set random seeds for reproducibility
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Input validation
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    if len(train_X) == 0 or len(test_X) == 0:
        raise ValueError("Training or test set cannot be empty")

    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train_X has {train_X.shape[1]} features, "
            f"test_X has {test_X.shape[1]} features"
        )

    if len(train_y.shape) > 1 and train_y.shape[1] != 1:
        raise ValueError("train_y must be a 1D array or column vector")

    if len(test_y.shape) > 1 and test_y.shape[1] != 1:
        raise ValueError("test_y must be a 1D array or column vector")

    # Flatten y arrays if needed
    train_y = train_y.flatten()
    test_y = test_y.flatten()

    # Standardize features only (do not include target in input)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale features (fit only on training data)
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)

    # Create sequences using only features (X), target (y) is separate
    # IMPORTANT: Use original train_y (not scaled) for sequence creation
    X_train_seq, y_train_seq = create_sequences(train_X_scaled, train_y, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_X_scaled, test_y, seq_length)

    # Scale target ONLY on the actual training sequences (y_train_seq)
    # This ensures no data leakage: scaler statistics are based only on data actually used for training
    scaler_y.fit(y_train_seq.reshape(-1, 1))
    y_train_seq_scaled = scaler_y.transform(y_train_seq.reshape(-1, 1)).flatten()

    if len(X_train_seq) == 0:
        raise ValueError(
            f"Sequence length {seq_length} is too large. "
            f"Need at least {seq_length + 1} samples."
        )

    n_features = X_train_seq.shape[2]

    # Define positional encoding layer
    def get_positional_encoding(seq_len, d_model):
        """Create positional encoding matrix."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pos_encoding, dtype=tf.float32)

    # Build Transformer model
    inputs = layers.Input(shape=(seq_length, n_features))

    # Embedding layer
    x = layers.Dense(d_model)(inputs)

    # Add positional encoding
    pos_encoding = get_positional_encoding(seq_length, d_model)
    x = x + pos_encoding

    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn_output = layers.Dense(dff, activation="relu")(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global average pooling over sequence dimension
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers for regression
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Manually split training sequences into train and validation sets
    # This ensures time-ordered validation (no data leakage from future data)
    # validation_split is now used to determine the split ratio
    split_index = int(len(X_train_seq) * (1 - validation_split))
    X_train_final = X_train_seq[:split_index]
    y_train_final = y_train_seq_scaled[:split_index]
    X_val = X_train_seq[split_index:]
    y_val = y_train_seq_scaled[split_index:]

    # Train model with explicit validation data (time-ordered)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),  # Explicit time-ordered validation set
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=verbose,
        shuffle=False,  # Explicitly disable shuffling for time series
    )

    # Make predictions
    y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()

    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # For evaluation, use the aligned test_y values (already aligned from create_sequences)
    # y_test_seq corresponds to test_y[seq_length-1:], which aligns with predictions
    y_test_aligned = y_test_seq

    # Evaluate
    metrics = evaluate_regression(y_test_aligned, y_pred)

    return {
        "model_name": "transformer",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
    }


def run_transformer_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seq_length: int = 10,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Run Transformer model with default parameters.

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix of shape (n_samples, n_features).
    train_y : np.ndarray
        Training target vector of shape (n_samples,).
    test_X : np.ndarray
        Test feature matrix of shape (n_samples_test, n_features).
    test_y : np.ndarray
        Test target vector of shape (n_samples_test,).
    seq_length : int, default=10
        Length of input sequences (number of time steps to look back).

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing Transformer model results with 'model_name', 'rmse', 'mae', 'y_pred'.
    """
    return train_transformer(
        train_X, train_y, test_X, test_y, seq_length=seq_length, verbose=0
    )


if __name__ == "__main__":
    print("Transformer regression model loaded.")

