"""
RNN (LSTM) classification model for lead price prediction.

This module implements LSTM (Long Short-Term Memory) neural network for time series classification.
The model is designed to capture temporal patterns and long-term dependencies in sequential data.

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Evaluate classification model performance using accuracy, precision, recall, F1, and AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values (binary).
    y_pred : np.ndarray
        Predicted target values (binary).
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class. Required for AUC calculation.

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'accuracy', 'precision', 'recall', 'f1', and 'auc' metrics.
        'auc' will be NaN if y_proba is None or if there's an error.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes or if arrays are empty.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"y_pred has shape {y_pred.shape}"
        )

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = np.nan
    if y_proba is not None:
        try:
            y_proba = np.array(y_proba)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten()
            auc = float(roc_auc_score(y_true, y_proba))
        except:
            auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def create_sequences(
    X_data: np.ndarray, y_data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data for LSTM input.

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
        # Target is at the end of the sequence
        y.append(y_data[i + seq_length - 1])
    return np.array(X), np.array(y)


def train_rnn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seq_length: int = 10,
    lstm_units: List[int] = None,
    dense_units: List[int] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    verbose: int = 0,
    random_state: int = 42,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate an LSTM classification model.

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
    lstm_units : List[int], default=None
        List of units for each LSTM layer. Default: [64, 32]
    dense_units : List[int], default=None
        List of units for dense layers after LSTM. Default: [16]
    dropout_rate : float, default=0.2
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
            - 'model_name': 'rnn'
            - 'accuracy': Accuracy score
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'auc': AUC-ROC score
            - 'y_pred': Predicted classes on test set
            - 'y_proba': Predicted probabilities on test set

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
            "tensorflow is required for RNN model. Install it using: pip install tensorflow"
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

    # Set default LSTM and dense units
    if lstm_units is None:
        lstm_units = [64, 32]
    if dense_units is None:
        dense_units = [16]

    # Standardize features
    scaler_X = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X)
    test_X_scaled = scaler_X.transform(test_X)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(train_X_scaled, train_y, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_X_scaled, test_y, seq_length)

    if len(X_train_seq) == 0:
        raise ValueError(
            f"Sequence length {seq_length} is too large. "
            f"Need at least {seq_length + 1} samples."
        )

    # Build LSTM model for classification
    model = keras.Sequential()

    # Add first LSTM layer
    model.add(
        layers.LSTM(
            lstm_units[0],
            activation="tanh",
            return_sequences=len(lstm_units) > 1,
            input_shape=(seq_length, train_X_scaled.shape[1]),
        )
    )
    model.add(layers.Dropout(dropout_rate))

    # Add additional LSTM layers if specified
    for units in lstm_units[1:]:
        model.add(layers.LSTM(units, activation="tanh", return_sequences=False))
        model.add(layers.Dropout(dropout_rate))

    # Add dense layers
    for units in dense_units:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Manually split training sequences into train and validation sets (time-ordered)
    split_index = int(len(X_train_seq) * (1 - validation_split))
    X_train_final = X_train_seq[:split_index]
    y_train_final = y_train_seq[:split_index]
    X_val = X_train_seq[split_index:]
    y_val = y_train_seq[split_index:]

    # Train model with explicit validation data (time-ordered)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=verbose,
        shuffle=False,  # Explicitly disable shuffling for time series
    )

    # Make predictions
    y_proba = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    # Convert to 2D array for consistency with other models
    y_proba_2d = np.column_stack([1 - y_proba, y_proba])

    # For evaluation, use the aligned test_y values
    y_test_aligned = y_test_seq

    # Evaluate
    metrics = evaluate_classification(y_test_aligned, y_pred, y_proba_2d)

    return {
        "model_name": "rnn",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "y_pred": y_pred,
        "y_proba": y_proba_2d,
        "y_test_aligned": y_test_aligned,  # Store aligned test_y for proper ROC plotting
    }


def run_rnn_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seq_length: int = 10,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Run RNN (LSTM) model with default parameters.

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
        Dictionary containing RNN model results with 'model_name', metrics, 'y_pred', 'y_proba'.
    """
    return train_rnn(
        train_X, train_y, test_X, test_y, seq_length=seq_length, verbose=0
    )


if __name__ == "__main__":
    print("RNN (LSTM) classification model loaded.")

