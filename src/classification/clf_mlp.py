"""
MLP (Multi-Layer Perceptron) classification model for lead price prediction.

This module implements MLP neural network for classification.
The model uses multiple fully connected layers to learn complex non-linear patterns.

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional


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


def train_mlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    hidden_layer_sizes: tuple = (100, 50),
    activation: str = 'relu',
    solver: str = 'adam',
    alpha: float = 0.0001,
    learning_rate: str = 'constant',
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
    batch_size: int = 'auto',
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    standardize: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate an MLP classification model.

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
    hidden_layer_sizes : tuple, default=(100, 50)
        Number of neurons in each hidden layer.
    activation : str, default='relu'
        Activation function for hidden layers. Options: 'identity', 'logistic', 'tanh', 'relu'.
    solver : str, default='adam'
        Solver for weight optimization. Options: 'lbfgs', 'sgd', 'adam'.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    learning_rate : str, default='constant'
        Learning rate schedule. Options: 'constant', 'invscaling', 'adaptive'.
    learning_rate_init : float, default=0.001
        Initial learning rate.
    max_iter : int, default=500
        Maximum number of iterations.
    batch_size : int or str, default='auto'
        Size of minibatches. 'auto' uses min(200, n_samples).
    early_stopping : bool, default=True
        Whether to use early stopping to terminate training when validation score is not improving.
    validation_fraction : float, default=0.1
        Proportion of training data to set aside as validation set for early stopping.
    random_state : int, default=42
        Random state for reproducibility.
    standardize : bool, default=True
        Whether to standardize features using StandardScaler.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'mlp'
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
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
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

    # Standardize features if requested (important for neural networks)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    # Train MLP model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        batch_size=batch_size,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)

    # Evaluate
    metrics = evaluate_classification(test_y, y_pred, y_proba)

    return {
        "model_name": "mlp",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_mlp_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Run MLP model with default parameters.

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

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing MLP model results with 'model_name', metrics, 'y_pred', 'y_proba'.
    """
    return train_mlp(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    print("MLP classification model loaded.")

