"""
Baseline linear regression models for lead price prediction.

This module implements two baseline regression models:
1. Linear Regression
2. Ridge Regression with cross-validation

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler


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


def train_linear_regression(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    standardize: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a linear regression model.

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
    standardize : bool, default=True
        Whether to standardize features using StandardScaler.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'linear'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    """
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

    # Standardize features if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    # Train linear regression model
    model = LinearRegression()
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "linear",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
    }


def train_ridge_regression(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    alphas: List[float] = None,
    standardize: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a Ridge regression model with cross-validation.

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
    alphas : List[float], default=None
        List of regularization strengths. Default: [0.1, 1.0, 10.0, 100.0]
    standardize : bool, default=True
        Whether to standardize features using StandardScaler.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'ridge'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set
            - 'best_alpha': Best regularization parameter found by CV

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    """
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

    # Set default alphas if not provided
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]

    # Standardize features if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    # Train Ridge regression with cross-validation
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "ridge",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
        "best_alpha": float(model.alpha_),
    }


def run_baseline_models(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> List[Dict[str, Union[str, float, np.ndarray]]]:
    """
    Run both baseline regression models and return their results.

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
    List[Dict[str, Union[str, float, np.ndarray]]]
        List of result dictionaries, one for each model:
        - First element: Linear regression results
        - Second element: Ridge regression results
    """
    # Train both models
    linear_results = train_linear_regression(train_X, train_y, test_X, test_y)
    ridge_results = train_ridge_regression(train_X, train_y, test_X, test_y)

    return [linear_results, ridge_results]


if __name__ == "__main__":
    print("Baseline regression models (Linear & Ridge) loaded.")

