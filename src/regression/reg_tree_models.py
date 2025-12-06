"""
Tree-based regression models for lead price prediction.

This module implements tree-based regression models:
1. Random Forest Regression
2. XGBoost Regression

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.ensemble import RandomForestRegressor
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


def train_random_forest(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a Random Forest regression model.

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
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    random_state : int, default=42
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means use all processors.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'random_forest'
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

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "random_forest",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
    }


def train_xgboost(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate an XGBoost regression model.

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
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    learning_rate : float, default=0.1
        Boosting learning rate.
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'xgboost'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    ImportError
        If xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for this function. Install it using: pip install xgboost"
        )

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

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "xgboost",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
    }


def run_tree_models(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    include_xgboost: bool = True,
) -> List[Dict[str, Union[str, float, np.ndarray]]]:
    """
    Run tree-based regression models and return their results.

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
    include_xgboost : bool, default=True
        Whether to include XGBoost model. If False, only Random Forest is trained.

    Returns
    -------
    List[Dict[str, Union[str, float, np.ndarray]]]
        List of result dictionaries, one for each model:
        - First element: Random Forest regression results
        - Second element: XGBoost regression results (if include_xgboost=True)
    """
    results = []

    # Train Random Forest
    rf_results = train_random_forest(train_X, train_y, test_X, test_y)
    results.append(rf_results)

    # Train XGBoost if requested
    if include_xgboost:
        try:
            xgb_results = train_xgboost(train_X, train_y, test_X, test_y)
            results.append(xgb_results)
        except ImportError:
            print(
                "Warning: XGBoost not available. Skipping XGBoost model. "
                "Install it using: pip install xgboost"
            )

    return results


if __name__ == "__main__":
    print("Tree-based regression models (Random Forest & XGBoost) loaded.")

