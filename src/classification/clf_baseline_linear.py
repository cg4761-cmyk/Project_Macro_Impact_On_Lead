"""
Baseline linear classification models for lead price prediction.

This module implements baseline classification models:
1. Logistic Regression
2. Ridge Classification with cross-validation

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


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


def train_logistic_regression(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    standardize: bool = True,
    class_weight: str = 'balanced',
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a logistic regression model.

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
    class_weight : str or dict, default='balanced'
        Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'logistic'
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

    # Train logistic regression model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=class_weight
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)

    # Evaluate
    metrics = evaluate_classification(test_y, y_pred, y_proba)

    return {
        "model_name": "logistic",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_ridge_classification(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    alphas: List[float] = None,
    standardize: bool = True,
    class_weight: str = 'balanced',
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a Ridge classification model with cross-validation.

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
    class_weight : str or dict, default='balanced'
        Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'ridge'
            - 'accuracy': Accuracy score
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score
            - 'auc': AUC-ROC score (may be NaN as RidgeClassifier doesn't provide probabilities by default)
            - 'y_pred': Predicted classes on test set
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

    # Train Ridge classification with cross-validation
    model = RidgeClassifierCV(
        alphas=alphas,
        cv=5,
        class_weight=class_weight
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)
    
    # RidgeClassifier doesn't provide probability estimates by default
    # Use decision function as proxy (not true probabilities)
    y_proba = None

    # Evaluate
    metrics = evaluate_classification(test_y, y_pred, y_proba)

    return {
        "model_name": "ridge",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
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
    Run both baseline classification models and return their results.

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
        - First element: Logistic regression results
        - Second element: Ridge classification results
    """
    # Train both models
    logistic_results = train_logistic_regression(train_X, train_y, test_X, test_y)
    ridge_results = train_ridge_classification(train_X, train_y, test_X, test_y)

    return [logistic_results, ridge_results]


if __name__ == "__main__":
    print("Baseline classification models (Logistic & Ridge) loaded.")

