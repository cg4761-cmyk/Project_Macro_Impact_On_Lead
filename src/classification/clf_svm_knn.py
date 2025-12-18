"""
SVM and KNN classification models for lead price prediction.

This module implements:
1. Support Vector Machine (SVM) Classification
2. K-Nearest Neighbors (KNN) Classification

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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


def train_svm(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    standardize: bool = True,
    class_weight: str = 'balanced',
    random_state: int = 42,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate an SVM classification model.

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
    kernel : str, default='rbf'
        Specifies the kernel type. Options: 'linear', 'poly', 'rbf', 'sigmoid'.
    C : float, default=1.0
        Regularization parameter. Strength of regularization is inversely proportional to C.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()).
    standardize : bool, default=True
        Whether to standardize features using StandardScaler.
    class_weight : str or dict, default='balanced'
        Weights associated with classes. 'balanced' adjusts weights inversely proportional to class frequencies.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'svm'
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

    # Set numpy random seed for additional reproducibility
    np.random.seed(random_state)

    # Standardize features if requested (important for SVM)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    # Train SVM model
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,  # Enable probability estimates for AUC
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(train_X, train_y)

    # Make predictions
    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)

    # Evaluate
    metrics = evaluate_classification(test_y, y_pred, y_proba)

    return {
        "model_name": "svm",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_knn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_neighbors: int = 5,
    weights: str = 'uniform',
    algorithm: str = 'auto',
    standardize: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a KNN classification model.

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
    n_neighbors : int, default=5
        Number of neighbors to use for classification.
    weights : str, default='uniform'
        Weight function used in prediction. 'uniform' or 'distance'.
    algorithm : str, default='auto'
        Algorithm used to compute nearest neighbors.
    standardize : bool, default=True
        Whether to standardize features using StandardScaler (important for KNN).

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'knn'
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

    # Set numpy random seed for reproducibility (KNN is deterministic but numpy operations might vary)
    np.random.seed(42)

    # Standardize features if requested (important for KNN)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

    # Workaround for Windows threadpoolctl issue
    # The AttributeError: 'NoneType' object has no attribute 'split' occurs
    # when threadpoolctl tries to detect thread libraries on Windows during sklearn operations
    # Solution: Use 'brute' algorithm on Windows which avoids threadpool detection
    import platform
    knn_algorithm = algorithm
    if platform.system() == 'Windows':
        # On Windows, default to 'brute' to avoid threadpoolctl issues
        # 'brute' doesn't use advanced optimizations that trigger threadpool detection
        if algorithm == 'auto':
            knn_algorithm = 'brute'
        elif algorithm not in ['brute', 'ball_tree', 'kd_tree']:
            knn_algorithm = 'brute'  # Fallback to brute for unknown algorithms
    
    # Train KNN model
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=knn_algorithm,
    )
    
    try:
        model.fit(train_X, train_y)
        # Make predictions
        y_pred = model.predict(test_X)
        y_proba = model.predict_proba(test_X)
    except (AttributeError, TypeError) as e:
        error_str = str(e)
        # Check if it's the threadpoolctl issue
        if "'NoneType' object has no attribute 'split'" in error_str or "split" in error_str.lower():
            # Retry with brute algorithm which doesn't use threadpool detection
            import warnings
            warnings.warn(f"KNN encountered threadpoolctl issue ({type(e).__name__}). Retrying with 'brute' algorithm.")
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm='brute',  # Brute force avoids threadpool issues
            )
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)
            y_proba = model.predict_proba(test_X)
        else:
            # Re-raise if it's a different error
            raise

    # Evaluate
    metrics = evaluate_classification(test_y, y_pred, y_proba)

    return {
        "model_name": "knn",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_svm_knn_models(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> List[Dict[str, Union[str, float, np.ndarray]]]:
    """
    Run SVM and KNN classification models and return their results.

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
        - First element: SVM classification results
        - Second element: KNN classification results
    """
    results = []

    # Train SVM
    svm_results = train_svm(train_X, train_y, test_X, test_y)
    results.append(svm_results)

    # Train KNN
    knn_results = train_knn(train_X, train_y, test_X, test_y)
    results.append(knn_results)

    return results


if __name__ == "__main__":
    print("SVM and KNN classification models loaded.")

