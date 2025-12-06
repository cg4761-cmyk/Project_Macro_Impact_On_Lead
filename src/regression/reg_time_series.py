"""
Time series regression models for lead price prediction.

This module implements classical time series models:
1. ARIMA (AutoRegressive Integrated Moving Average)
2. SARIMA (Seasonal AutoRegressive Integrated Moving Average)

These models are designed for univariate time series forecasting and serve as
traditional statistical baselines for comparison with machine learning models.

All models return standardized result dictionaries with predictions and metrics.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


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


def _find_best_arima_params(
    train_y: np.ndarray, max_p: int = 5, max_d: int = 2, max_q: int = 5
) -> Tuple[int, int, int]:
    """
    Find best ARIMA parameters using grid search with AIC.

    Parameters
    ----------
    train_y : np.ndarray
        Training target values.
    max_p : int, default=5
        Maximum value for AR order.
    max_d : int, default=2
        Maximum value for differencing order.
    max_q : int, default=5
        Maximum value for MA order.

    Returns
    -------
    Tuple[int, int, int]
        Best (p, d, q) parameters.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
    except ImportError:
        raise ImportError(
            "statsmodels is required for ARIMA models. Install it using: pip install statsmodels"
        )

    best_aic = np.inf
    best_params = (1, 1, 1)

    # Suppress convergence warnings during parameter search
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(train_y, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_params = (p, d, q)
                    except:
                        continue

    return best_params


def train_arima(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    order: Optional[Tuple[int, int, int]] = None,
    auto_select: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate an ARIMA model.

    Note: ARIMA is a univariate model, so it only uses train_y and test_y,
    ignoring train_X and test_X features.

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix (not used by ARIMA, kept for interface consistency).
    train_y : np.ndarray
        Training target vector of shape (n_samples,).
    test_X : np.ndarray
        Test feature matrix (not used by ARIMA, kept for interface consistency).
    test_y : np.ndarray
        Test target vector of shape (n_samples_test,).
    order : Tuple[int, int, int], default=None
        ARIMA order (p, d, q). If None and auto_select=True, will be auto-selected.
    auto_select : bool, default=True
        Whether to automatically select best (p, d, q) parameters using AIC.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'arima'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set
            - 'order': ARIMA order (p, d, q) used

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    ImportError
        If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError(
            "statsmodels is required for ARIMA models. Install it using: pip install statsmodels"
        )

    # Input validation
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    if len(train_y) == 0 or len(test_y) == 0:
        raise ValueError("Training or test set cannot be empty")

    if len(train_y.shape) > 1 and train_y.shape[1] != 1:
        raise ValueError("train_y must be a 1D array or column vector")

    if len(test_y.shape) > 1 and test_y.shape[1] != 1:
        raise ValueError("test_y must be a 1D array or column vector")

    # Flatten y arrays if needed
    train_y = train_y.flatten()
    test_y = test_y.flatten()

    # Auto-select order if requested
    if order is None and auto_select:
        order = _find_best_arima_params(train_y)
    elif order is None:
        order = (1, 1, 1)  # Default order

    # Train ARIMA model (suppress warnings during fitting)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        model = ARIMA(train_y, order=order)
        fitted_model = model.fit()

    # Make predictions
    # Get forecasts for the test period
    forecast = fitted_model.forecast(steps=len(test_y))
    y_pred = np.array(forecast)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "arima",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
        "order": order,
    }


def _find_best_sarima_params(
    train_y: np.ndarray,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 2,
    max_D: int = 1,
    max_Q: int = 2,
    seasonal_periods: int = 12,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Find best SARIMA parameters using grid search with AIC.

    Parameters
    ----------
    train_y : np.ndarray
        Training target values.
    max_p : int, default=3
        Maximum value for AR order.
    max_d : int, default=2
        Maximum value for differencing order.
    max_q : int, default=3
        Maximum value for MA order.
    max_P : int, default=2
        Maximum value for seasonal AR order.
    max_D : int, default=1
        Maximum value for seasonal differencing order.
    max_Q : int, default=2
        Maximum value for seasonal MA order.
    seasonal_periods : int, default=12
        Number of seasonal periods.

    Returns
    -------
    Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]
        Best ((p, d, q), (P, D, Q, s)) parameters.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
    except ImportError:
        raise ImportError(
            "statsmodels is required for SARIMA models. Install it using: pip install statsmodels"
        )

    best_aic = np.inf
    best_params = ((1, 1, 1), (1, 1, 1, seasonal_periods))

    # Suppress convergence warnings during parameter search
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        # Limit search space to avoid excessive computation
        for p in range(min(max_p + 1, 3)):
            for d in range(min(max_d + 1, 2)):
                for q in range(min(max_q + 1, 3)):
                    for P in range(min(max_P + 1, 2)):
                        for D in range(min(max_D + 1, 2)):
                            for Q in range(min(max_Q + 1, 2)):
                                try:
                                    model = SARIMAX(
                                        train_y,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_periods),
                                    )
                                    fitted_model = model.fit()
                                    aic = fitted_model.aic
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (
                                            (p, d, q),
                                            (P, D, Q, seasonal_periods),
                                        )
                                except:
                                    continue

    return best_params


def train_sarima(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    seasonal_periods: int = 12,
    auto_select: bool = True,
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Train and evaluate a SARIMA (Seasonal ARIMA) model.

    Note: SARIMA is a univariate model, so it only uses train_y and test_y,
    ignoring train_X and test_X features.

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix (not used by SARIMA, kept for interface consistency).
    train_y : np.ndarray
        Training target vector of shape (n_samples,).
    test_X : np.ndarray
        Test feature matrix (not used by SARIMA, kept for interface consistency).
    test_y : np.ndarray
        Test target vector of shape (n_samples_test,).
    order : Tuple[int, int, int], default=None
        ARIMA order (p, d, q). If None and auto_select=True, will be auto-selected.
    seasonal_order : Tuple[int, int, int, int], default=None
        Seasonal order (P, D, Q, s). If None, will use (1, 1, 1, seasonal_periods).
    seasonal_periods : int, default=12
        Number of seasonal periods. Default 12 for monthly data.
    auto_select : bool, default=True
        Whether to automatically select best parameters using AIC.

    Returns
    -------
    Dict[str, Union[str, float, np.ndarray]]
        Dictionary containing:
            - 'model_name': 'sarima'
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'y_pred': Predicted values on test set
            - 'order': ARIMA order (p, d, q) used
            - 'seasonal_order': Seasonal order (P, D, Q, s) used

    Raises
    ------
    ValueError
        If input shapes are invalid or arrays are empty.
    ImportError
        If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError(
            "statsmodels is required for SARIMA models. Install it using: pip install statsmodels"
        )

    # Input validation
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    if len(train_y) == 0 or len(test_y) == 0:
        raise ValueError("Training or test set cannot be empty")

    if len(train_y.shape) > 1 and train_y.shape[1] != 1:
        raise ValueError("train_y must be a 1D array or column vector")

    if len(test_y.shape) > 1 and test_y.shape[1] != 1:
        raise ValueError("test_y must be a 1D array or column vector")

    # Flatten y arrays if needed
    train_y = train_y.flatten()
    test_y = test_y.flatten()

    # Auto-select parameters if requested
    if auto_select and (order is None or seasonal_order is None):
        best_order, best_seasonal = _find_best_sarima_params(
            train_y, seasonal_periods=seasonal_periods
        )
        if order is None:
            order = best_order
        if seasonal_order is None:
            seasonal_order = best_seasonal
    else:
        if order is None:
            order = (1, 1, 1)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, seasonal_periods)

    # Train SARIMA model (suppress warnings during fitting)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning)
        model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()

    # Make predictions
    # Get forecasts for the test period
    forecast = fitted_model.forecast(steps=len(test_y))
    y_pred = np.array(forecast)

    # Evaluate
    metrics = evaluate_regression(test_y, y_pred)

    return {
        "model_name": "sarima",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "y_pred": y_pred,
        "order": order,
        "seasonal_order": seasonal_order,
    }


def run_time_series_models(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    include_sarima: bool = True,
    seasonal_periods: int = 12,
) -> List[Dict[str, Union[str, float, np.ndarray]]]:
    """
    Run time series regression models and return their results.

    Parameters
    ----------
    train_X : np.ndarray
        Training feature matrix (not used by time series models, kept for interface consistency).
    train_y : np.ndarray
        Training target vector of shape (n_samples,).
    test_X : np.ndarray
        Test feature matrix (not used by time series models, kept for interface consistency).
    test_y : np.ndarray
        Test target vector of shape (n_samples_test,).
    include_sarima : bool, default=True
        Whether to include SARIMA model. If False, only ARIMA is trained.
    seasonal_periods : int, default=12
        Seasonal periods for SARIMA model. Default 12 for monthly data.

    Returns
    -------
    List[Dict[str, Union[str, float, np.ndarray]]]
        List of result dictionaries, one for each model:
        - First element: ARIMA regression results
        - Second element: SARIMA regression results (if include_sarima=True)
    """
    results = []

    # Train ARIMA
    arima_results = train_arima(train_X, train_y, test_X, test_y)
    results.append(arima_results)

    # Train SARIMA if requested
    if include_sarima:
        sarima_results = train_sarima(
            train_X, train_y, test_X, test_y, seasonal_periods=seasonal_periods
        )
        results.append(sarima_results)

    return results


if __name__ == "__main__":
    print("Time series regression models (ARIMA & SARIMA) loaded.")

