"""
Train-test split for time series data using walk-forward validation.

This module provides functions for splitting time series data while preserving
temporal order, essential for time series forecasting tasks.
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    test_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into training and testing sets while preserving temporal order.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    test_size : float, default=None
        Proportion of dataset to include in the test split. Must be between 0 and 1.
        If None, will use test_samples or default to 0.2.
    train_size : float, default=None
        Proportion of dataset to include in the train split. Must be between 0 and 1.
        If None, will be inferred from test_size.
    test_samples : int, default=None
        Number of samples to include in the test set. If specified, takes precedence over test_size.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        train_X, train_y, test_X, test_y

    Raises
    ------
    ValueError
        If input shapes are invalid or parameters are inconsistent.

    Examples
    --------
    >>> X = np.random.rand(1000, 10)
    >>> y = np.random.rand(1000)
    >>> train_X, train_y, test_X, test_y = train_test_split_time_series(X, y, test_size=0.2)
    >>> len(train_X), len(test_X)
    (800, 200)
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
        )

    if len(X) == 0:
        raise ValueError("Input arrays cannot be empty")

    n_samples = len(X)

    # Determine test set size
    if test_samples is not None:
        if test_samples <= 0 or test_samples >= n_samples:
            raise ValueError(
                f"test_samples must be between 1 and {n_samples-1}. Got {test_samples}"
            )
        test_len = test_samples
    elif test_size is not None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")
        test_len = int(n_samples * test_size)
    elif train_size is not None:
        if not 0 < train_size < 1:
            raise ValueError(f"train_size must be between 0 and 1. Got {train_size}")
        test_len = n_samples - int(n_samples * train_size)
    else:
        # Default to 20% test set
        test_len = int(n_samples * 0.2)

    # Ensure at least one sample in each set
    if test_len >= n_samples:
        test_len = n_samples - 1
    if test_len < 1:
        test_len = 1

    train_len = n_samples - test_len

    # Split while preserving temporal order
    train_X = X[:train_len]
    train_y = y[:train_len]
    test_X = X[train_len:]
    test_y = y[train_len:]

    return train_X, train_y, test_X, test_y


def walk_forward_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: Optional[float] = None,
    test_samples: Optional[int] = None,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate multiple train-test splits using walk-forward validation.

    This creates multiple splits where the training set grows over time
    and the test set moves forward. Suitable for time series cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    n_splits : int, default=5
        Number of splits to generate. Must be at least 2.
    test_size : float, default=None
        Proportion of each test set relative to total samples. Must be between 0 and 1.
        If None, will use test_samples or default to 0.2 / n_splits.
    test_samples : int, default=None
        Number of samples in each test set. If specified, takes precedence over test_size.
    gap : int, default=0
        Gap between training and test sets (to avoid data leakage).
        Number of samples to skip between train and test.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of (train_X, train_y, test_X, test_y) tuples, one for each split.

    Examples
    --------
    >>> X = np.random.rand(1000, 10)
    >>> y = np.random.rand(1000)
    >>> splits = walk_forward_validation_split(X, y, n_splits=3, test_samples=100)
    >>> len(splits)
    3
    >>> len(splits[0][2])  # test_X size
    100
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
        )

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2. Got {n_splits}")

    n_samples = len(X)

    # Determine test set size per split
    if test_samples is not None:
        if test_samples <= 0:
            raise ValueError(f"test_samples must be positive. Got {test_samples}")
        test_len = test_samples
    elif test_size is not None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")
        test_len = int(n_samples * test_size / n_splits)
    else:
        # Default: each test set is 20% / n_splits of total
        test_len = max(1, int(n_samples * 0.2 / n_splits))

    # Calculate total test samples needed
    total_test_needed = n_splits * test_len + (n_splits - 1) * gap
    min_train_size = max(1, int(n_samples * 0.5))  # At least 50% for initial training

    if total_test_needed + min_train_size > n_samples:
        warnings.warn(
            f"Requested splits require {total_test_needed + min_train_size} samples, "
            f"but only {n_samples} available. Adjusting test_len and n_splits."
        )
        # Adjust test_len to fit available data
        available_for_test = n_samples - min_train_size
        test_len = max(1, available_for_test // (n_splits + gap * (n_splits - 1) / test_len))

    splits = []
    current_train_end = min_train_size

    for i in range(n_splits):
        # Training set: from start to current_train_end
        train_X = X[:current_train_end]
        train_y = y[:current_train_end]

        # Test set: after gap
        test_start = current_train_end + gap
        test_end = test_start + test_len

        if test_end > n_samples:
            # If not enough samples, use remaining data
            test_end = n_samples
            if test_start >= test_end:
                # No more data for test sets
                break

        test_X = X[test_start:test_end]
        test_y = y[test_start:test_end]

        if len(test_X) == 0:
            warnings.warn(f"Split {i+1} has empty test set. Stopping.")
            break

        splits.append((train_X, train_y, test_X, test_y))

        # Move forward: next train set includes current test set
        current_train_end = test_end

    if len(splits) < n_splits:
        warnings.warn(
            f"Only generated {len(splits)} splits instead of requested {n_splits}."
        )

    return splits


def create_walk_forward_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_size: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create sliding windows for walk-forward validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    window_size : int
        Size of each window.
    step_size : int, default=1
        Step size between windows.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (X_window, y_window) tuples.

    Examples
    --------
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.rand(100)
    >>> windows = create_walk_forward_windows(X, y, window_size=20, step_size=10)
    >>> len(windows)
    9
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
        )

    if window_size <= 0 or window_size > len(X):
        raise ValueError(
            f"window_size must be between 1 and {len(X)}. Got {window_size}"
        )

    windows = []
    for i in range(0, len(X) - window_size + 1, step_size):
        X_window = X[i : i + window_size]
        y_window = y[i : i + window_size]
        windows.append((X_window, y_window))

    return windows


if __name__ == "__main__":
    print("Time series train-test split utilities loaded.")
    print("\nAvailable functions:")
    print("  - train_test_split_time_series: Simple time-ordered split")
    print("  - walk_forward_validation_split: Multiple walk-forward splits")
    print("  - create_walk_forward_windows: Sliding window creation")

