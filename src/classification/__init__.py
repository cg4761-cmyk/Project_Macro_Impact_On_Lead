"""
Classification module for lead price prediction.

This module contains various classification models organized by type:
- Baseline linear models (Logistic Regression, Ridge Classification)
- Tree-based models (Random Forest)
- SVM and KNN models
- MLP (Multi-Layer Perceptron)
- RNN (LSTM)
"""

# Baseline linear models
from .clf_baseline_linear import (
    evaluate_classification,
    train_logistic_regression,
    train_ridge_classification,
    run_baseline_models,
)

# Tree-based models
from .clf_tree_models import (
    train_random_forest,
    run_tree_models,
)

# SVM and KNN models
from .clf_svm_knn import (
    train_svm,
    train_knn,
    run_svm_knn_models,
)

# MLP model
from .clf_mlp import (
    train_mlp,
    run_mlp_model,
)

# RNN model
from .clf_rnn import (
    create_sequences,
    train_rnn,
    run_rnn_model,
)

__all__ = [
    # Evaluation
    "evaluate_classification",
    # Baseline linear
    "train_logistic_regression",
    "train_ridge_classification",
    "run_baseline_models",
    # Tree-based
    "train_random_forest",
    "run_tree_models",
    # SVM and KNN
    "train_svm",
    "train_knn",
    "run_svm_knn_models",
    # MLP
    "train_mlp",
    "run_mlp_model",
    # RNN
    "create_sequences",
    "train_rnn",
    "run_rnn_model",
]
