# src/imbalance_utils.py - Advanced class imbalance handling techniques
# -----------------------------------------------------------------------------
# This file contains utilities for handling class imbalance in your sleep-depression
# prediction model. Import and use these functions in your main training script.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


def enhanced_sampling_strategies(X_train, y_train):
    """Try multiple sampling strategies and compare"""

    strategies = {
        'smote': SMOTE(random_state=42, k_neighbors=3),
        'adasyn': ADASYN(random_state=42, n_neighbors=3),
        'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=3),
        'smote_tomek': SMOTETomek(random_state=42),
        'smote_enn': SMOTEENN(random_state=42)
    }

    return strategies


# Focal Loss for severe class imbalance
class FocalLoss:
    def __init__(self, alpha=1, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1 / (1 + np.exp(-y_pred))

        # Focal loss calculation
        focal_weight = self.alpha * (1 - p) ** self.gamma
        loss = -focal_weight * np.log(p + 1e-8)

        grad = focal_weight * (p - y_true)
        hess = focal_weight * p * (1 - p) * (1 - self.gamma * (1 - 2 * p))

        return grad, hess


# Cost-sensitive learning
def get_cost_sensitive_weights(y_train):
    """Calculate class weights inversely proportional to class frequencies"""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def evaluate_sampling_strategies(X_train, y_train, model_class, model_params, cv=5):
    """
    Compare different sampling strategies using cross-validation
    Returns the best strategy based on AUPRC
    """

    strategies = enhanced_sampling_strategies(X_train, y_train)
    results = {}

    print("🔍 Evaluating sampling strategies...")
    print("=" * 50)

    # Add baseline (no sampling)
    strategies['baseline'] = None

    for name, sampler in strategies.items():
        print(f"Testing {name}...")

        if sampler is None:
            # Baseline without sampling
            model = model_class(**model_params)
            auroc_scores = cross_val_score(model, X_train, y_train,
                                           cv=cv, scoring='roc_auc')
            auprc_scores = cross_val_score(model, X_train, y_train,
                                           cv=cv, scoring='average_precision')
        else:
            # With sampling
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('model', model_class(**model_params))
            ])

            auroc_scores = cross_val_score(pipeline, X_train, y_train,
                                           cv=cv, scoring='roc_auc')
            auprc_scores = cross_val_score(pipeline, X_train, y_train,
                                           cv=cv, scoring='average_precision')

        results[name] = {
            'auroc_mean': auroc_scores.mean(),
            'auroc_std': auroc_scores.std(),
            'auprc_mean': auprc_scores.mean(),
            'auprc_std': auprc_scores.std()
        }

        print(f"  AUROC: {auroc_scores.mean():.3f} ± {auroc_scores.std():.3f}")
        print(f"  AUPRC: {auprc_scores.mean():.3f} ± {auprc_scores.std():.3f}")
        print()

    # Find best strategy based on AUPRC
    best_strategy = max(results.keys(), key=lambda k: results[k]['auprc_mean'])

    print(f"🏆 Best strategy: {best_strategy}")
    print(f"   AUPRC: {results[best_strategy]['auprc_mean']:.3f}")
    print(f"   AUROC: {results[best_strategy]['auroc_mean']:.3f}")

    return best_strategy, results


def apply_best_sampling(X_train, y_train, strategy_name):
    """Apply the best sampling strategy to training data"""

    if strategy_name == 'baseline':
        return X_train, y_train

    strategies = enhanced_sampling_strategies(X_train, y_train)
    sampler = strategies[strategy_name]

    print(f"🔄 Applying {strategy_name} sampling...")
    print(f"Original distribution: {np.bincount(y_train)}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"Resampled distribution: {np.bincount(y_resampled)}")
    print(f"Original samples: {len(y_train)}")
    print(f"Resampled samples: {len(y_resampled)}")

    return X_resampled, y_resampled


def get_auprc_optimized_params():
    """
    Get hyperparameter ranges optimized for AUPRC instead of AUROC
    These ranges work better for imbalanced data
    """

    params = {
        # More conservative learning to avoid overfitting to majority class
        "learning_rate": (0.01, 0.1),  # Lower range

        # Deeper trees can help capture minority class patterns
        "max_depth": (4, 10),  # Slightly deeper

        # More estimators for complex imbalanced patterns
        "n_estimators": (200, 800),  # Higher range

        # Higher subsample to ensure minority class representation
        "subsample": (0.7, 1.0),  # Higher minimum

        # More feature sampling for diversity
        "colsample_bytree": (0.5, 1.0),

        # Lower gamma to allow more splits
        "gamma": (0.0, 2.0),  # Lower range

        # Lower min_child_weight for minority class
        "min_child_weight": (1, 5),  # Lower range

        # L1 regularization helps with feature selection
        "reg_alpha": (0.0, 1.0),

        # L2 regularization for generalization
        "reg_lambda": (0.0, 1.0),
    }

    return params


def create_auprc_objective(sampler=None):
    """
    Create Optuna objective function optimized for AUPRC
    """

    def objective(trial, X_train, y_train, X_val, y_val):
        """Optuna objective optimized for AUPRC"""

        # Get AUPRC-optimized parameter ranges
        param_ranges = get_auprc_optimized_params()

        params = {
            "learning_rate": trial.suggest_float("learning_rate", *param_ranges["learning_rate"], log=True),
            "max_depth": trial.suggest_int("max_depth", *param_ranges["max_depth"]),
            "n_estimators": trial.suggest_int("n_estimators", *param_ranges["n_estimators"], step=50),
            "subsample": trial.suggest_float("subsample", *param_ranges["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *param_ranges["colsample_bytree"]),
            "gamma": trial.suggest_float("gamma", *param_ranges["gamma"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *param_ranges["min_child_weight"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *param_ranges["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *param_ranges["reg_lambda"]),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # Focus on AUPRC
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 4,
        }

        # Calculate scale_pos_weight
        neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        params["scale_pos_weight"] = neg_pos_ratio

        # Apply sampling if provided
        if sampler is not None:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train model
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Return AUPRC (what we want to maximize)
        auprc = average_precision_score(y_val, y_pred_proba)

        return auprc

    return objective


def print_class_distribution(y, title="Class Distribution"):
    """Print detailed class distribution information"""

    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\n📊 {title}")
    print("-" * 30)
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"Class {cls}: {count:,} samples ({percentage:.1f}%)")

    if len(unique) == 2:
        ratio = counts[0] / counts[1] if counts[1] > 0 else float('inf')
        print(f"Imbalance ratio: {ratio:.1f}:1")