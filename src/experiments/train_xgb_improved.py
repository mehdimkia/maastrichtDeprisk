# src/train_xgb_improved.py - Enhanced training with class imbalance handling
# -----------------------------------------------------------------------------
# This is an improved version of your train_xgb_server.py that incorporates
# advanced class imbalance handling techniques for better AUPRC performance
# -----------------------------------------------------------------------------

import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Import our new imbalance utilities
from imbalance_utils import (
    evaluate_sampling_strategies,
    apply_best_sampling,
    create_auprc_objective,
    print_class_distribution,
    get_cost_sensitive_weights
)

# ------------------------------------------------------------------
# Paths - Update BASE to match your actual data location
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")

# If the above doesn't work, try these alternatives:
if not BASE.exists():
    alternative_paths = [
        Path("/Users/mkia/Downloads/SummerPredectiveModel/maastrichtDeprisk/Databases"),
        Path("../Databases"),
        Path("./Databases")
    ]

    for alt_path in alternative_paths:
        if alt_path.exists():
            print(f"✅ Using alternative path: {alt_path}")
            BASE = alt_path
            break
    else:
        print("❌ Cannot find Databases directory. Please update BASE variable.")
        print("Current directory:", Path.cwd())
        exit(1)

TRAIN_PQ = BASE / "Week7_train_interactions.parquet"
TEST_PQ = BASE / "Week7_test_interactions.parquet"

# New output file name to avoid overwriting
OUT_PIPE = BASE / "Week7_xgb_improved_imbalance.joblib"

N_TRIALS = 50
RANDOM_SEED = 42

# Load data
print("📁 Loading data...")
train_df = pd.read_parquet(TRAIN_PQ)
test_df = pd.read_parquet(TEST_PQ)
y_train = train_df.pop("LD_PHQ9depr_event")
y_test = test_df.pop("LD_PHQ9depr_event")

# Print initial class distribution
print_class_distribution(y_train, "Training Set - Original Distribution")
print_class_distribution(y_test, "Test Set Distribution")

# Feature preprocessing (same as your original)
cat_cols = train_df.select_dtypes("category").columns.tolist()

spline = SplineTransformer(
    degree=3,
    knots=np.array([[360], [420], [480], [540], [600]]),  # 6h, 7h, 8h, 9h, 10h
    include_bias=False,
)

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("sleep_spline", spline, ["sleep_minutes"]),
    ],
    remainder="passthrough",
    sparse_threshold=0.3,
)

# Transform features
print("🔧 Preprocessing features...")
X_train_processed = preprocessor.fit_transform(train_df)
X_test_processed = preprocessor.transform(test_df)

# Get feature names after fitting
print("🏷️ Generating feature names...")
try:
    # Get categorical feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()

    # Clean categorical feature names - remove problematic characters for XGBoost
    cat_feature_names_clean = []
    for name in cat_feature_names:
        # Replace problematic characters
        clean_name = name.replace('[', '_').replace(']', '_').replace('<', 'lt').replace('>', 'gt').replace(' ', '_')
        cat_feature_names_clean.append(clean_name)

    # Get spline feature names (now that it's fitted)
    n_spline_features = preprocessor.named_transformers_['sleep_spline'].n_features_out_
    spline_feature_names = [f'sleep_spline_{i}' for i in range(n_spline_features)]

    # Get remaining feature names (passthrough)
    remaining_cols = [col for col in train_df.columns if col not in cat_cols and col != 'sleep_minutes']

    # Combine all feature names
    feature_names = cat_feature_names_clean + spline_feature_names + remaining_cols

    print(f"✅ Cleaned {len(cat_feature_names)} categorical feature names")

except Exception as e:
    print(f"⚠️ Feature naming issue: {e}")
    print("Using generic feature names...")
    n_features = X_train_processed.shape[1]
    feature_names = [f'feature_{i}' for i in range(n_features)]

print(f"✅ Created {len(feature_names)} feature names")

# Convert to DataFrame for easier handling with imbalance techniques
if hasattr(X_train_processed, 'toarray'):
    X_train_array = X_train_processed.toarray()
    X_test_array = X_test_processed.toarray()
else:
    X_train_array = X_train_processed
    X_test_array = X_test_processed

# Use generic feature names for XGBoost compatibility
generic_feature_names = [f'feature_{i}' for i in range(X_train_array.shape[1])]

X_train_df = pd.DataFrame(X_train_array, columns=generic_feature_names)
X_test_df = pd.DataFrame(X_test_array, columns=generic_feature_names)

print(f"✅ Training set shape: {X_train_df.shape}")
print(f"✅ Test set shape: {X_test_df.shape}")

# Keep track of original feature names for interpretation (optional)
feature_name_mapping = dict(zip(generic_feature_names, feature_names[:len(generic_feature_names)]))
print("✅ Created feature name mapping for interpretation")

# Step 1: Evaluate different sampling strategies
print("\n🎯 Step 1: Evaluating sampling strategies...")
print("=" * 60)

# Create a validation split for strategy evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_df, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
)

# Basic XGBoost parameters for strategy evaluation
basic_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'random_state': RANDOM_SEED,
    'n_jobs': 4,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()
}

# Evaluate sampling strategies
best_strategy, strategy_results = evaluate_sampling_strategies(
    X_train_split, y_train_split, XGBClassifier, basic_params, cv=3
)

# Step 2: Apply best sampling strategy
print(f"\n🔄 Step 2: Applying best strategy: {best_strategy}")
print("=" * 60)

if best_strategy != 'baseline':
    from imbalance_utils import enhanced_sampling_strategies

    strategies = enhanced_sampling_strategies(X_train_df, y_train)
    best_sampler = strategies[best_strategy]
    X_train_resampled, y_train_resampled = apply_best_sampling(X_train_df, y_train, best_strategy)
else:
    X_train_resampled, y_train_resampled = X_train_df, y_train
    best_sampler = None

# Step 3: Hyperparameter optimization with AUPRC focus
print(f"\n⚡ Step 3: Hyperparameter optimization (AUPRC-focused)")
print("=" * 60)


def objective(trial):
    """Optuna objective optimized for AUPRC"""

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
    }

    # Calculate scale_pos_weight based on resampled data
    neg_pos_ratio = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
    params["scale_pos_weight"] = neg_pos_ratio

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict on original test set (not resampled)
    y_pred_proba = model.predict_proba(X_test_df)[:, 1]

    # Return AUPRC (what we want to maximize)
    auprc = average_precision_score(y_test, y_pred_proba)

    return auprc


# Run optimization
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n🏆 Best trial AUPRC: {study.best_value:.3f}")
print(f"Best parameters: {study.best_params}")

# Step 4: Train final model with best parameters
print(f"\n🎓 Step 4: Training final model")
print("=" * 60)

# Get best parameters and add fixed settings
best_params = study.best_params.copy()
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
})

# Calculate scale_pos_weight for resampled data
neg_pos_ratio = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
best_params["scale_pos_weight"] = neg_pos_ratio

# Train final model
final_model = XGBClassifier(**best_params)
final_model.fit(X_train_resampled, y_train_resampled)

# Step 5: Final evaluation
print(f"\n📊 Step 5: Final Evaluation")
print("=" * 60)

# Predictions
train_probs = final_model.predict_proba(X_train_df)[:, 1]
test_probs = final_model.predict_proba(X_test_df)[:, 1]

# Metrics
train_auroc = roc_auc_score(y_train, train_probs)
train_auprc = average_precision_score(y_train, train_probs)
test_auroc = roc_auc_score(y_test, test_probs)
test_auprc = average_precision_score(y_test, test_probs)

print(f"📈 TRAINING SCORES:")
print(f"   AUROC: {train_auroc:.3f}")
print(f"   AUPRC: {train_auprc:.3f}")
print(f"\n🎯 TEST SCORES:")
print(f"   AUROC: {test_auroc:.3f}")
print(f"   AUPRC: {test_auprc:.3f}")

# Improvement comparison (assuming your original scores)
original_auroc = 0.710
original_auprc = 0.294

print(f"\n📊 IMPROVEMENT vs ORIGINAL:")
print(f"   AUROC: {original_auroc:.3f} → {test_auroc:.3f} ({((test_auroc / original_auroc - 1) * 100):+.1f}%)")
print(f"   AUPRC: {original_auprc:.3f} → {test_auprc:.3f} ({((test_auprc / original_auprc - 1) * 100):+.1f}%)")

# Create final pipeline
if best_strategy != 'baseline':
    # Include sampling in the pipeline
    from imblearn.pipeline import Pipeline as ImbPipeline

    final_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler', best_sampler),
        ('model', final_model)
    ])
else:
    # Standard pipeline without sampling
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', final_model)
    ])

# Save results
print(f"\n💾 Saving results...")
joblib.dump(final_pipeline, OUT_PIPE)

# Save metadata
metadata = {
    "test_auroc": test_auroc,
    "test_auprc": test_auprc,
    "train_auroc": train_auroc,
    "train_auprc": train_auprc,
    "best_params": study.best_params,
    "best_sampling_strategy": best_strategy,
    "sampling_strategy_results": strategy_results,
    "improvement_auroc_pct": ((test_auroc / original_auroc - 1) * 100),
    "improvement_auprc_pct": ((test_auprc / original_auprc - 1) * 100),
    "original_auroc": original_auroc,
    "original_auprc": original_auprc
}

metadata_file = OUT_PIPE.with_suffix(".meta.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved to: {OUT_PIPE}")
print(f"✅ Metadata saved to: {metadata_file}")
print(f"\n🎉 Training complete! Expected AUPRC improvement: {((test_auprc / original_auprc - 1) * 100):+.1f}%")