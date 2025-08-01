# src/train_xgb_server.py - Fixed LightGBM early stopping

import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer

from xgboost import XGBClassifier

# ------------------------------------------------------------------
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")

# Check if the base path exists, if not try alternatives
if not BASE.exists():
    alternative_paths = [
        Path("/Users/mkia/Downloads/SummerPredectiveModel/maastrichtDeprisk/Databases"),
        Path("../Databases"),
        Path("./Databases")
    ]
    for alt_path in alternative_paths:
        if alt_path.exists():
            BASE = alt_path
            break

#Training features with interactions included
TRAIN_PQ = BASE / "Week8_train_interactions.parquet"
TEST_PQ  = BASE / "Week8_test_interactions.parquet"

# >>> write the model under a new name so nothing gets overwritten
OUT_PIPE = BASE / "Week8_xgb_onehot_interactions.joblib"

N_TRIALS = 100
RANDOM_SEED = 42

train_df = pd.read_parquet(TRAIN_PQ)
test_df  = pd.read_parquet(TEST_PQ)
y_train  = train_df.pop("LD_PHQ9depr_event")
y_test   = test_df.pop("LD_PHQ9depr_event")

cat_cols = train_df.select_dtypes("category").columns.tolist()
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Get categorical feature indices
cat_idx = [train_df.columns.get_loc(c) for c in cat_cols]

# FIXED: Move early_stopping_rounds to the constructor
lgbm = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    metric="auc",
    num_leaves=128,
    learning_rate=0.03,
    n_estimators=1500,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=1,
    reg_lambda=2,
    is_unbalance=True,          # handles class skew
    early_stopping_rounds=100,  # ← MOVED HERE from fit()
    random_state=RANDOM_SEED,
    verbose=-1  # Reduce output verbosity
)

print("🔧  Training LightGBM …")

# FIXED: Remove early_stopping_rounds from fit() call
lgbm.fit(
    train_df, y_train,
    categorical_feature=cat_idx,
    eval_set=[(test_df, y_test)],
    eval_metric="aucpr"
    # early_stopping_rounds removed from here
)

probs_lgbm = lgbm.predict_proba(test_df)[:, 1]
print(f"🔍  LightGBM AUROC = {roc_auc_score(y_test, probs_lgbm):.3f}")
print(f"🔍  LightGBM AUPRC = {average_precision_score(y_test, probs_lgbm):.3f}")

# XGBoost preprocessing pipeline
spline = SplineTransformer(
    degree=3,
    knots=np.array([[360],   # 6 h
                    [420],   # 7 h
                    [480],   # 8 h
                    [540],   # 9 h
                    [600]]), #10 h
    include_bias=False,
)

pre = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore",
         sparse_output=True,
         min_frequency=30),  # ⇦ new
         cat_cols),
        ("sleep_spline", spline, ["sleep_minutes"]),
    ],
    remainder="passthrough",
    sparse_threshold=0.3,
)

def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "n_estimators":    trial.suggest_int("n_estimators", 100, 600, step=50),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight":trial.suggest_float("min_child_weight", 1, 10),
        "objective":       "binary:logistic",
        "eval_metric":     "auc",
        "tree_method":     "hist",
        "scale_pos_weight":neg_pos_ratio,
        "random_state":    RANDOM_SEED,
        "n_jobs": 4,
    }

    pipe = Pipeline([
        ("pre", pre),
        ("xgb", XGBClassifier(**params))
    ])
    pipe.fit(train_df, y_train)
    preds = pipe.predict_proba(test_df)[:, 1]
    return roc_auc_score(y_test, preds)

print("\n🔧  Starting XGBoost hyperparameter optimization...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n🏆  Best XGBoost trial: {study.best_value:.3f}")
print(f"🔧  Best parameters: {study.best_params}")

best_xgb = XGBClassifier(
    **study.best_params,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=neg_pos_ratio,
    random_state=RANDOM_SEED,
    n_jobs=4,
)
best_pipe = Pipeline([("pre", pre), ("xgb", best_xgb)])
best_pipe.fit(train_df, y_train)

probs_xgb = best_pipe.predict_proba(test_df)[:, 1]
auroc_xgb = roc_auc_score(y_test, probs_xgb)
auprc_xgb = average_precision_score(y_test, probs_xgb)

print(f"\n🎯  RESULTS COMPARISON:")
print("=" * 40)
print(f"📊  LightGBM:")
print(f"    AUROC = {roc_auc_score(y_test, probs_lgbm):.3f}")
print(f"    AUPRC = {average_precision_score(y_test, probs_lgbm):.3f}")
print(f"\n📊  XGBoost:")
print(f"    AUROC = {auroc_xgb:.3f}")
print(f"    AUPRC = {auprc_xgb:.3f}")

# Save the better model
if auprc_xgb >= average_precision_score(y_test, probs_lgbm):
    print(f"\n🏆  XGBoost performs better, saving XGBoost model")
    best_model = best_pipe
    best_auroc = auroc_xgb
    best_auprc = auprc_xgb
    best_params = study.best_params
    model_type = "XGBoost"
else:
    print(f"\n🏆  LightGBM performs better, saving LightGBM model")
    best_model = lgbm
    best_auroc = roc_auc_score(y_test, probs_lgbm)
    best_auprc = average_precision_score(y_test, probs_lgbm)
    best_params = lgbm.get_params()
    model_type = "LightGBM"

joblib.dump(best_model, OUT_PIPE)

# Save metadata
metadata = {
    "model_type": model_type,
    "auroc": float(best_auroc),  # Convert to float to avoid JSON serialization issues
    "auprc": float(best_auprc),
    "params": best_params,
    "lgbm_auroc": float(roc_auc_score(y_test, probs_lgbm)),
    "lgbm_auprc": float(average_precision_score(y_test, probs_lgbm)),
    "xgb_auroc": float(auroc_xgb),
    "xgb_auprc": float(auprc_xgb)
}

Path(OUT_PIPE.with_suffix(".meta.json")).write_text(
    json.dumps(metadata, indent=2)
)

print(f"\n💾  Saved {model_type} pipeline to: {OUT_PIPE}")
print(f"💾  Saved metadata to: {OUT_PIPE.with_suffix('.meta.json')}")
print(f"\n✨  Final best model: {model_type}")
print(f"📊  AUROC: {best_auroc:.3f}")
print(f"📊  AUPRC: {best_auprc:.3f}")