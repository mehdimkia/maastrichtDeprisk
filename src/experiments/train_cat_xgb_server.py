# src/train_cat_xgb_server.py  –  XGBoost  +  CatBoost

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from xgboost import XGBClassifier

# --------------------------------------------------------------- #
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
for alt in [
    Path("/Users/mkia/Downloads/SummerPredectiveModel/maastrichtDeprisk/Databases"),
    Path("../Databases"), Path("./Databases")
]:
    if not BASE.exists() and alt.exists():
        BASE = alt

TRAIN_PQ = BASE / "Week7_train_interactions.parquet"
TEST_PQ  = BASE / "Week7_test_interactions.parquet"
OUT_PIPE = BASE / "Week7_best_model.joblib"

N_TRIALS    = 100
RANDOM_SEED = 42
MIN_FREQ_OH = 30

# --------------------------------------------------------------- #
# Data
train_df = pd.read_parquet(TRAIN_PQ)
test_df  = pd.read_parquet(TEST_PQ)

y_train = train_df.pop("LD_PHQ9depr_event")
y_test  = test_df.pop("LD_PHQ9depr_event")

cat_cols      = train_df.select_dtypes("category").columns.tolist()
cat_idx       = [train_df.columns.get_loc(c) for c in cat_cols]
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# --------------------------------------------------------------- #
# 1.  XGBoost  (same preprocessing as before)
spline = SplineTransformer(
    degree=3,
    knots=np.array([[360], [420], [480], [540], [600]]),
    include_bias=False,
)

pre = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=True,
                              min_frequency=MIN_FREQ_OH),
         cat_cols),
        ("sleep_spline", spline, ["sleep_minutes"]),
    ],
    remainder="passthrough",
    sparse_threshold=0.3,
)

def xgb_objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":    trial.suggest_float("lr",   0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("depth", 3, 8),
        "n_estimators":     trial.suggest_int("n_est", 150, 800, step=50),
        "subsample":        trial.suggest_float("sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_float("mcw", 1, 10),
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "tree_method":      "hist",
        "scale_pos_weight": neg_pos_ratio,
        "random_state":     RANDOM_SEED,
        "n_jobs":           4,
    }
    pipe = Pipeline([("pre", pre), ("xgb", XGBClassifier(**params))])
    pipe.fit(train_df, y_train)
    pred = pipe.predict_proba(test_df)[:, 1]
    return roc_auc_score(y_test, pred)

print("🔧  Optimising XGBoost …")
xgb_study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"🏆  XGB best AUROC = {xgb_study.best_value:.3f}")

best_xgb = XGBClassifier(
    **xgb_study.best_params,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=neg_pos_ratio,
    random_state=RANDOM_SEED,
    n_jobs=4,
)
xgb_pipe = Pipeline([("pre", pre), ("xgb", best_xgb)])
xgb_pipe.fit(train_df, y_train)
probs_xgb = xgb_pipe.predict_proba(test_df)[:, 1]
auroc_xgb = roc_auc_score(y_test, probs_xgb)
auprc_xgb = average_precision_score(y_test, probs_xgb)

# --------------------------------------------------------------- #
# 2.  CatBoost  (no one‑hot needed)
print("\n🔧  Training CatBoost …")
train_pool = Pool(train_df, label=y_train, cat_features=cat_idx)
test_pool  = Pool(test_df,  label=y_test,  cat_features=cat_idx)

cat = CatBoostClassifier(
    depth=6,
    learning_rate=0.05,
    iterations=2000,
    eval_metric="AUC",
    loss_function="Logloss",
    l2_leaf_reg=3,
    random_state=RANDOM_SEED,
    verbose=False,
    auto_class_weights="SBalanced"  # handles imbalance
)
cat.fit(train_pool, eval_set=test_pool, early_stopping_rounds=200, verbose=False)

probs_cat = cat.predict_proba(test_pool)[:, 1]
auroc_cat = roc_auc_score(y_test, probs_cat)
auprc_cat = average_precision_score(y_test, probs_cat)

print(f"🔍  CatBoost AUROC = {auroc_cat:.3f}")
print(f"🔍  CatBoost AUPRC = {auprc_cat:.3f}")

# --------------------------------------------------------------- #
# Pick the winner
if auprc_cat > auprc_xgb:
    winner, win_type, win_auroc, win_auprc = cat, "CatBoost", auroc_cat, auprc_cat
else:
    winner, win_type, win_auroc, win_auprc = xgb_pipe, "XGBoost", auroc_xgb, auprc_xgb

joblib.dump(winner, OUT_PIPE)

meta = {
    "model_type": win_type,
    "auroc": float(win_auroc),
    "auprc": float(win_auprc),
    "xgb": {"auroc": auroc_xgb, "auprc": auprc_xgb,
            "params": xgb_study.best_params},
    "cat": {"auroc": auroc_cat, "auprc": auprc_cat,
            "iterations": int(cat.get_best_iteration())},
}
Path(OUT_PIPE.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))

print(f"\n✨  Best model: {win_type}")
print(f"📈  AUROC = {win_auroc:.3f}   •   AUPRC = {win_auprc:.3f}")
print(f"💾  Saved → {OUT_PIPE}")
