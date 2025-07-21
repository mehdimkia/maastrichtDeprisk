# src/train_xgb_onehot_server.py  (fixed)

import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer
from xgboost import XGBClassifier

# ------------------------------------------------------------------
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
""" Base Training Features
TRAIN_PQ = BASE / "Week7_train.parquet"
TEST_PQ  = BASE / "Week7_test.parquet"
OUT_PIPE = BASE / "Week7_xgb_onehot.joblib"
"""
#Training features with interactions included
TRAIN_PQ = BASE / "Week7_train_interactions.parquet"
TEST_PQ  = BASE / "Week7_test_interactions.parquet"

# >>> write the model under a new name so nothing gets overwritten
OUT_PIPE = BASE / "Week7_xgb_onehot_interactions.joblib"

N_TRIALS = 50
RANDOM_SEED = 42

train_df = pd.read_parquet(TRAIN_PQ)
test_df  = pd.read_parquet(TEST_PQ)
y_train  = train_df.pop("LD_PHQ9depr_event")
y_test   = test_df.pop("LD_PHQ9depr_event")

cat_cols = train_df.select_dtypes("category").columns.tolist()
neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# ---------- fix here ▼ -------------------------------------------------------
spline = SplineTransformer(
    degree=3,
    knots=np.array([[360],   # 6 h
                    [420],   # 7 h
                    [480],   # 8 h
                    [540],   # 9 h
                    [600]]), #10 h
    include_bias=False,
)

pre = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("sleep_spline", spline, ["sleep_minutes"]),
    ],
    remainder="passthrough",
    sparse_threshold=0.3,
)
# ------------------------------------------------------------------------------

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

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

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

probs  = best_pipe.predict_proba(test_df)[:, 1]
auroc  = roc_auc_score(y_test, probs)
auprc  = average_precision_score(y_test, probs)

print(f"\n🎯  AUROC = {auroc:.3f}")
print(f"🎯  AUPRC = {auprc:.3f}")

joblib.dump(best_pipe, OUT_PIPE)
Path(OUT_PIPE.with_suffix(".meta.json")).write_text(
    json.dumps({"auroc": auroc, "auprc": auprc,
                "params": study.best_params}, indent=2)
)
print(f"\n💾  Saved pipeline to: {OUT_PIPE}")
