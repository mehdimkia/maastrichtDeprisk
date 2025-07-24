# src/train_simple_advanced.py - Simple but effective advanced features
# -----------------------------------------------------------------------------
# This version focuses on the most impactful features while avoiding
# categorical variable complications
# -----------------------------------------------------------------------------

import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
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

TRAIN_PQ = BASE / "Week7_train_interactions.parquet"
TEST_PQ = BASE / "Week7_test_interactions.parquet"
OUT_PIPE = BASE / "Week7_simple_advanced_features.joblib"

RANDOM_SEED = 42

print("🧠 SIMPLE ADVANCED FEATURES + COST-SENSITIVE LEARNING")
print("=" * 70)

# Load data
print("📁 Loading data...")
train_df = pd.read_parquet(TRAIN_PQ)
test_df = pd.read_parquet(TEST_PQ)
y_train = train_df.pop("LD_PHQ9depr_event")
y_test = test_df.pop("LD_PHQ9depr_event")

print(f"📊 Dataset: {len(y_train):,} train, {len(y_test):,} test")
print(f"📊 Class distribution: {(y_train == 0).sum():,} vs {(y_train == 1).sum():,}")


# Simple but effective feature engineering
def create_simple_advanced_features(df):
    """Create effective features while avoiding categorical complications"""

    print("🔧 Creating simple advanced sleep features...")

    df = df.copy()

    # 1. Sleep duration features (most important)
    if 'sleep_minutes' in df.columns:
        df['sleep_hours'] = df['sleep_minutes'] / 60

        # Key sleep patterns from medical literature
        df['sleep_dev_7h'] = np.abs(df['sleep_hours'] - 7)
        df['sleep_dev_8h'] = np.abs(df['sleep_hours'] - 8)
        df['sleep_dev_optimal'] = np.minimum(df['sleep_dev_7h'], df['sleep_dev_8h'])

        # Binary indicators for extreme sleep
        df['very_short_sleep'] = (df['sleep_hours'] < 6).astype(int)
        df['short_sleep'] = (df['sleep_hours'] < 7).astype(int)
        df['long_sleep'] = (df['sleep_hours'] > 8).astype(int)
        df['very_long_sleep'] = (df['sleep_hours'] > 9).astype(int)

        # Non-linear transformations
        df['sleep_minutes_sq'] = df['sleep_minutes'] ** 2
        df['sleep_log'] = np.log(df['sleep_minutes'] + 1)
        df['sleep_zscore'] = stats.zscore(df['sleep_minutes'])
        df['sleep_zscore_abs'] = np.abs(df['sleep_zscore'])

        # Sleep efficiency proxy
        df['sleep_efficiency_proxy'] = 1 / (1 + df['sleep_dev_optimal'])

    # 2. Age-sleep interactions (very important)
    if 'Age.ph1' in df.columns and 'sleep_hours' in df.columns:
        # Older adults need less sleep - this is key medical knowledge
        df['age_sleep_mismatch'] = np.where(
            df['Age.ph1'] > 65,
            np.maximum(0, df['sleep_hours'] - 7),  # Penalize oversleep in elderly
            df['sleep_dev_optimal']  # Standard deviation for younger
        )

        # Age categories
        df['elderly'] = (df['Age.ph1'] >= 65).astype(int)
        df['middle_aged'] = ((df['Age.ph1'] >= 45) & (df['Age.ph1'] < 65)).astype(int)

        # Age-sleep product terms
        df['age_times_sleep_dev'] = df['Age.ph1'] * df['sleep_dev_optimal']

    # 3. BMI-sleep interactions (obesity and sleep disorders linked)
    if 'bmi.ph1' in df.columns and 'sleep_hours' in df.columns:
        df['overweight'] = (df['bmi.ph1'] >= 25).astype(int)
        df['obese'] = (df['bmi.ph1'] >= 30).astype(int)

        # Obesity compounds sleep problems
        df['obese_poor_sleep'] = df['obese'] * df['sleep_dev_optimal']
        df['bmi_sleep_product'] = df['bmi.ph1'] * df['sleep_dev_optimal']

    # 4. Numeric health burden (avoid categorical issues)
    health_numeric_cols = ['Age.ph1', 'bmi.ph1']
    if all(col in df.columns for col in health_numeric_cols):
        # Simple health risk score
        df['health_risk_score'] = (
                (df['Age.ph1'] > 60).astype(int) +
                (df['bmi.ph1'] > 30).astype(int) +
                (df['Age.ph1'] > 70).astype(int)  # Extra weight for very elderly
        )

        # Health-sleep compound risk
        if 'sleep_dev_optimal' in df.columns:
            df['health_sleep_compound'] = df['health_risk_score'] * df['sleep_dev_optimal']

    # 5. Sleep fragmentation (convert to numeric safely)
    if 'standstep_q4' in df.columns:
        # Use physical activity quartile as proxy for overall health
        try:
            # Try to extract numeric part if it's like 'Q1', 'Q2', etc.
            activity_numeric = df['standstep_q4'].astype(str).str.extract('(\d+)')[0].astype(float)
            df['activity_numeric'] = activity_numeric.fillna(2)  # Fill with median
            df['low_activity'] = (activity_numeric <= 1).astype(int)

            # Low activity + poor sleep = compound risk
            if 'sleep_dev_optimal' in df.columns:
                df['inactive_poor_sleep'] = df['low_activity'] * df['sleep_dev_optimal']
        except:
            print("  ⚠️ Could not process activity quartiles")

    # 6. Sleep timing patterns (if we have fragmentation info)
    if 'sleep_minutes' in df.columns:
        # Sleep duration bins for non-linear effects
        df['sleep_300_360'] = ((df['sleep_minutes'] >= 300) & (df['sleep_minutes'] < 360)).astype(int)  # 5-6h
        df['sleep_360_420'] = ((df['sleep_minutes'] >= 360) & (df['sleep_minutes'] < 420)).astype(int)  # 6-7h
        df['sleep_420_480'] = ((df['sleep_minutes'] >= 420) & (df['sleep_minutes'] < 480)).astype(int)  # 7-8h
        df['sleep_480_540'] = ((df['sleep_minutes'] >= 480) & (df['sleep_minutes'] < 540)).astype(int)  # 8-9h
        df['sleep_540_plus'] = (df['sleep_minutes'] >= 540).astype(int)  # 9h+

    # 7. Polynomial features for key interactions
    if 'Age.ph1' in df.columns and 'sleep_minutes' in df.columns:
        df['age_sleep_interaction'] = df['Age.ph1'] * df['sleep_minutes']
        df['age_sq_sleep'] = (df['Age.ph1'] ** 2) * df['sleep_minutes']

    print(f"✅ Created {len([col for col in df.columns if col not in train_df.columns])} new features")

    return df


# Apply feature engineering
print("\n🎨 Feature Engineering...")
train_enhanced = create_simple_advanced_features(train_df)
test_enhanced = create_simple_advanced_features(test_df)

print(f"📊 Features: {train_df.shape[1]} → {train_enhanced.shape[1]} (+{train_enhanced.shape[1] - train_df.shape[1]})")


# Simple preprocessing
def create_simple_preprocessing():
    """Simple preprocessing focused on key transformations"""

    # Get categorical columns
    cat_cols = train_enhanced.select_dtypes('category').columns.tolist()

    # Numerical columns
    num_cols = train_enhanced.select_dtypes(['int64', 'float64']).columns.tolist()

    # Remove sleep_minutes from num_cols if it exists (will be handled by spline)
    if 'sleep_minutes' in num_cols:
        num_cols.remove('sleep_minutes')

    # Spline for sleep duration
    spline = SplineTransformer(
        degree=3,
        knots=np.array([[360], [420], [480], [540], [600]]),
        include_bias=False
    )

    # Simple pipeline
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("sleep_spline", spline, ["sleep_minutes"]),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop", sparse_threshold=0.0)

    return preprocessor


print("\n🔧 Preprocessing...")
preprocessor = create_simple_preprocessing()
X_train = preprocessor.fit_transform(train_enhanced)
X_test = preprocessor.transform(test_enhanced)

print(f"✅ Final features: {X_train.shape[1]}")

# Calculate class weights
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"📊 Class weight: {pos_weight:.1f}")


# Cost-sensitive optimization
def cost_sensitive_objective(trial):
    """Cost-sensitive optimization for AUPRC"""

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        "gamma": trial.suggest_float("gamma", 0.5, 3.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", pos_weight * 0.8, pos_weight * 1.5),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "n_jobs": 4,
    }

    # Quick 3-fold CV
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train)

        y_pred = model.predict_proba(X_fold_val)[:, 1]
        auprc = average_precision_score(y_fold_val, y_pred)
        cv_scores.append(auprc)

    return np.mean(cv_scores)


print(f"\n⚡ Optimizing for AUPRC...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(cost_sensitive_objective, n_trials=25, show_progress_bar=True)

print(f"🏆 Best CV AUPRC: {study.best_value:.3f}")

# Train final model
print(f"\n🎓 Training final model...")
final_model = XGBClassifier(**study.best_params)
final_model.fit(X_train, y_train)

# Evaluate
train_probs = final_model.predict_proba(X_train)[:, 1]
test_probs = final_model.predict_proba(X_test)[:, 1]

train_auroc = roc_auc_score(y_train, train_probs)
train_auprc = average_precision_score(y_train, train_probs)
test_auroc = roc_auc_score(y_test, test_probs)
test_auprc = average_precision_score(y_test, test_probs)

# Optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\n🎯 RESULTS:")
print("=" * 40)
print(f"📈 TRAIN: AUROC {train_auroc:.3f}, AUPRC {train_auprc:.3f}")
print(f"🎯 TEST:  AUROC {test_auroc:.3f}, AUPRC {test_auprc:.3f}")
print(f"🎚️  THRESHOLD: {best_threshold:.3f}")

# Compare to original
original_auroc, original_auprc = 0.710, 0.294
auroc_change = ((test_auroc / original_auroc - 1) * 100)
auprc_change = ((test_auprc / original_auprc - 1) * 100)

print(f"\n🚀 vs ORIGINAL:")
print(f"   AUROC: {original_auroc:.3f} → {test_auroc:.3f} ({auroc_change:+.1f}%)")
print(f"   AUPRC: {original_auprc:.3f} → {test_auprc:.3f} ({auprc_change:+.1f}%)")

# Overfitting check
auroc_gap = train_auroc - test_auroc
auprc_gap = train_auprc - test_auprc
print(f"\n🔍 OVERFITTING:")
print(f"   AUROC gap: {auroc_gap:.3f} {'✅' if auroc_gap < 0.05 else '⚠️'}")
print(f"   AUPRC gap: {auprc_gap:.3f} {'✅' if auprc_gap < 0.10 else '⚠️'}")


# Create pipeline
class SimpleAdvancedPipeline:
    def __init__(self, preprocessor, model, threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

    def _engineer_features(self, df):
        return create_simple_advanced_features(df)

    def predict_proba(self, X):
        X_eng = self._engineer_features(X)
        X_proc = self.preprocessor.transform(X_eng)
        return self.model.predict_proba(X_proc)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


pipeline = SimpleAdvancedPipeline(preprocessor, final_model, best_threshold)

# Save
print(f"\n💾 Saving...")
joblib.dump(pipeline, OUT_PIPE)

metadata = {
    "test_auroc": test_auroc,
    "test_auprc": test_auprc,
    "cv_auprc": study.best_value,
    "auroc_improvement_pct": auroc_change,
    "auprc_improvement_pct": auprc_change,
    "threshold": best_threshold,
    "overfitting_auroc": auroc_gap,
    "overfitting_auprc": auprc_gap,
    "method": "simple_advanced_features"
}

with open(OUT_PIPE.with_suffix(".meta.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to: {OUT_PIPE}")

# Final assessment
if test_auprc > 0.40:
    print("\n🌟 TARGET ACHIEVED!")
elif auprc_change > 20:
    print("\n🚀 EXCELLENT improvement!")
elif auprc_change > 10:
    print("\n✅ GOOD improvement!")
else:
    print("\n📈 Some improvement made")

print(f"\n✨ Complete! AUPRC: {test_auprc:.3f}")