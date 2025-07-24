# src/feature_engineering_server.py

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
RAW_PKL        = BASE / "Week7.pkl"
FEATURES_PQ    = BASE / "Week7_features.parquet"
TRAIN_PQ       = BASE / "Week7_train.parquet"
TEST_PQ        = BASE / "Week7_test.parquet"

def load_raw() -> pd.DataFrame:
    if not RAW_PKL.exists():
        sys.exit(f"Raw pickle not found: {RAW_PKL}")
    return pd.read_pickle(RAW_PKL)

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    # Select only the SPSS logistic‑regression covariates + target
    cols = [
        "sleep_dur2",
        "frag_break_q4",
        "standstep_q4",
        "Age.ph1",
        "Sex",
        "N_Diabetes_WHO.ph1",
        "n_education_3cat.ph1",
        "marital_status.ph1",
        "smoking_3cat.ph1",
        "N_alcohol_cat.ph1",
        "bmi.ph1",
        "N_CVD.ph1",
        "dhd_sum_min_alc.ph1",
        "mvpatile",
        "LD_PHQ9depr_event"   # target: 0/1
    ]
    feats = df[cols].copy()
    # 1️⃣  Replace sentinel codes first
    feats = feats.replace({-777: pd.NA, -888: pd.NA, -999: pd.NA})
    # Cast the categorical columns
    cat_cols = [
        "sleep_dur2",
        "frag_break_q4",
        "standstep_q4",
        "Sex",
        "N_Diabetes_WHO.ph1",
        "n_education_3cat.ph1",
        "marital_status.ph1",
        "smoking_3cat.ph1",
        "N_alcohol_cat.ph1",
        "N_CVD.ph1",
        "mvpatile"
    ]
    for c in cat_cols:
        feats[c] = feats[c].astype("category")

    # Convert SPSS sentinels (−777/−888/−999) to true missing
    #feats = feats.replace({-777: pd.NA, -888: pd.NA, -999: pd.NA})

    return feats

def impute_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Identify numeric predictors (exclude the target)
    num_cols = train_df.select_dtypes("number").columns.drop("LD_PHQ9depr_event")

    # Median imputation + z‑score scaling
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols]  = imputer.transform(test_df[num_cols])
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols]  = scaler.transform(test_df[num_cols])

    return train_df, test_df

def main():
    df_raw  = load_raw()
    df_feat = build_feature_df(df_raw)

    # Persist the full feature set
    df_feat.to_parquet(FEATURES_PQ)
    print(f"✅ Features written: {FEATURES_PQ}")

    # Stratified 80/20 split
    train_df, test_df = train_test_split(
        df_feat,
        test_size=0.20,
        stratify=df_feat["LD_PHQ9depr_event"],
        random_state=42
    )

    train_df, test_df = impute_and_scale(train_df, test_df)

    # Write out splits
    train_df.to_parquet(TRAIN_PQ)
    test_df.to_parquet(TEST_PQ)
    print(f"💾 Train set: {TRAIN_PQ}")
    print(f"💾 Test  set: {TEST_PQ}")
    print("🎉 Feature engineering complete.")

if __name__ == "__main__":
    main()
