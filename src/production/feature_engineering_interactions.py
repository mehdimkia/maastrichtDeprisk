# src/feature_engineering_interactions.py – v1 (adds vulnerability composite + interaction features)
# -----------------------------------------------------------------------------
# This is a *new* script so the original `feature_engineering_server.py` stays as‑is.
# File & output paths have been tweaked to write separate Parquet artefacts
# (…_interactions.parquet) and avoid collisions.
# -----------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# Paths – adjust BASE if you keep data elsewhere --------------------------------
# -----------------------------------------------------------------------------
BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
RAW_PKL = BASE / "Week7.pkl"
FEATURES_PQ = BASE / "Week7_features_interactions.parquet"
TRAIN_PQ = BASE / "Week7_train_interactions.parquet"
TEST_PQ = BASE / "Week7_test_interactions.parquet"

# -----------------------------------------------------------------------------
# Utility ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
SENTINELS = {-777: pd.NA, -888: pd.NA, -999: pd.NA}

def load_raw() -> pd.DataFrame:
    if not RAW_PKL.exists():
        sys.exit(f"❌ Raw pickle not found: {RAW_PKL}")
    return pd.read_pickle(RAW_PKL)

# -----------------------------------------------------------------------------
# Feature builder --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select predictors, clean sentinels, cast categoricals, add composites & interactions."""

    # 1 ▸ Keep only needed columns ------------------------------------------------
    cols = [
        # sleep exposure & modifiers
        "sleep_dur2",          # short / normal / long sleep category
        "frag_break_q4",       # fragmentation quartile
        "standstep_q4",        # MVPA quartile (already engineered upstream)
        # baseline covariates
        "Age.ph1", "Sex", "N_Diabetes_WHO.ph1",
        "n_education_3cat.ph1", "marital_status.ph1", "smoking_3cat.ph1",
        "N_alcohol_cat.ph1", "bmi.ph1", "N_CVD.ph1", "dhd_sum_min_alc.ph1", "mvpatile",
        # mental‑health / neuropathy flags
        "MINIlifedepr.ph1", "med_depression.ph1", "impaired_vibration_sense.ph1",
        # outcome
        "LD_PHQ9depr_event",
    ]
    feats = df[cols].copy()

    # 2 ▸ Sentinel → NA ----------------------------------------------------------
    feats.replace(SENTINELS, inplace=True)

    # 3 ▸ Harmonise categorical labels ------------------------------------------
    sleep_map = {2: "7-9 h", 1: "<7 h", 3: ">=9 h"}
    feats["sleep_dur2"] = feats["sleep_dur2"].map(sleep_map).astype("category")

    # frag_break_q4: 1–4 → Q1–Q4
    if feats["frag_break_q4"].notna().any():
        frag_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
        feats["frag_break_q4"] = feats["frag_break_q4"].map(frag_map).astype("category")

    # Binary flags (0/1 ▸ "No"/"Yes") – tolerate NA
    for flag in ("MINIlifedepr.ph1", "med_depression.ph1", "impaired_vibration_sense.ph1"):
        feats[flag] = feats[flag].map({0: "No", 1: "Yes"}).astype("category")

    # Cast pre‑existing categorical variables (numeric codes → category)
    cat_cols = [
        "sleep_dur2", "frag_break_q4", "standstep_q4", "Sex", "N_Diabetes_WHO.ph1",
        "n_education_3cat.ph1", "marital_status.ph1", "smoking_3cat.ph1",
        "N_alcohol_cat.ph1", "N_CVD.ph1", "mvpatile",
    ] + ["MINIlifedepr.ph1", "med_depression.ph1", "impaired_vibration_sense.ph1"]
    for c in cat_cols:
        if c in feats.columns:
            feats[c] = feats[c].astype("category")

    # 4 ▸ Composite vulnerability flag ------------------------------------------
    feats["any_vuln"] = (
        feats[["MINIlifedepr.ph1", "med_depression.ph1", "impaired_vibration_sense.ph1"]]
        .eq("Yes").any(axis=1)
        .map({True: "Yes", False: "No"})
        .astype("category")
    )

    # 5 ▸ Interaction features ---------------------------------------------------
    def cross_cat(a: str, b: str, name: str) -> None:
        """Make categorical cross by concatenation preserving NAs."""
        feats[name] = (
            feats[a].astype(str).where(~feats[a].isna(), "NA") + "__" +
            feats[b].astype(str).where(~feats[b].isna(), "NA")
        ).replace("NA__NA", pd.NA).astype("category")

    cross_cat("sleep_dur2", "any_vuln", "sleep_anyvuln")
    cross_cat("sleep_dur2", "med_depression.ph1", "sleep_med")
    # cross_cat("sleep_dur2", "mvpatile", "sleep_mvpa")  # optional extra

    # --- Continuous sleep duration (minutes) and non‑linear terms -------------
    feats["sleep_minutes"] = df["mean_inbed_min_night_t.ph1"]  # raw minutes

    # centre at 7 h (420 min) so the linear term is interpretable
    feats["sleep_minutes_ctr"] = feats["sleep_minutes"] - 420

    # simple quadratic J‑shape term
    feats["sleep_minutes_sq"] = feats["sleep_minutes_ctr"] ** 2


    return feats

# -----------------------------------------------------------------------------
# Imputation & scaling ---------------------------------------------------------
# -----------------------------------------------------------------------------

def impute_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Median‑impute and z‑scale numeric predictors (target excluded)."""
    num_cols = train_df.select_dtypes("number").columns.drop("LD_PHQ9depr_event")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    return train_df, test_df

# -----------------------------------------------------------------------------
# CLI entry‑point --------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    df_raw = load_raw()
    df_feat = build_feature_df(df_raw)

    df_feat.to_parquet(FEATURES_PQ)
    print(f"✅ Features (interaction set) written → {FEATURES_PQ}")

    train_df, test_df = train_test_split(
        df_feat,
        test_size=0.20,
        stratify=df_feat["LD_PHQ9depr_event"],
        random_state=42,
    )
    train_df, test_df = impute_and_scale(train_df, test_df)

    train_df.to_parquet(TRAIN_PQ)
    test_df.to_parquet(TEST_PQ)
    print(f"💾 Train set → {TRAIN_PQ}\n💾 Test set  → {TEST_PQ}")
    print("🎉 Interaction feature engineering complete.")


if __name__ == "__main__":
    # This block runs only when the file is executed directly:
    #   $ python feature_engineering_interactions.py
    # It prevents `main()` from firing if the module is merely imported.
    main()
